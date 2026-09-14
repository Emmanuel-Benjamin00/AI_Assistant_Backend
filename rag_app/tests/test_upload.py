from unittest import mock

from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings
from rest_framework.test import APIClient

from ..models import Chunk, Document
from ..services import extraction, llm
from ..services.extraction import ExtractionError, Section, extract_sections
from ..services.indexing import plan_chunks
from .helpers import fake_embed, make_docx, make_pdf


class ExtractionTests(SimpleTestCase):
    def test_pdf_gives_one_section_per_page(self):
        source, sections = extract_sections("r.pdf", make_pdf(["First page.", "Second page."]))
        self.assertEqual(source, "pdf")
        self.assertEqual(sections, [Section("First page.", 1), Section("Second page.", 2)])

    def test_pdf_blank_pages_are_dropped(self):
        _, sections = extract_sections("r.pdf", make_pdf(["", "Only text."]))
        self.assertEqual(sections, [Section("Only text.", 2)])

    def test_pdf_without_text_mentions_ocr(self):
        with self.assertRaisesRegex(ExtractionError, "OCR"):
            extract_sections("scan.pdf", make_pdf(["", ""]))

    def test_docx_includes_paragraphs_and_tables(self):
        data = make_docx(["Intro paragraph.", "Second paragraph."], table=[["Code", "Meaning"], ["CO", "Contractual"]])
        source, sections = extract_sections("notes.DOCX", data)
        self.assertEqual(source, "docx")
        self.assertEqual(
            sections[0].text,
            "Intro paragraph.\n\nSecond paragraph.\n\nCode | Meaning\n\nCO | Contractual",
        )

    def test_plain_text_and_markdown(self):
        self.assertEqual(extract_sections("a.txt", "héllo".encode())[1], [Section("héllo")])
        self.assertEqual(extract_sections("a.md", b"\xef\xbb\xbf# Title")[1], [Section("# Title")])

    def test_rejects_non_utf8_text(self):
        with self.assertRaisesRegex(ExtractionError, "UTF-8"):
            extract_sections("a.txt", b"\xff\xfe\x00bad")

    def test_rejects_unsupported_extension(self):
        with self.assertRaisesRegex(ExtractionError, "Unsupported"):
            extract_sections("a.exe", b"MZ")

    def test_rejects_content_that_does_not_match_extension(self):
        with self.assertRaisesRegex(ExtractionError, "not a valid PDF"):
            extract_sections("fake.pdf", b"hello")
        with self.assertRaisesRegex(ExtractionError, "not a valid DOCX"):
            extract_sections("fake.docx", b"%PDF-1.4")

    def test_damaged_pdf(self):
        with self.assertRaisesRegex(ExtractionError, "could not be read"):
            extract_sections("broken.pdf", b"%PDF-1.4 garbage")

    def test_size_and_text_limits(self):
        with mock.patch.object(extraction, "MAX_UPLOAD_BYTES", 10):
            with self.assertRaisesRegex(ExtractionError, "larger than"):
                extract_sections("a.txt", b"x" * 11)
        with mock.patch.object(extraction, "MAX_EXTRACTED_CHARS", 5):
            with self.assertRaisesRegex(ExtractionError, "more than"):
                extract_sections("a.txt", b"x" * 6)


class PlanChunksTests(SimpleTestCase):
    def test_chunks_keep_page_numbers_and_a_global_index(self):
        planned = plan_chunks([Section("Page one.", 1), Section("Page two.", 2), Section("No page.")])
        self.assertEqual(
            [p.metadata for p in planned],
            [{"index": 0, "page": 1}, {"index": 1, "page": 2}, {"index": 2}],
        )


@mock.patch.object(llm, "embed_texts", side_effect=fake_embed)
class UploadApiTests(TestCase):
    def setUp(self):
        cache.clear()
        self.client = APIClient()

    def upload(self, name, data, **extra):
        return self.client.post(
            "/api/documents/upload/",
            {"file": SimpleUploadedFile(name, data), **extra},
            format="multipart",
        )

    def test_pdf_upload_indexes_pages(self, _embed):
        r = self.upload("Claims Guide.pdf", make_pdf(["The 835 is the remittance.", "The 837 is the claim."]))
        self.assertEqual(r.status_code, 201, r.content)
        self.assertEqual(r.json()["chunks_created"], 2)
        doc = Document.objects.get()
        self.assertEqual((doc.title, doc.source, doc.filename), ("Claims Guide", "pdf", "Claims Guide.pdf"))
        self.assertEqual(
            sorted(Chunk.objects.values_list("metadata__page", flat=True)),
            [1, 2],
        )

    def test_title_field_overrides_filename(self, _embed):
        r = self.upload("x.txt", b"Some text", title="My Notes")
        self.assertEqual(r.status_code, 201)
        self.assertEqual(Document.objects.get().title, "My Notes")

    def test_bad_file_is_400_and_saves_nothing(self, embed):
        r = self.upload("scan.pdf", make_pdf([""]))
        self.assertEqual(r.status_code, 400)
        self.assertIn("OCR", r.json()["detail"])
        self.assertFalse(Document.objects.exists())
        embed.assert_not_called()

    def test_oversized_upload_is_413(self, _embed):
        with mock.patch("rag_app.views.MAX_UPLOAD_BYTES", 4):
            r = self.upload("a.txt", b"too long")
        self.assertEqual(r.status_code, 413)

    def test_missing_file_is_400(self, _embed):
        r = self.client.post("/api/documents/upload/", {"title": "T"}, format="multipart")
        self.assertEqual(r.status_code, 400)

    @override_settings(INGEST_OPEN=False, INGEST_API_KEY="secret")
    def test_upload_uses_the_ingest_lock(self, _embed):
        self.assertEqual(self.upload("a.txt", b"text").status_code, 403)
        r = self.client.post(
            "/api/documents/upload/",
            {"file": SimpleUploadedFile("a.txt", b"text")},
            format="multipart",
            HTTP_X_INGEST_KEY="secret",
        )
        self.assertEqual(r.status_code, 201)

    def test_document_list_shows_source(self, _embed):
        self.upload("a.md", b"# Hi\n\nBody")
        self.assertEqual(self.client.get("/api/documents/").json()[0]["source"], "plain")
