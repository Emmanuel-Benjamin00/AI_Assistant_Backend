from unittest import mock

from django.core.cache import cache
from django.test import RequestFactory, SimpleTestCase, TestCase, override_settings
from rest_framework.test import APIClient

from ..chunking import chunk_text
from ..models import Chunk, Document
from ..services import llm
from ..throttling import client_ip, strip_port
from .helpers import fake_embed


class ChunkingTests(SimpleTestCase):
    def test_empty_text_has_no_chunks(self):
        self.assertEqual(chunk_text("   \n\n  "), [])

    def test_small_paragraphs_are_packed_together(self):
        chunks = chunk_text("First paragraph.\n\nSecond paragraph.", max_chars=100, overlap_chars=10)
        self.assertEqual(chunks, ["First paragraph.\n\nSecond paragraph."])

    def test_paragraphs_start_a_new_chunk_when_full(self):
        a, b = "a" * 60, "b" * 60
        self.assertEqual(chunk_text(f"{a}\n\n{b}", max_chars=100, overlap_chars=10), [a, b])

    def test_long_paragraph_is_split_within_limit_with_overlap(self):
        sentences = " ".join(f"Sentence number {i} has some words in it." for i in range(60))
        chunks = chunk_text(sentences, max_chars=300, overlap_chars=60)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 300 for c in chunks))
        # Every sentence survives in at least one chunk.
        for i in range(60):
            self.assertTrue(any(f"Sentence number {i} " in c for c in chunks), i)

    def test_text_without_spaces_still_terminates(self):
        chunks = chunk_text("x" * 1000, max_chars=300, overlap_chars=50)
        self.assertTrue(all(len(c) <= 300 for c in chunks))
        self.assertEqual(chunks[-1][-1], "x")

    def test_rejects_overlap_too_large(self):
        with self.assertRaises(ValueError):
            chunk_text("text", max_chars=100, overlap_chars=60)


class ClientIPTests(SimpleTestCase):
    def test_strip_port(self):
        self.assertEqual(strip_port("203.0.113.5:51234"), "203.0.113.5")
        self.assertEqual(strip_port("203.0.113.5"), "203.0.113.5")
        self.assertEqual(strip_port("[2001:db8::1]:443"), "2001:db8::1")
        self.assertEqual(strip_port("2001:db8::1"), "2001:db8::1")

    def test_uses_last_forwarded_entry(self):
        request = RequestFactory().get("/", HTTP_X_FORWARDED_FOR="1.1.1.1, 203.0.113.5:40000")
        self.assertEqual(client_ip(request), "203.0.113.5")


@mock.patch.object(llm, "chat", return_value="Support is 9 to 5 [1].")
@mock.patch.object(llm, "embed_texts", side_effect=fake_embed)
class ApiTests(TestCase):
    def setUp(self):
        cache.clear()
        self.client = APIClient()

    def test_ingest_list_and_ask(self, _embed, _chat):
        r = self.client.post(
            "/api/documents/",
            {"title": "Facts", "text": "Support hours are 9 to 5.\n\nRefunds within 30 days."},
            format="json",
        )
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.json()["chunks_created"], 1)
        self.assertEqual(Chunk.objects.count(), 1)

        r = self.client.get("/api/documents/")
        self.assertEqual(r.json()[0]["title"], "Facts")

        r = self.client.post("/api/ask/", {"question": "When is support?"}, format="json")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["answer"], "Support is 9 to 5 [1].")
        self.assertEqual(body["sources"][0]["document_title"], "Facts")
        self.assertIn("similarity", body["sources"][0])

    def test_ask_without_documents(self, _embed, chat):
        r = self.client.post("/api/ask/", {"question": "Anything?"}, format="json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["sources"], [])
        chat.assert_not_called()

    def test_rejects_oversized_question(self, _embed, _chat):
        r = self.client.post("/api/ask/", {"question": "x" * 2001}, format="json")
        self.assertEqual(r.status_code, 400)

    def test_provider_failure_saves_nothing(self, embed, _chat):
        embed.side_effect = llm.LLMRequestError("boom", 500)
        r = self.client.post("/api/documents/", {"title": "T", "text": "Body"}, format="json")
        self.assertEqual(r.status_code, 502)
        self.assertFalse(Document.objects.exists())

    def test_missing_provider_config_is_503(self, embed, _chat):
        embed.side_effect = llm.LLMConfigError("no key")
        r = self.client.post("/api/ask/", {"question": "Q"}, format="json")
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("no key", r.json()["detail"])

    @override_settings(INGEST_OPEN=False, INGEST_API_KEY="secret")
    def test_locked_ingest_needs_key(self, _embed, _chat):
        payload = {"title": "T", "text": "Body"}
        self.assertEqual(self.client.post("/api/documents/", payload, format="json").status_code, 403)
        r = self.client.post("/api/documents/", payload, format="json", HTTP_X_INGEST_KEY="wrong")
        self.assertEqual(r.status_code, 403)
        r = self.client.post("/api/documents/", payload, format="json", HTTP_X_INGEST_KEY="secret")
        self.assertEqual(r.status_code, 201)
        # Listing stays public.
        self.assertEqual(self.client.get("/api/documents/").status_code, 200)

    @override_settings(INGEST_OPEN=False, INGEST_API_KEY="")
    def test_locked_ingest_without_configured_key_rejects_all(self, _embed, _chat):
        r = self.client.post(
            "/api/documents/", {"title": "T", "text": "Body"}, format="json", HTTP_X_INGEST_KEY=""
        )
        self.assertEqual(r.status_code, 403)

    def test_ask_is_rate_limited(self, _embed, _chat):
        from rest_framework.throttling import SimpleRateThrottle

        with mock.patch.dict(SimpleRateThrottle.THROTTLE_RATES, {"ask": "2/hour"}):
            codes = [
                self.client.post("/api/ask/", {"question": "Q"}, format="json").status_code
                for _ in range(3)
            ]
        self.assertEqual(codes, [200, 200, 429])

    def test_health(self, _embed, _chat):
        r = self.client.get("/api/health/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})


class LLMRetryTests(SimpleTestCase):
    def setUp(self):
        patcher = mock.patch.multiple(llm, LLM_PROVIDER="openai", OPENAI_API_KEY="k")
        patcher.start()
        self.addCleanup(patcher.stop)
        sleep = mock.patch.object(llm.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    def _response(self, status, json=None):
        import httpx

        return httpx.Response(status, json=json or {}, request=httpx.Request("POST", "https://x"))

    def test_retries_then_succeeds(self):
        ok = self._response(200, {"data": [{"index": 0, "embedding": [1.0]}]})
        with mock.patch.object(llm._client, "post", side_effect=[self._response(429), ok]) as post:
            self.assertEqual(llm.embed_texts(["a"]), [[1.0]])
        self.assertEqual(post.call_count, 2)

    def test_client_error_is_not_retried(self):
        with mock.patch.object(llm._client, "post", return_value=self._response(400)) as post:
            with self.assertRaises(llm.LLMRequestError):
                llm.chat("s", "u")
        self.assertEqual(post.call_count, 1)

    def test_batches_embedding_requests(self):
        def reply(url, json, headers):
            return self._response(
                200, {"data": [{"index": i, "embedding": [float(i)]} for i in range(len(json["input"]))]}
            )

        with mock.patch.object(llm, "EMBED_BATCH_SIZE", 2), mock.patch.object(
            llm._client, "post", side_effect=reply
        ) as post:
            self.assertEqual(len(llm.embed_texts(["a", "b", "c"])), 3)
        self.assertEqual(post.call_count, 2)
