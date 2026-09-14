from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from rag_app.models import Document
from rag_app.services import indexing, llm
from rag_app.services.extraction import Section

DEMO_DOCS_DIR = Path(__file__).resolve().parents[2] / "demo_docs"


class Command(BaseCommand):
    help = "Index the sample documents in rag_app/demo_docs/ (skips titles already present)."

    def handle(self, *args, **options):
        paths = sorted(DEMO_DOCS_DIR.glob("*.md"))
        if not paths:
            raise CommandError(f"No demo documents found in {DEMO_DOCS_DIR}")

        for path in paths:
            text = path.read_text(encoding="utf-8")
            # The first line is "# Title"; the rest is the body.
            first_line, _, body = text.partition("\n")
            title = first_line.lstrip("# ").strip() or path.stem

            if Document.objects.filter(title=title).exists():
                self.stdout.write(f"skip  {title} (already indexed)")
                continue

            try:
                _, chunk_count = indexing.index_document(
                    title, [Section(body)], source=Document.Source.PLAIN, filename=path.name
                )
            except (indexing.NothingToIndex, llm.LLMConfigError, llm.LLMRequestError) as e:
                raise CommandError(f"Indexing failed for {title}: {e}") from e
            self.stdout.write(self.style.SUCCESS(f"added {title} ({chunk_count} chunks)"))
