"""
Evaluate retrieval and answers on rag_app/eval/dataset.json, comparing search configurations.

    python manage.py seed_demo                        # index the documents the dataset asks about
    python manage.py evaluate_rag --retrieval-only    # hit@k and MRR only: one embedding per question
    python manage.py evaluate_rag                     # + answers, fact recall, abstention, LLM-judged faithfulness
    python manage.py evaluate_rag --configs vector,hybrid --top-k 3 --output eval-report.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from rag_app.models import Chunk, Document
from rag_app.services import evaluation, langchain_rag, llm, rag, retrieval

# name -> (retrieval mode, rerank, pipeline)
CONFIGS = {
    "vector": (retrieval.VECTOR, False, "core"),
    "hybrid": (retrieval.HYBRID, False, "core"),
    "hybrid+rerank": (retrieval.HYBRID, True, "core"),
    "langchain": (retrieval.HYBRID, False, "langchain"),
}


class Command(BaseCommand):
    help = "Score RAG retrieval (hit@k, MRR) and answers (fact recall, faithfulness, abstention)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--configs",
            default="vector,hybrid,hybrid+rerank",
            help=f"Comma-separated, from: {', '.join(CONFIGS)}",
        )
        parser.add_argument("--top-k", type=int, default=3)
        parser.add_argument("--dataset", type=Path, default=evaluation.DATASET_PATH)
        parser.add_argument("--retrieval-only", action="store_true", help="Skip answers and the judge.")
        parser.add_argument("--no-judge", action="store_true", help="Generate answers but skip the LLM judge.")
        parser.add_argument("--limit", type=int, help="Only the first N questions (quick smoke run).")
        parser.add_argument("--output", type=Path, help="Write a JSON report with every answer.")

    def handle(self, *args, **opts):
        configs = [c.strip() for c in opts["configs"].split(",") if c.strip()]
        unknown = [c for c in configs if c not in CONFIGS]
        if unknown:
            raise CommandError(f"Unknown config(s): {', '.join(unknown)}. Choose from {', '.join(CONFIGS)}.")
        if opts["retrieval_only"] and "langchain" in configs:
            raise CommandError("The langchain config only changes answer generation; drop --retrieval-only.")

        questions = evaluation.load_dataset(opts["dataset"])[: opts["limit"]]
        top_k = opts["top_k"]
        if not Chunk.objects.exists():
            raise CommandError("No indexed chunks. Run `python manage.py seed_demo` first.")
        self._warn_missing_documents(questions)

        # Embed each question once and reuse the vector for every core config.
        try:
            vectors = llm.embed_texts([q.question for q in questions])
        except (llm.LLMConfigError, llm.LLMRequestError) as e:
            raise CommandError(f"Embedding failed: {e}") from e

        report = {"top_k": top_k, "dataset": str(opts["dataset"]), "configs": {}}
        for name in configs:
            self.stdout.write(f"\n== {name}")
            rows = [
                self._evaluate_one(name, q, vec, top_k, opts)
                for q, vec in zip(questions, vectors, strict=True)
            ]
            summary = evaluation.summarize(rows, top_k)
            report["configs"][name] = {"summary": summary, "rows": [self._row_json(r) for r in rows]}
            self.stdout.write(json.dumps(summary))

        self._print_table(report, top_k)
        if opts["output"]:
            opts["output"].write_text(json.dumps(report, indent=2), encoding="utf-8")
            self.stdout.write(f"\nReport written to {opts['output']}")

    def _evaluate_one(self, name, q, vec, top_k, opts) -> evaluation.EvalRow:
        mode, rerank, pipeline = CONFIGS[name]
        started = time.perf_counter()
        try:
            if pipeline == "langchain":
                result = langchain_rag.ask(q.question, top_k, mode, rerank)
                results, answer = result["results"], result["answer"]
            else:
                results = retrieval.retrieve_with_vector(q.question, vec, top_k, mode=mode, rerank=rerank)
                answer = None if opts["retrieval_only"] else rag.answer(q.question, results)
        except (llm.LLMConfigError, llm.LLMRequestError) as e:
            raise CommandError(f"{name} / {q.id}: {e}") from e
        latency_ms = (time.perf_counter() - started) * 1000

        texts = [r.chunk.text for r in results]
        row = evaluation.EvalRow(
            question=q,
            retrieved_texts=texts,
            hit_rank=evaluation.first_hit_rank(texts, q.evidence) if q.answerable else None,
            latency_ms=round(latency_ms, 1),
        )
        if answer is not None:
            row.answer = answer
            if q.answerable:
                row.fact_recall = evaluation.fact_recall(answer, q.facts)
            else:
                row.abstained = evaluation.is_abstention(answer)
            if not opts["no_judge"] and results:
                try:
                    row.faithful, row.judge_notes = evaluation.judge_faithfulness(
                        q.question, rag.build_context(results), answer
                    )
                except llm.LLMRequestError as e:
                    self.stderr.write(f"judge failed for {q.id}: {e}")

        mark = "hit@%s" % row.hit_rank if row.hit_rank else ("miss" if q.answerable else "n/a")
        self.stdout.write(f"  {q.id:<26} {mark:<7} {latency_ms:7.0f} ms")
        return row

    def _warn_missing_documents(self, questions):
        titles = {q.expected_document for q in questions if q.expected_document}
        indexed = set(Document.objects.filter(title__in=titles).values_list("title", flat=True))
        for title in sorted(titles - indexed):
            self.stderr.write(self.style.WARNING(f"Document not indexed: {title!r} (run seed_demo)"))

    def _print_table(self, report, top_k):
        columns = [f"hit@{top_k}", "mrr", "fact_recall", "faithfulness", "abstention", "avg_latency_ms"]
        self.stdout.write("\n" + f"{'config':<16}" + "".join(f"{c:>15}" for c in columns))
        for name, data in report["configs"].items():
            cells = []
            for c in columns:
                value = data["summary"][c]
                cells.append(f"{'-':>15}" if value is None else f"{value:>15}")
            self.stdout.write(f"{name:<16}" + "".join(cells))

    @staticmethod
    def _row_json(row: evaluation.EvalRow) -> dict:
        return {
            "id": row.question.id,
            "question": row.question.question,
            "answerable": row.question.answerable,
            "hit_rank": row.hit_rank,
            "answer": row.answer,
            "fact_recall": row.fact_recall,
            "abstained": row.abstained,
            "faithful": row.faithful,
            "judge_notes": row.judge_notes,
            "latency_ms": row.latency_ms,
            "retrieved": [t[:200] for t in row.retrieved_texts],
        }
