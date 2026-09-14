import io
import json
import tempfile
from pathlib import Path
from unittest import mock

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase, TestCase

from ..chunking import chunk_text
from ..services import evaluation, llm
from ..services.evaluation import EvalQuestion, EvalRow
from .helpers import fake_embed

DEMO_DOCS = Path(__file__).resolve().parents[1] / "demo_docs"


def question(**overrides):
    base = {"id": "q", "question": "Q?", "evidence": ["needle"], "facts": [["x"]], "answerable": True}
    return EvalQuestion(**{**base, **overrides})


class MetricTests(SimpleTestCase):
    def test_first_hit_rank(self):
        texts = ["nothing here", "The NEEDLE  is\nhere", "needle again"]
        self.assertEqual(evaluation.first_hit_rank(texts, ["needle is here"]), 2)
        self.assertIsNone(evaluation.first_hit_rank(texts, ["missing"]))

    def test_fact_recall_counts_groups_with_alternatives(self):
        facts = [["835"], ["remittance advice", "ERA"], ["payer"]]
        self.assertAlmostEqual(evaluation.fact_recall("The 835 (ERA) explains payment.", facts), 2 / 3)
        self.assertEqual(evaluation.fact_recall("anything", []), 1.0)

    def test_fact_recall_ignores_dash_style(self):
        self.assertEqual(evaluation.fact_recall("It replaced the UB‑04 form", [["UB-04"]]), 1.0)

    def test_abstention(self):
        self.assertTrue(evaluation.is_abstention("I don’t know based on the documents."))
        self.assertTrue(evaluation.is_abstention("The context does not mention the price."))
        self.assertFalse(evaluation.is_abstention("The 835 is the remittance advice."))

    def test_summarize(self):
        rows = [
            EvalRow(question(), [], hit_rank=1, fact_recall=1.0, faithful=True, latency_ms=100),
            EvalRow(question(), [], hit_rank=2, fact_recall=0.5, faithful=False, latency_ms=300),
            EvalRow(question(), [], hit_rank=None, fact_recall=0.0, latency_ms=200),
            EvalRow(question(answerable=False, evidence=[], facts=[]), [], hit_rank=None, abstained=True, latency_ms=0),
        ]
        summary = evaluation.summarize(rows, top_k=3)
        self.assertEqual(summary["hit@3"], round(2 / 3, 4))
        self.assertEqual(summary["mrr"], round((1 + 0.5 + 0) / 3, 4))
        self.assertEqual(summary["fact_recall"], 0.5)
        self.assertEqual(summary["faithfulness"], 0.5)
        self.assertEqual(summary["abstention"], 1.0)
        self.assertEqual(summary["avg_latency_ms"], 150)

    def test_judge_parses_json_and_rejects_bad_replies(self):
        with mock.patch.object(llm, "chat_json", return_value={"faithful": False, "unsupported_claims": ["a", "b"]}):
            self.assertEqual(evaluation.judge_faithfulness("q", "ctx", "ans"), (False, "a; b"))
        with mock.patch.object(llm, "chat_json", return_value={"faithful": "yes"}):
            with self.assertRaises(llm.LLMRequestError):
                evaluation.judge_faithfulness("q", "ctx", "ans")


class DatasetTests(SimpleTestCase):
    """Guards the dataset against drifting away from the demo documents."""

    def setUp(self):
        self.questions = evaluation.load_dataset()
        self.docs = {}
        for path in DEMO_DOCS.glob("*.md"):
            title, _, body = path.read_text(encoding="utf-8").partition("\n")
            self.docs[title.lstrip("# ").strip()] = body

    def test_size_and_mix(self):
        self.assertEqual(sum(q.answerable for q in self.questions), 20)
        self.assertGreaterEqual(sum(not q.answerable for q in self.questions), 3)

    def test_evidence_is_quoted_from_the_expected_document_and_fits_in_one_chunk(self):
        for q in self.questions:
            if not q.answerable:
                continue
            with self.subTest(q.id):
                body = self.docs[q.expected_document]
                self.assertIsNotNone(evaluation.first_hit_rank([body], q.evidence))
                # A phrase split across chunks could never be a hit.
                self.assertIsNotNone(evaluation.first_hit_rank(chunk_text(body), q.evidence))


@mock.patch.object(llm, "embed_texts", side_effect=fake_embed)
class EvaluateCommandTests(TestCase):
    def test_requires_indexed_documents(self, _embed):
        with self.assertRaisesRegex(CommandError, "seed_demo"):
            call_command("evaluate_rag", "--retrieval-only", stdout=io.StringIO(), stderr=io.StringIO())

    def test_full_run_writes_a_report(self, _embed):
        call_command("seed_demo", stdout=io.StringIO())
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            llm, "chat", return_value="I don't know."
        ), mock.patch.object(llm, "chat_json", return_value={"faithful": True, "unsupported_claims": []}):
            report_path = Path(tmp) / "report.json"
            call_command(
                "evaluate_rag",
                "--configs", "vector,hybrid",
                "--limit", "4",
                "--output", str(report_path),
                stdout=out,
                stderr=io.StringIO(),
            )
            report = json.loads(report_path.read_text())

        self.assertEqual(set(report["configs"]), {"vector", "hybrid"})
        summary = report["configs"]["hybrid"]["summary"]
        self.assertEqual(summary["questions"], 4)
        self.assertEqual(summary["faithfulness"], 1.0)
        self.assertEqual(summary["fact_recall"], 0.0)  # "I don't know" has none of the facts
        self.assertIn("hit@3", out.getvalue())

    def test_unknown_config(self, _embed):
        with self.assertRaisesRegex(CommandError, "Unknown config"):
            call_command("evaluate_rag", "--configs", "magic", stdout=io.StringIO())
