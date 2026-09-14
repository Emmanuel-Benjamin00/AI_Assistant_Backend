from types import SimpleNamespace
from unittest import mock

from django.core.cache import cache
from django.test import SimpleTestCase, TestCase
from rest_framework.test import APIClient

from ..services import llm, retrieval
from ..services.retrieval import RetrievedChunk, fuse, keyword_query
from .helpers import fake_vector, make_chunk


def item(chunk_id, **ranks):
    return RetrievedChunk(chunk=SimpleNamespace(id=chunk_id), similarity=0.0, **ranks)


class FusionTests(SimpleTestCase):
    def test_chunk_found_by_both_searches_ranks_first(self):
        vector = [item(1, vector_rank=1), item(2, vector_rank=2)]
        keyword = [item(3, keyword_rank=1), item(2, keyword_rank=2)]
        fused = fuse(vector, keyword)
        self.assertEqual([r.chunk.id for r in fused], [2, 1, 3])
        both = fused[0]
        self.assertEqual((both.vector_rank, both.keyword_rank), (2, 2))
        self.assertAlmostEqual(both.rrf_score, 2 / (retrieval.RRF_K + 2))

    def test_equal_scores_prefer_the_vector_result(self):
        fused = fuse([item(1, vector_rank=1)], [item(2, keyword_rank=1)])
        self.assertEqual([r.chunk.id for r in fused], [1, 2])

    def test_keyword_query_ors_safe_words(self):
        self.assertIsNone(keyword_query("?!"))
        query = keyword_query("What's the 835 & (CO) | code?")
        self.assertEqual(query.source_expressions[1].value, "what | s | the | 835 | co | code")


class SearchTests(TestCase):
    def setUp(self):
        self.remit = make_chunk("EDI", "The 835 transaction is the electronic remittance advice.", seed=1)
        self.claim = make_chunk("EDI", "The 837 transaction is the claim itself.", seed=2)
        self.hipaa = make_chunk("HIPAA", "A business associate agreement protects patient data.", seed=3)

    def test_keyword_search_matches_exact_terms_and_stems(self):
        results = retrieval.keyword_search("what is an 835?", 5)
        self.assertEqual([r.chunk.id for r in results], [self.remit.id])
        # "agreements" is stemmed to match "agreement".
        results = retrieval.keyword_search("associate agreements", 5)
        self.assertEqual([r.chunk.id for r in results], [self.hipaa.id])

    def test_keyword_search_ignores_stop_words_only_queries(self):
        self.assertEqual(retrieval.keyword_search("what is the", 5), [])

    def test_hybrid_rescues_a_keyword_match_that_vector_search_ranks_last(self):
        # Question vector: identical to the HIPAA chunk, partly similar to the 837 chunk and
        # unrelated to the 835 chunk, so vector search alone ranks the 835 chunk last.
        q_vec = fake_vector(3)
        self.claim.embedding = [0.0] * len(q_vec)
        self.claim.embedding[2] = self.claim.embedding[3] = 0.5
        self.claim.save()
        vector_only = retrieval.vector_search(q_vec, 3)
        self.assertEqual([r.chunk.id for r in vector_only], [self.hipaa.id, self.claim.id, self.remit.id])

        # "835" only matches the 835 chunk by keyword. RRF: 1/63 + 1/61 beats 1/61 alone.
        hybrid = retrieval.hybrid_search("835 remittance", q_vec, 2)
        self.assertEqual([r.chunk.id for r in hybrid], [self.remit.id, self.hipaa.id])
        self.assertEqual((hybrid[0].vector_rank, hybrid[0].keyword_rank), (3, 1))

    def test_rerank_reorders_by_llm_score(self):
        results = retrieval.vector_search(fake_vector(1), 3)
        by_id = {r.chunk.id: i for i, r in enumerate(results)}
        scores = {"scores": [{"id": by_id[self.claim.id], "score": 9}, {"id": by_id[self.remit.id], "score": 2}]}
        with mock.patch.object(llm, "chat_json", return_value=scores):
            reranked = retrieval.rerank_chunks("claim?", results, 2)
        self.assertEqual([r.chunk.id for r in reranked], [self.claim.id, self.remit.id])
        self.assertEqual(reranked[0].rerank_score, 9)

    def test_rerank_failure_keeps_first_pass_order(self):
        results = retrieval.vector_search(fake_vector(1), 3)
        with mock.patch.object(llm, "chat_json", side_effect=llm.LLMRequestError("bad json")):
            reranked = retrieval.rerank_chunks("q", results, 2)
        self.assertEqual([r.chunk.id for r in reranked], [r.chunk.id for r in results[:2]])

    def test_retrieve_with_rerank_reads_more_candidates_than_it_returns(self):
        with mock.patch.object(llm, "embed_texts", return_value=[fake_vector(1)]), mock.patch.object(
            retrieval, "rerank_chunks", side_effect=lambda q, results, k: results[:k]
        ) as rerank:
            results = retrieval.retrieve("835", 1, mode=retrieval.HYBRID, rerank=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(rerank.call_args.args[1]), 3)  # every chunk was a candidate

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            retrieval.retrieve("q", 3, mode="magic")


@mock.patch.object(llm, "chat", return_value="It is the 835 [1].")
@mock.patch.object(llm, "embed_texts", return_value=[fake_vector(3)])
class AskModeApiTests(TestCase):
    def setUp(self):
        cache.clear()
        self.client = APIClient()
        make_chunk("EDI", "The 835 transaction is the electronic remittance advice.", seed=1, page=4)
        make_chunk("HIPAA", "A business associate agreement protects patient data.", seed=3)

    def test_sources_include_page_and_ranks(self, _embed, _chat):
        r = self.client.post("/api/ask/", {"question": "835 remittance", "mode": "hybrid"}, format="json")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["mode"], "hybrid")
        edi = next(s for s in body["sources"] if s["document_title"] == "EDI")
        self.assertEqual(edi["page"], 4)
        self.assertEqual(edi["keyword_rank"], 1)
        self.assertEqual([s["ref"] for s in body["sources"]], [1, 2])

    def test_vector_mode_has_no_keyword_ranks(self, _embed, _chat):
        r = self.client.post("/api/ask/", {"question": "835", "mode": "vector"}, format="json")
        self.assertTrue(all(s["keyword_rank"] is None for s in r.json()["sources"]))

    def test_invalid_mode_is_400(self, _embed, _chat):
        r = self.client.post("/api/ask/", {"question": "q", "mode": "magic"}, format="json")
        self.assertEqual(r.status_code, 400)
