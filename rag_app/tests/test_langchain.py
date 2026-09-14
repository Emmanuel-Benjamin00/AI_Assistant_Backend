from unittest import mock

from django.core.cache import cache
from django.test import SimpleTestCase, TestCase
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from rest_framework.test import APIClient

from ..services import langchain_rag, llm
from .helpers import fake_embed, make_chunk


class RecordingChatModel(FakeListChatModel):
    """Fake chat model that remembers the prompts it received."""

    seen: list = []

    def _call(self, messages, *args, **kwargs):
        self.seen.append(messages)
        return super()._call(messages, *args, **kwargs)


@mock.patch.object(llm, "embed_texts", side_effect=fake_embed)
class LangChainPipelineTests(TestCase):
    def test_chain_retrieves_builds_prompt_and_answers(self, _embed):
        make_chunk("Facts", "Support hours are 9 to 5.", seed=1, page=2)
        model = RecordingChatModel(responses=["Support is 9 to 5 [1]."], seen=[])

        result = langchain_rag.ask("When is support?", top_k=3, mode="hybrid", chat_model=model)

        self.assertEqual(result["answer"], "Support is 9 to 5 [1].")
        self.assertEqual(result["results"][0].chunk.text, "Support hours are 9 to 5.")
        system, human = model.seen[0]
        self.assertIn("ONLY the provided context", system.content)
        self.assertIn("[1] (from: Facts, page 2)", human.content)
        self.assertIn("When is support?", human.content)

    def test_empty_index_skips_the_model(self, _embed):
        model = RecordingChatModel(responses=["should not be used"], seen=[])
        result = langchain_rag.ask("q", top_k=3, mode="vector", chat_model=model)
        self.assertIn("Ingest documents first", result["answer"])
        self.assertEqual(model.seen, [])

    def test_openai_sdk_errors_are_mapped(self, _embed):
        import httpx
        import openai

        make_chunk("Facts", "Support hours are 9 to 5.", seed=1)
        response = httpx.Response(429, request=httpx.Request("POST", "https://x"))
        failing = mock.MagicMock(side_effect=openai.RateLimitError("slow down", response=response, body=None))
        with mock.patch.object(FakeListChatModel, "_call", failing):
            with self.assertRaises(llm.LLMRequestError) as ctx:
                langchain_rag.ask("q", 3, "hybrid", chat_model=FakeListChatModel(responses=["x"]))
        self.assertEqual(ctx.exception.status_code, 429)


class ChatModelConfigTests(SimpleTestCase):
    def test_openai_model_uses_our_env_settings(self):
        with mock.patch.multiple(llm, LLM_PROVIDER="openai", OPENAI_API_KEY="k", OPENAI_CHAT_MODEL="gpt-test"):
            model = langchain_rag.build_chat_model()
        self.assertEqual(model.model_name, "gpt-test")
        self.assertEqual(model.max_retries, llm.MAX_ATTEMPTS - 1)

    def test_azure_model(self):
        with mock.patch.multiple(
            llm,
            LLM_PROVIDER="azure",
            AZURE_OPENAI_ENDPOINT="https://x.openai.azure.com",
            AZURE_OPENAI_API_KEY="k",
            AZURE_OPENAI_CHAT_DEPLOYMENT="chat",
        ):
            model = langchain_rag.build_chat_model()
        self.assertEqual(model.deployment_name, "chat")

    def test_missing_key_is_a_config_error(self):
        with mock.patch.multiple(llm, LLM_PROVIDER="openai", OPENAI_API_KEY=""):
            with self.assertRaises(llm.LLMConfigError):
                langchain_rag.build_chat_model()


class LangChainApiTests(TestCase):
    def setUp(self):
        cache.clear()
        self.client = APIClient()

    def test_endpoint_returns_pipeline_and_sources(self):
        with mock.patch.object(langchain_rag, "ask", return_value={"question": "q", "results": [], "answer": "A"}):
            r = self.client.post("/api/ask/langchain/", {"question": "q"}, format="json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"answer": "A", "mode": "hybrid", "pipeline": "langchain", "sources": []})
