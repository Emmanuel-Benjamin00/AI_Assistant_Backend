import json
from unittest import mock

from django.core.cache import cache
from django.test import TestCase
from rest_framework.test import APIClient

from ..services import agent, llm
from .helpers import fake_embed, make_chunk


def tool_call(call_id, name, **arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def assistant(content=None, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


@mock.patch.object(llm, "embed_texts", side_effect=fake_embed)
class AgentLoopTests(TestCase):
    def setUp(self):
        self.remit = make_chunk("EDI Primer", "The 835 transaction is the electronic remittance advice.", seed=1)
        make_chunk("HIPAA", "A business associate agreement protects patient data.", seed=2)

    def test_search_then_answer(self, _embed):
        replies = [
            assistant(tool_calls=[tool_call("c1", "search_documents", query="835 remittance", top_k=2)]),
            assistant("The 835 is the electronic remittance advice [1]."),
        ]
        with mock.patch.object(llm, "complete", side_effect=replies) as complete:
            result = agent.run_agent("What is an 835?")

        self.assertEqual(result["answer"], "The 835 is the electronic remittance advice [1].")
        self.assertEqual(result["steps"][0]["tool"], "search_documents")
        self.assertEqual(result["steps"][0]["arguments"], {"query": "835 remittance", "top_k": 2})
        self.assertEqual(result["sources"][0]["chunk_id"], self.remit.id)
        self.assertEqual(result["sources"][0]["ref"], 1)

        # The second model call saw the assistant tool call and the tool result, in order.
        messages = complete.call_args_list[1].args[0]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool"])
        self.assertEqual(messages[3]["tool_call_id"], "c1")
        tool_output = json.loads(messages[3]["content"])
        self.assertEqual(tool_output["results"][0]["ref"], 1)
        self.assertNotIn("_summary", tool_output)
        self.assertEqual(complete.call_args_list[0].kwargs["tools"], agent.TOOLS)

    def test_list_and_read_document_for_a_summary(self, _embed):
        replies = [
            assistant(tool_calls=[tool_call("c1", "list_documents")]),
            assistant(tool_calls=[tool_call("c2", "read_document", document_id=self.remit.document_id)]),
            assistant("Summary: it explains the 835."),
        ]
        with mock.patch.object(llm, "complete", side_effect=replies) as complete:
            result = agent.run_agent("Summarize the EDI primer")

        self.assertEqual([s["tool"] for s in result["steps"]], ["list_documents", "read_document"])
        self.assertEqual(result["documents_read"], [{"document_id": self.remit.document_id, "document_title": "EDI Primer"}])
        read_output = json.loads(complete.call_args_list[2].args[0][-1]["content"])
        self.assertIn("electronic remittance advice", read_output["text"])
        self.assertFalse(read_output["truncated"])

    def test_read_document_is_truncated(self, _embed):
        with mock.patch.object(agent, "READ_DOCUMENT_MAX_CHARS", 10):
            output = agent._read_document(agent.AgentRun("q"), self.remit.document_id)
        self.assertEqual(len(output["text"]), 10)
        self.assertTrue(output["truncated"])

    def test_bad_tool_calls_are_reported_to_the_model(self, _embed):
        bad_json = {"id": "c3", "type": "function", "function": {"name": "search_documents", "arguments": "{oops"}}
        replies = [
            assistant(
                tool_calls=[
                    tool_call("c1", "delete_everything"),
                    tool_call("c2", "read_document", document_id=999),
                    bad_json,
                    tool_call("c4", "search_documents", query="x", unexpected=1),
                ]
            ),
            assistant("I don't know."),
        ]
        with mock.patch.object(llm, "complete", side_effect=replies) as complete:
            result = agent.run_agent("q")

        self.assertEqual(result["answer"], "I don't know.")
        tool_messages = [m for m in complete.call_args_list[1].args[0] if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 4)
        errors = [json.loads(m["content"])["error"] for m in tool_messages]
        self.assertIn("unknown tool", errors[0])
        self.assertIn("no document with id", errors[1])
        self.assertTrue(all(s["summary"].startswith("error") for s in result["steps"]))

    def test_step_limit_forces_a_final_answer(self, _embed):
        looping = assistant(tool_calls=[tool_call("c", "list_documents")])
        with mock.patch.object(agent, "MAX_STEPS", 2), mock.patch.object(
            llm, "complete", side_effect=[looping, looping, assistant("Best effort answer.")]
        ) as complete:
            result = agent.run_agent("q")
        self.assertEqual(result["answer"], "Best effort answer.")
        self.assertEqual(len(result["steps"]), 2)
        self.assertEqual(complete.call_args_list[-1].kwargs["tool_choice"], "none")

    def test_search_refs_are_unique_across_searches(self, _embed):
        run = agent.AgentRun("q")
        agent._search_documents(run, "835", top_k=2)
        agent._search_documents(run, "835 remittance", top_k=2)
        refs = [s["ref"] for s in run.sources]
        self.assertEqual(refs, list(range(1, len(refs) + 1)))
        self.assertEqual(len({s["chunk_id"] for s in run.sources}), len(refs))


class AgentApiTests(TestCase):
    def setUp(self):
        cache.clear()
        self.client = APIClient()

    def test_agent_endpoint(self):
        with mock.patch.object(agent, "run_agent", return_value={"answer": "A", "steps": [], "sources": [], "documents_read": []}):
            r = self.client.post("/api/agent/", {"question": "q"}, format="json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["answer"], "A")

    def test_agent_provider_error(self):
        with mock.patch.object(agent, "run_agent", side_effect=llm.LLMConfigError("no key")):
            r = self.client.post("/api/agent/", {"question": "q"}, format="json")
        self.assertEqual(r.status_code, 503)
