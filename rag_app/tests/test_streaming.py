import json
from unittest import mock

import httpx
from django.core.cache import cache
from django.test import SimpleTestCase, TestCase
from rest_framework.test import APIClient

from ..services import llm
from .helpers import fake_embed, make_chunk


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


class ChatStreamParsingTests(SimpleTestCase):
    def setUp(self):
        patcher = mock.patch.multiple(llm, LLM_PROVIDER="openai", OPENAI_API_KEY="k")
        patcher.start()
        self.addCleanup(patcher.stop)

    def stream_response(self, status, body: bytes):
        return httpx.Response(status, content=body, request=httpx.Request("POST", "https://x"))

    def test_yields_content_deltas_until_done(self):
        body = (
            b'data: {"choices": []}\n\n'  # Azure content-filter preamble
            b": keep-alive\n\n"
            b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n'
            b'data: {"choices": [{"delta": {"content": "Hel"}}]}\n\n'
            b'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n'
            b"data: [DONE]\n\n"
            b'data: {"choices": [{"delta": {"content": "ignored"}}]}\n\n'
        )
        with mock.patch.object(llm._client, "send", return_value=self.stream_response(200, body)) as send:
            self.assertEqual(list(llm.chat_stream("s", "u")), ["Hel", "lo"])
        request = send.call_args.args[0]
        self.assertTrue(json.loads(request.content)["stream"])

    def test_http_error_raises_before_streaming(self):
        with mock.patch.object(llm._client, "send", return_value=self.stream_response(401, b"{}")):
            with self.assertRaises(llm.LLMRequestError):
                llm.chat_stream("s", "u")

    def test_retries_retryable_status_before_streaming(self):
        ok = self.stream_response(200, b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\ndata: [DONE]\n\n')
        with mock.patch.object(llm.time, "sleep"), mock.patch.object(
            llm._client, "send", side_effect=[self.stream_response(503, b""), ok]
        ):
            self.assertEqual(list(llm.chat_stream("s", "u")), ["ok"])

    def test_malformed_event_raises_request_error(self):
        with mock.patch.object(llm._client, "send", return_value=self.stream_response(200, b"data: {oops\n\n")):
            with self.assertRaises(llm.LLMRequestError):
                list(llm.chat_stream("s", "u"))


@mock.patch.object(llm, "embed_texts", side_effect=fake_embed)
class AskStreamApiTests(TestCase):
    def setUp(self):
        cache.clear()
        self.client = APIClient()

    def post(self, question="When is support?"):
        return self.client.post("/api/ask/stream/", {"question": question}, format="json")

    def body(self, response) -> str:
        return b"".join(response.streaming_content).decode()

    def test_streams_sources_then_tokens_then_done(self, _embed):
        make_chunk("Facts", "Support hours are 9 to 5.", seed=1)
        with mock.patch.object(llm, "chat_stream", return_value=iter(["Support is ", "9 to 5 [1]."])):
            r = self.post()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r["Content-Type"], "text/event-stream")
        events = parse_sse(self.body(r))
        self.assertEqual([e for e, _ in events], ["sources", "token", "token", "done"])
        self.assertEqual(events[0][1]["sources"][0]["document_title"], "Facts")
        self.assertEqual("".join(d["text"] for e, d in events if e == "token"), "Support is 9 to 5 [1].")

    def test_empty_index_streams_the_no_content_answer_without_calling_the_model(self, _embed):
        with mock.patch.object(llm, "chat_stream") as chat_stream:
            events = parse_sse(self.body(self.post()))
        chat_stream.assert_not_called()
        self.assertEqual([e for e, _ in events], ["sources", "token", "done"])
        self.assertIn("Ingest documents first", events[1][1]["text"])

    def test_failure_before_first_token_is_a_json_error(self, _embed):
        make_chunk("Facts", "Support hours are 9 to 5.", seed=1)
        with mock.patch.object(llm, "chat_stream", side_effect=llm.LLMRequestError("down", 500)):
            r = self.post()
        self.assertEqual(r.status_code, 502)
        self.assertIn("detail", r.json())

    def test_failure_mid_stream_sends_an_error_event(self, _embed):
        make_chunk("Facts", "Support hours are 9 to 5.", seed=1)

        def broken():
            yield "Support"
            raise llm.LLMRequestError("connection reset")

        with mock.patch.object(llm, "chat_stream", return_value=broken()):
            events = parse_sse(self.body(self.post()))
        self.assertEqual([e for e, _ in events], ["sources", "token", "error"])

    def test_validation_error_is_400(self, _embed):
        r = self.client.post("/api/ask/stream/", {"question": ""}, format="json")
        self.assertEqual(r.status_code, 400)
