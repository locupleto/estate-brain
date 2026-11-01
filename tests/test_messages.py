"""The messages engine, against a local fake of both wire formats."""

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

import estate_brain
from estate_brain import Task, ToolPolicy
from estate_brain.engines import messages as M

BARE = ToolPolicy(files=False)


def _anthropic_sse(text_parts, stop="end_turn", tool=None, usage_in=100, usage_out=7):
    ev = [{"type": "message_start", "message": {"id": "m1", "model": "claude-haiku-5-5",
                                                "usage": {"input_tokens": usage_in,
                                                          "cache_read_input_tokens": 1000}}},
          {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}]
    ev += [{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}}
           for t in text_parts]
    ev.append({"type": "content_block_stop", "index": 0})
    if tool:
        ev += [{"type": "content_block_start", "index": 1,
                "content_block": {"type": "tool_use", "id": "tu1", "name": tool, "input": {}}},
               {"type": "content_block_delta", "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"app": "Sa'}},
               {"type": "content_block_delta", "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": 'fari"}'}},
               {"type": "content_block_stop", "index": 1}]
    ev += [{"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": usage_out}},
           {"type": "message_stop"}]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in ev)


def _openai_sse(text_parts, tool=None):
    fr = [{"id": "c1", "model": "gpt-x", "choices": [{"index": 0, "delta": {"role": "assistant"}}]}]
    fr += [{"id": "c1", "choices": [{"index": 0, "delta": {"content": t}}]} for t in text_parts]
    if tool:
        fr += [{"id": "c1", "choices": [{"index": 0, "delta": {"tool_calls": [
                   {"index": 0, "id": "call_9", "function": {"name": tool, "arguments": '{"q":'}}]}}]},
               {"id": "c1", "choices": [{"index": 0, "delta": {"tool_calls": [
                   {"index": 0, "function": {"arguments": ' "x"}'}}]}}]}]
    fr.append({"id": "c1", "choices": [{"index": 0, "delta": {},
                                        "finish_reason": "tool_calls" if tool else "stop"}]})
    fr.append({"id": "c1", "choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 5}})
    return "".join(f"data: {json.dumps(f)}\n\n" for f in fr) + "data: [DONE]\n\n"


class _H(BaseHTTPRequestHandler):
    log = []
    replies = []          # queue of (status, body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/api/keyring/values":
            _H.log.append(("GET", self.path, {}, None))
            self._send(200, json.dumps({"anthropic_api_key": "sk-ring", "custom_key": "sk-custom"}),
                       "application/json")
        else:
            self._send(404, "{}", "application/json")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _H.log.append(("POST", self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        status, out = _H.replies.pop(0)
        self._send(status, out, "text/event-stream" if status == 200 else "application/json")

    def _send(self, status, body, ctype):
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class MessagesEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), _H)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        _H.log, _H.replies = [], []
        M._keys.update(at=0.0, values={})
        self.env = mock.patch.dict(os.environ, {"BRAIN_LEDGER": "off", "BRAIN_REGISTRY": self.base,
                                                "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "sk-env"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def anth(self):
        return {"api": "anthropic", "base_url": self.base, "key": "keyring:anthropic_api_key"}

    def test_anthropic_text_usage_cost_and_key_from_the_keyring(self):
        _H.replies.append((200, _anthropic_sse(["Hello", ", Sir."])))
        seen = []
        res = M.run(Task(prompt="hi", system="be brief", tools=BARE, tier="fast", effort="low",
                         on_text=seen.append), profile=self.anth())
        self.assertTrue(res.ok, res.error)
        self.assertEqual((res.text, seen), ("Hello, Sir.", ["Hello", ", Sir."]))
        self.assertEqual((res.model, res.input_tokens, res.output_tokens), ("claude-haiku-5-5", 1100, 7))
        self.assertAlmostEqual(res.cost_usd, (100 * 0.10 + 7 * 0.50 + 1000 * 0.01) / 1e6)
        _, path, headers, body = _H.log[-1]
        self.assertEqual(path, "/v1/messages")
        self.assertEqual(headers.get("x-api-key"), "sk-ring")
        self.assertEqual((body["model"], body["system"], body["output_config"]),
                         ("claude-haiku-5-5", "be brief", {"effort": "low"}))
        self.assertNotIn("temperature", body)
        self.assertTrue(body["stream"])

    def test_history_images_and_functions_go_out_as_turns(self):
        _H.replies.append((200, _anthropic_sse(["Opening it."], stop="tool_use", tool="open_app")))
        fn = [{"name": "open_app", "description": "open", "input_schema": {"type": "object"}}]
        png = b"\x89PNG\r\n\x1a\n" + b"0" * 10
        t = Task(prompt="and now?", tools=BARE, functions=fn, images=[png],
                 messages=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}])
        res = M.run(t, "claude-sonnet-5-5", profile=self.anth())
        self.assertTrue(res.ok, res.error)
        self.assertEqual(len(res.tool_calls), 1)
        tc = res.tool_calls[0]
        self.assertEqual((tc.name, tc.input, tc.id), ("fn:open_app", {"app": "Safari"}, "tu1"))
        body = _H.log[-1][3]
        self.assertEqual([m["role"] for m in body["messages"]], ["user", "assistant", "user"])
        self.assertEqual(body["messages"][2]["content"][0]["source"]["media_type"], "image/png")
        self.assertEqual(body["tools"], fn)

    def test_openai_compatible_is_translated_into_messages_events(self):
        _H.replies.append((200, _openai_sse(["Hej", " då"], tool="search")))
        prof = {"api": "openai-compatible", "base_url": self.base + "/v1", "key": "keyring:custom_key"}
        t = Task(prompt="q", system="sys", tools=BARE,
                 functions=[{"name": "search", "input_schema": {"type": "object"}}],
                 messages=[{"role": "user", "content": "a"},
                           {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "search",
                                                              "input": {"q": 1}}]},
                           {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0",
                                                         "content": "found"}]}])
        s = M.stream(t, "gpt-x", profile=prof)
        kinds = [e["type"] for e in s]
        res = s.close()
        self.assertTrue(res.ok, res.error)
        self.assertEqual(kinds[0], "message_start")
        self.assertEqual(kinds[-2:], ["message_delta", "message_stop"])
        self.assertEqual(s.stop_reason, "tool_use")
        self.assertEqual((res.text, res.input_tokens, res.output_tokens), ("Hej då", 50, 5))
        self.assertEqual((res.tool_calls[0].input, res.tool_calls[0].id), ({"q": "x"}, "call_9"))
        self.assertIsNone(res.cost_usd)                       # no price for an unknown model
        _, path, headers, body = _H.log[-1]
        self.assertEqual((path, headers.get("authorization")), ("/v1/chat/completions", "Bearer sk-custom"))
        roles = [m["role"] for m in body["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "user"])
        self.assertEqual(body["messages"][2]["tool_calls"][0]["function"]["name"], "search")
        self.assertEqual(body["tools"][0]["function"]["name"], "search")
        self.assertNotIn("stream_options", body)              # unknown host: not sent

    def test_close_from_another_thread_ends_quietly(self):
        _H.replies.append((200, _anthropic_sse(["a"] * 200)))
        s = M.stream(Task(prompt="x", tools=BARE), "claude-haiku-5-5", profile=self.anth())
        it = iter(s)
        next(it)
        s._resp.fp = None                      # what a cross-thread close leaves behind
        s._closed = True
        rest = list(it)                        # must not raise
        self.assertLess(len(rest), 400)
        self.assertEqual(s.result.error, "")

    def test_schema_reply_is_validated_with_one_retry(self):
        schema = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}}
        _H.replies += [(200, _anthropic_sse(['{"n": "two"}'])), (200, _anthropic_sse(['{"n": 2}']))]
        res = M.run(Task(prompt="count", schema=schema, tools=BARE), "claude-haiku-5-5", profile=self.anth())
        self.assertTrue(res.ok, res.error)
        self.assertEqual(res.structured, {"n": 2})
        self.assertEqual(len(_H.log) - 1, 2)                  # one keyring read, two calls
        self.assertIn("not valid", _H.log[-1][3]["messages"][-1]["content"])
        self.assertEqual(res.output_tokens, 14)               # both calls counted

    def test_refuses_tools_it_does_not_have(self):
        res = M.run(Task(prompt="x"), "claude-haiku-5-5", profile=self.anth())
        self.assertIn("no built-in tools", res.error)
        self.assertEqual(_H.log, [])                          # nothing was called
        self.assertIsNone(M.refuse(Task(prompt="x", tools=BARE)))
        self.assertIn("web", M.refuse(Task(prompt="x", tools=ToolPolicy(files=False, web=True))))

    def test_http_errors_read_as_api_errors(self):
        _H.replies.append((529, '{"type":"error","error":{"type":"overloaded_error"}}'))
        res = M.run(Task(prompt="x", tools=BARE), "claude-haiku-5-5", profile=self.anth())
        self.assertTrue(res.error.startswith("API Error: 529 from 127.0.0.1"), res.error)
        self.assertIn("overloaded", res.raw_tail)

    def test_no_key_no_call(self):
        prof = {"api": "openai-compatible", "base_url": "https://api.example.invalid/v1",
                "key": "keyring:missing"}
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            res = M.run(Task(prompt="x", tools=BARE), "gpt-x", profile=prof)
        self.assertIn("no API key", res.error)

    def test_routing_by_model_prefix(self):
        r = M.route("gpt-5.5")
        self.assertEqual((r["url"], r["keyring"], r["env"]),
                         ("https://api.openai.com/v1/chat/completions", "openai_api_key", "OPENAI_API_KEY"))
        self.assertEqual(M.route("claude-opus-5-5")["api"], "anthropic")
        self.assertTrue(M.route("grok-4.6")["url"].startswith("https://api.x.ai/"))
        local = M.route("qwen3-8b", {"api": "openai-compatible", "base_url": "http://localhost:11434/v1"})
        self.assertEqual(local["url"], "http://localhost:11434/v1/chat/completions")
        self.assertTrue(M._local(local["url"]))
        with self.assertRaises(ValueError):
            M.route("mystery-model")

    def test_public_run_routes_to_it_and_agent_engines_adapt(self):
        _H.replies.append((200, _anthropic_sse(["ok"])))
        with mock.patch.object(M, "route", lambda model, profile=None:
                               {"api": "anthropic", "url": self.base + "/v1/messages",
                                "keyring": "anthropic_api_key", "env": ""}):
            res = estate_brain.run(Task(prompt="x", tools=BARE, tier="fast"), engine="messages")
        self.assertEqual((res.engine, res.text), ("messages", "ok"))
        # caller functions cannot run on an agent CLI: refused, no model call
        res = estate_brain.run(Task(prompt="x", functions=[{"name": "f"}]), engine="claude-code")
        self.assertIn("cannot run caller functions", res.error)
        t, err = estate_brain._adapt(Task(prompt="now?", messages=[
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "text", "text": "yo"}]}]),
            "opencode")
        self.assertEqual(err, "")
        self.assertEqual(t.prompt, "Conversation so far:\nUser: hi\n\nAssistant: yo\n\nnow?")


if __name__ == "__main__":
    unittest.main()
