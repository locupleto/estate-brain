"""The fallback chain: which failures retry, where, and what the ledger says."""

import json
import os
import tempfile
import unittest
from unittest import mock

import estate_brain
from estate_brain import Result, Task, ToolCall, ToolPolicy, registry
from estate_brain import engines
from estate_brain.engines import claude_code

DOC = {
    "version": "v9",
    "profiles": {
        "claude-sub": {"engine": "claude-code", "tiers": {"fast": "haiku", "standard": "sonnet", "deep": "opus"}},
        "claude-api": {"engine": "messages", "api": "anthropic", "key": "keyring:anthropic_api_key",
                       "tiers": {"fast": "claude-haiku-5-5", "standard": "claude-sonnet-5-5",
                                 "deep": "claude-opus-5-5"}},
        "claude-api-cli": {"engine": "claude-code", "key": "keyring:anthropic_api_key",
                           "tiers": {"fast": "haiku", "standard": "sonnet", "deep": "opus"}},
        "off": {"engine": "messages", "enabled": False},
    },
    "defaults": {"profile": "claude-sub", "fallback": ["claude-api", "off", "claude-api-cli"]},
    "assign": {"quiet": {"fallback": []}, "solo": {"fallback": "claude-api-cli"}},
}


class _Fake:
    """An engine module stand-in: scripted results, records its calls."""

    def __init__(self, name, results):
        self.NAME, self.results, self.calls = name, list(results), []

    def run(self, task, model=None, profile=None):
        self.calls.append((task, model, profile))
        r = self.results.pop(0)
        return Result(engine=self.NAME, model=model or "", **r)


class FallbackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ledger = os.path.join(self.tmp, "l.jsonl")
        self.env = mock.patch.dict(os.environ, {"BRAIN_LEDGER": self.ledger})
        self.env.start()
        self.reg = mock.patch.object(registry, "current", lambda environ=None: (DOC, "live"))
        self.reg.start()

    def tearDown(self):
        self.reg.stop()
        self.env.stop()

    def _engines(self, cc, ms):
        real = dict(engines.ENGINES)
        fake = {"claude-code": cc, "messages": ms, "opencode": real["opencode"]}
        # keep the real refuse() rule for the messages engine
        ms.refuse = real["messages"].refuse
        return mock.patch.dict(engines.ENGINES, fake, clear=True)

    def rows(self):
        return [json.loads(x) for x in open(self.ledger)]

    def test_registry_fallbacks_order_skip_and_override(self):
        fb = registry.fallbacks(DOC, "bots", "JARVIS", "standard", "opus")
        self.assertEqual([(c.profile, c.engine, c.model) for c in fb],
                         [("claude-api", "messages", "claude-opus-5-5"),   # opus -> deep tier
                          ("claude-api-cli", "claude-code", "opus")])       # alias kept on claude-code
        self.assertEqual(registry.fallbacks(DOC, "quiet", "", "fast", None), [])
        self.assertEqual([c.profile for c in registry.fallbacks(DOC, "solo", "x", "fast", None)],
                         ["claude-api-cli"])
        self.assertEqual(fb[0].spec["key"], "keyring:anthropic_api_key")

    def test_overload_on_a_bare_task_falls_back_to_the_api(self):
        cc = _Fake("claude-code", [{"error": "error_during_execution", "raw_tail": "API Error: 529 overloaded"}])
        ms = _Fake("messages", [{"text": "ok", "cost_usd": 0.01}])
        with self._engines(cc, ms):
            res = estate_brain.run(Task(prompt="x", tools=ToolPolicy(files=False), app="photo-ocr"))
        self.assertEqual((res.ok, res.engine, res.text), (True, "messages", "ok"))
        self.assertEqual(ms.calls[0][2]["api"], "anthropic")              # the profile travelled
        rows = self.rows()
        self.assertEqual([(r["engine"], r["ok"], r["source"], r["profile"]) for r in rows],
                         [("claude-code", False, "registry", "claude-sub"),
                          ("messages", True, "fallback", "claude-api")])

    def test_a_tool_task_skips_the_api_and_runs_the_metered_cli(self):
        cc = _Fake("claude-code", [{"error": "success", "text": "Claude AI usage limit reached|1762000000"},
                                   {"text": "done"}])
        ms = _Fake("messages", [])
        with self._engines(cc, ms):
            res = estate_brain.run(Task(prompt="x", app="bots", role="JARVIS", model="opus"))
        self.assertEqual((res.ok, res.text), (True, "done"))
        self.assertEqual(ms.calls, [])                                    # refused: the task has file tools
        task, model, profile = cc.calls[1]
        self.assertEqual((model, profile["key"]), ("opus", "keyring:anthropic_api_key"))
        self.assertEqual([r["profile"] for r in self.rows()], ["claude-sub", "claude-api-cli"])

    def test_never_after_an_action_or_for_the_wrong_failures(self):
        cases = [
            {"error": "error_during_execution", "raw_tail": "529 overloaded",
             "tool_calls": [ToolCall("edit", {"file": "x"})]},            # something acted
            {"error": "timeout", "raw_tail": "overloaded"},
            {"error": "budget: Exceeded USD budget"},
            {"error": "API Error: 400 from api.anthropic.com", "raw_tail": "invalid_request_error"},
        ]
        for case in cases:
            with self.subTest(case=case["error"]):
                cc = _Fake("claude-code", [case])
                ms = _Fake("messages", [])
                with self._engines(cc, ms):
                    res = estate_brain.run(Task(prompt="x", tools=ToolPolicy(files=False)))
                self.assertFalse(res.ok)
                self.assertEqual((len(cc.calls), ms.calls), (1, []))

    def test_both_fail_says_so_and_stays_transient(self):
        cc = _Fake("claude-code", [{"error": "x", "raw_tail": "API Error: 529"},
                                   {"error": "y", "raw_tail": "API Error: 529"}])
        ms = _Fake("messages", [{"error": "API Error: 529 from api.anthropic.com"}])
        with self._engines(cc, ms):
            res = estate_brain.run(Task(prompt="x", tools=ToolPolicy(files=False)))
        self.assertIn("fallback claude-api-cli", res.error)
        from estate_brain.runcli import TRANSIENT, classify
        self.assertEqual(classify(res), TRANSIENT)
        self.assertEqual(len(self.rows()), 3)

    def test_explicit_engines_never_fall_back(self):
        cc = _Fake("claude-code", [{"error": "x", "raw_tail": "overloaded"}])
        ms = _Fake("messages", [])
        with self._engines(cc, ms):
            res = estate_brain.run(Task(prompt="x", tools=ToolPolicy(files=False)), engine="claude-code")
        self.assertFalse(res.ok)
        self.assertEqual(len(cc.calls), 1)


class MissingBinaryTest(unittest.TestCase):
    def test_a_missing_binary_is_an_error_result_not_an_exception(self):
        with mock.patch.dict(os.environ, {"CLAUDE_BIN": "/nonexistent/claude"}):
            res = claude_code.run(Task(prompt="x"), "haiku")
        self.assertEqual(res.error, "claude binary not found")
        self.assertTrue(estate_brain.fallback_worthy(res))

    def test_a_binary_gone_at_spawn_time_is_an_error_result(self):
        with mock.patch.object(claude_code, "find_binary", lambda *a: "/nonexistent/claude"):
            res = claude_code.run(Task(prompt="x"), "haiku")
        self.assertTrue(res.error.startswith("claude binary not found"))
        self.assertTrue(estate_brain.fallback_worthy(res))


class MeteredCliTest(unittest.TestCase):
    def test_the_key_goes_in_only_when_the_profile_names_one(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-leak"}):
            self.assertNotIn("ANTHROPIC_API_KEY", claude_code.env_for())
            self.assertEqual(claude_code.env_for("sk-metered")["ANTHROPIC_API_KEY"], "sk-metered")

    def test_a_keyless_metered_profile_never_runs_on_the_subscription(self):
        with mock.patch("estate_brain.keys.keyring", lambda name: None), \
                mock.patch.object(claude_code, "find_binary", lambda *a: "/bin/true"):
            res = claude_code.run(Task(prompt="x"), "haiku", profile={"key": "keyring:anthropic_api_key"})
        self.assertIn("no API key", res.error)


if __name__ == "__main__":
    unittest.main()
