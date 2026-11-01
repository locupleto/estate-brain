"""The central registry: resolution, cache, outage behaviour, ledger fields."""

import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from estate_brain import Task, registry
from estate_brain.resolve import resolve_full

DOC = {
    "version": "v1",
    "profiles": {
        "claude-sub": {"engine": "claude-code", "tiers": {"fast": "haiku", "standard": "sonnet", "deep": "opus"}},
        "gpt": {"engine": "opencode", "tiers": {"fast": "openai/f", "standard": "openai/s", "deep": "openai/d"},
                "key": "keyring:openai_api_key"},
        "local": {"engine": "messages", "enabled": False, "tiers": {"fast": "qwen"}},
        "future": {"engine": "messages", "tiers": {"fast": "x"}},
    },
    "defaults": {"profile": "claude-sub"},
    "assign": {"bots.JARVIS": {"profile": "gpt"}, "academy": {"tier": "standard"},
               "academy.exams": {"profile": "gpt", "tier": "fast"},
               "dormant": {"profile": "local"}, "later": {"profile": "future"}},
}


class _Handler(BaseHTTPRequestHandler):
    hits = []
    doc = DOC

    def do_GET(self):
        _Handler.hits.append((self.path, self.headers.get("X-Brain-Host")))
        body = json.dumps(_Handler.doc).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class ChooseTest(unittest.TestCase):
    def test_most_specific_assignment_wins(self):
        c = registry.choose(DOC, "academy", "exams", "deep", None)
        self.assertEqual((c.profile, c.engine, c.model, c.tier), ("gpt", "opencode", "openai/f", "fast"))
        c = registry.choose(DOC, "academy", "lectures", "deep", None)
        self.assertEqual((c.profile, c.model, c.tier), ("claude-sub", "sonnet", "standard"))

    def test_a_claude_alias_never_travels(self):
        c = registry.choose(DOC, "bots", "JARVIS", "deep", "opus")
        self.assertEqual((c.engine, c.model), ("opencode", "openai/d"))
        c = registry.choose(DOC, "bots", "GORDON", "deep", "opus")
        self.assertEqual((c.engine, c.model), ("claude-code", "opus"))

    def test_disabled_profile_falls_to_default(self):
        self.assertEqual(registry.choose(DOC, "dormant", "", "fast", None).profile, "claude-sub")

    def test_forced_profile(self):
        self.assertEqual(registry.choose(DOC, "x", "", "deep", None, profile="gpt").profile, "gpt")


class LiveRegistryTest(unittest.TestCase):
    def setUp(self):
        self.srv = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.dir = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {
            "BRAIN_REGISTRY": f"http://127.0.0.1:{self.srv.server_port}", "BRAIN_CONFIG_DIR": self.dir})
        self.env.start()
        for k in ("BRAIN_ENGINE", "BRAIN_MODEL", "BRAIN_PROFILE"):
            os.environ.pop(k, None)
        _Handler.hits.clear()
        _Handler.doc = DOC

    def tearDown(self):
        self.env.stop()
        if self.srv:
            self.srv.shutdown()
            self.srv.server_close()

    def test_resolves_through_the_registry_and_caches(self):
        r = resolve_full(Task(prompt="x", app="bots", role="JARVIS", model="opus", tier="deep"))
        self.assertEqual((r.engine, r.model, r.source, r.profile, r.version),
                         ("opencode", "openai/d", "registry", "gpt", "v1"))
        resolve_full(Task(prompt="x", app="bots", role="JARVIS"))
        self.assertEqual(len(_Handler.hits), 1, "a second call within the TTL must not fetch")
        self.assertEqual(_Handler.hits[0][0], "/api/brains/registry")
        self.assertTrue(_Handler.hits[0][1])
        self.assertTrue(Path(self.dir, "registry.cache.json").exists())

    def test_env_beats_registry(self):
        with mock.patch.dict(os.environ, {"BRAIN_BOTS_ENGINE": "claude-code"}):
            r = resolve_full(Task(prompt="x", app="bots", role="JARVIS", model="opus"))
        self.assertEqual((r.engine, r.model, r.source), ("claude-code", "opus", "env"))

    def test_a_profile_that_refuses_the_task_is_passed_over(self):
        # The messages engine has no built-in tools: a default Task (file
        # tools on) cannot run on it faithfully, so the default takes it.
        r = resolve_full(Task(prompt="x", app="later"))
        self.assertEqual((r.engine, r.source), ("claude-code", "default"))
        from estate_brain import ToolPolicy
        r = resolve_full(Task(prompt="x", app="later", tier="fast", tools=ToolPolicy(files=False)))
        self.assertEqual((r.engine, r.model, r.source, r.profile), ("messages", "x", "registry", "future"))
        self.assertEqual(r.spec, {"tiers": {"fast": "x"}})

    def test_stream_skips_an_engine_that_cannot_stream(self):
        r = resolve_full(Task(prompt="x", app="bots", role="JARVIS"), need_stream=True)
        self.assertEqual((r.engine, r.source), ("claude-code", "default"))

    def test_outage_uses_the_cache_and_does_not_hammer(self):
        resolve_full(Task(prompt="x", app="bots", role="JARVIS"))
        self.srv.shutdown()
        self.srv.server_close()
        c = json.loads(Path(self.dir, "registry.cache.json").read_text())
        c["fetched_at"] = time.time() - 3600
        Path(self.dir, "registry.cache.json").write_text(json.dumps(c))
        t0 = time.monotonic()
        r = resolve_full(Task(prompt="x", app="bots", role="JARVIS"))
        self.assertEqual((r.engine, r.source), ("opencode", "registry-cache"))
        t1 = time.monotonic()
        r = resolve_full(Task(prompt="x", app="bots", role="JARVIS"))
        self.assertEqual(r.source, "registry-cache")
        self.assertLess(time.monotonic() - t1, 0.2, "a failed fetch must not be retried within the TTL")
        self.assertLess(t1 - t0, registry.TIMEOUT_S + 1)
        self.srv = None   # already shut down

    def test_no_registry_is_phase_2_behaviour(self):
        with mock.patch.dict(os.environ, {"BRAIN_REGISTRY": "off"}):
            r = resolve_full(Task(prompt="x", app="bots", role="JARVIS", model="opus"))
        self.assertEqual((r.engine, r.model, r.source), ("claude-code", "opus", "default"))
        self.assertEqual(_Handler.hits, [])

    def test_ledger_names_the_source(self):
        from estate_brain import ledger
        from estate_brain.task import Result
        p = Path(self.dir, "l.jsonl")
        with mock.patch.dict(os.environ, {"BRAIN_LEDGER": str(p)}):
            r = resolve_full(Task(prompt="x", app="academy", role="exams"))
            ledger.record(Task(prompt="x", app="academy", role="exams"), Result(engine=r.engine), r)
        row = json.loads(p.read_text())
        self.assertEqual((row["source"], row["profile"], row["registry"]), ("registry", "gpt", "v1"))


if __name__ == "__main__":
    unittest.main()
