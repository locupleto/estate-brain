"""The opencode engine on a bot turn: key, catalogue, budget, transcript."""

import json
import os
import stat
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

from estate_brain import Task, ToolPolicy
from estate_brain.engines import opencode

CAT = {"openai": {"env": ["OPENAI_API_KEY"], "models": {"gpt-6.1-sol": {}, "gpt-6.1-terra": {}}}}


def _step(cost, sid="ses_1"):
    return {"type": "step_finish", "sessionID": sid,
            "part": {"type": "step-finish", "cost": cost, "tokens": {"input": 10, "output": 2}}}


def _text(t, sid="ses_1"):
    return {"type": "text", "sessionID": sid, "part": {"type": "text", "text": t}}


class OpencodeBotTurnTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.home = os.path.join(self.tmp, "eb")
        os.makedirs(self.home)
        with open(os.path.join(self.home, "opencode-models.json"), "w") as f:
            json.dump(CAT, f)
        self.envdump = os.path.join(self.tmp, "env.json")
        self.patch = mock.patch.dict(os.environ, {"ESTATE_BRAIN_HOME": self.home,
                                                  "ESTATE_BRAIN_OFFLINE": "1"})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def fake(self, events, sleep_after=0.0):
        """An `opencode` that dumps its env, prints the events, then lingers."""
        path = os.path.join(self.tmp, "opencode")
        body = textwrap.dedent(f"""\
            #!{sys.executable}
            import json, os, sys, time
            json.dump(dict(os.environ), open({self.envdump!r}, "w"))
            for ev in {events!r}:
                print(json.dumps(ev), flush=True)
            time.sleep({sleep_after})
            """)
        with open(path, "w") as f:
            f.write(body)
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        return path

    def run_(self, exe, task, model="openai/gpt-6.1-sol", profile=None):
        with mock.patch.dict(os.environ, {"OPENCODE_BIN": exe}):
            return opencode.run(task, model, profile=profile)

    def test_the_profile_key_lands_in_the_providers_own_variable(self):
        exe = self.fake([_text("pong"), _step(0.001)])
        with mock.patch("estate_brain.keys.keyring", lambda name: "sk-test" if name == "openai_api_key" else None):
            res = self.run_(exe, Task(prompt="x", tools=ToolPolicy(files=False)),
                            profile={"key": "keyring:openai_api_key"})
        self.assertEqual((res.ok, res.text), (True, "pong"))
        env = json.load(open(self.envdump))
        self.assertEqual(env["OPENAI_API_KEY"], "sk-test")
        self.assertEqual(env["OPENCODE_MODELS_PATH"], os.path.join(self.home, "opencode-models.json"))

    def test_without_a_catalogue_the_known_providers_still_get_their_key(self):
        os.remove(os.path.join(self.home, "opencode-models.json"))
        with mock.patch("estate_brain.keys.keyring", lambda name: "sk-test"):
            env, why = opencode.key_env("openai/gpt-6.1-sol", {"key": "keyring:openai_api_key"}, {})
        self.assertEqual((env, why), ({"OPENAI_API_KEY": "sk-test"}, None))

    def test_the_catalogue_fetch_sends_a_user_agent(self):
        os.remove(os.path.join(self.home, "opencode-models.json"))
        seen = {}
        def urlopen(req, timeout):
            seen["ua"] = req.get_header("User-agent")
            raise OSError("offline")
        with mock.patch.dict(os.environ, {"ESTATE_BRAIN_OFFLINE": ""}), \
                mock.patch("urllib.request.urlopen", urlopen):
            opencode.catalogue_path()
        self.assertTrue(seen["ua"].startswith("estate-brain/"))

    def test_an_unreadable_key_or_unknown_model_refuses_before_spawning(self):
        exe = self.fake([_text("should not run")])
        with mock.patch("estate_brain.keys.keyring", lambda name: None):
            res = self.run_(exe, Task(prompt="x"), profile={"key": "keyring:openai_api_key"})
        self.assertIn("no API key", res.error)
        res = self.run_(exe, Task(prompt="x"), model="openai/gpt-9-imaginary")
        self.assertIn("not in OpenCode's model catalogue", res.error)
        self.assertFalse(os.path.exists(self.envdump))

    def test_the_spend_ceiling_stops_the_run_at_the_crossing_step(self):
        exe = self.fake([_step(0.03), _text("half"), _step(0.03), _text("never"), _step(0.03)],
                        sleep_after=30)
        t0 = time.monotonic()
        res = self.run_(exe, Task(prompt="x", budget_usd=0.05, persist=False))
        self.assertLess(time.monotonic() - t0, 15)              # killed, not waited out
        self.assertTrue(res.error.startswith("budget: Exceeded USD budget"))
        from estate_brain.runcli import BUDGET, classify
        self.assertEqual(classify(res), BUDGET)

    def test_persist_keeps_the_event_stream_as_the_transcript(self):
        exe = self.fake([_text("pong", "ses_abc"), _step(0.001, "ses_abc")])
        self.run_(exe, Task(prompt="x", persist=True))
        day = time.strftime("%Y-%m-%d")
        path = os.path.join(self.home, "opencode-sessions", day, "ses_abc.jsonl")
        lines = [json.loads(x) for x in open(path)]
        self.assertEqual([e["type"] for e in lines], ["text", "step_finish"])
        exe = self.fake([_text("pong", "ses_quiet"), _step(0.001, "ses_quiet")])
        self.run_(exe, Task(prompt="x", persist=False))
        self.assertFalse(os.path.exists(os.path.join(self.home, "opencode-sessions", day, "ses_quiet.jsonl")))

    def test_a_subagent_tier_picks_the_profiles_tier_model(self):
        t = Task(prompt="x", tools=ToolPolicy(files=True, subagents={
            "scout": {"description": "d", "prompt": "p", "tier": "fast"}}))
        cfg = opencode.config_for(t, "openai/gpt-6.1-sol", {"fast": "openai/gpt-6.1-terra"})
        self.assertEqual(cfg["agent"]["scout"]["model"], "openai/gpt-6.1-terra")
        self.assertNotIn("model", opencode.config_for(t, "openai/gpt-6.1-sol")["agent"]["scout"])

    def test_the_catalogue_is_refreshed_when_stale_and_kept_when_the_fetch_fails(self):
        path = os.path.join(self.home, "opencode-models.json")
        old = time.time() - 3 * 24 * 3600
        os.utime(path, (old, old))
        with mock.patch.dict(os.environ, {"ESTATE_BRAIN_OFFLINE": ""}), \
                mock.patch("urllib.request.urlopen", side_effect=OSError("offline")):
            self.assertEqual(opencode.catalogue_path(), path)       # stale copy kept
        body = json.dumps({"openai": {"env": ["OPENAI_API_KEY"], "models": {"new": {}}}}).encode()
        fake = mock.MagicMock()
        fake.__enter__.return_value.read.return_value = body
        with mock.patch.dict(os.environ, {"ESTATE_BRAIN_OFFLINE": ""}), \
                mock.patch("urllib.request.urlopen", return_value=fake):
            opencode.catalogue_path()
        self.assertIn("new", json.load(open(path))["openai"]["models"])


if __name__ == "__main__":
    unittest.main()
