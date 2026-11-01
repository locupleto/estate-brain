"""Offline tests: no model calls, no network."""

import json
import unittest

from estate_brain import Task, ToolPolicy
from estate_brain.engines import claude_code, opencode
from estate_brain.evals.replay import infer_schema
from estate_brain.evals.score import contract_check
from estate_brain.structured import extract_json, validate
from estate_brain.task import Result

WS = "/tmp/ws"
CAL = {"command": ["/py", "/srv.py"], "env": {"A": "1"}}


class StructuredTest(unittest.TestCase):
    def test_extract_plain_fenced_and_trailing(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('Sure:\n```json\n{"a": 2}\n```'), {"a": 2})
        self.assertEqual(extract_json('blah {"a": {"b": 3}} done'), {"a": {"b": 3}})
        self.assertIsNone(extract_json("no json here"))

    def test_validate_subset(self):
        s = {"type": "object", "required": ["x", "y"],
             "properties": {"x": {"type": "integer"}, "y": {"type": "array", "items": {"type": "string"}},
                            "v": {"type": "string", "enum": ["a", "b"]}}}
        self.assertEqual(validate({"x": 1, "y": ["a"]}, s), [])
        self.assertTrue(validate({"x": True, "y": []}, s))     # bool is not an integer
        self.assertTrue(validate({"x": 1}, s))                  # missing y
        self.assertTrue(validate({"x": 1, "y": [2]}, s))        # item type
        self.assertTrue(validate({"x": 1, "y": [], "v": "c"}, s))


class ClaudeTranslationTest(unittest.TestCase):
    def test_bare_mode(self):
        t = Task(prompt="hi", system="persona", tools=ToolPolicy(files=False))
        cmd, stdin = claude_code.command(t, "haiku", "claude")
        self.assertEqual(cmd[cmd.index("--tools") + 1], "")
        self.assertEqual(cmd[cmd.index("--system-prompt") + 1], "persona")
        self.assertNotIn("--settings", cmd)
        self.assertEqual(stdin, "hi")
        self.assertEqual(cmd[cmd.index("--setting-sources") + 1], "")

    def test_agent_fence(self):
        t = Task(prompt="p", system="persona", workdir=WS,
                 tools=ToolPolicy(write_paths=[f"{WS}/data/memory"], web=True,
                                  mcp_servers={"cal": CAL}, mcp_deny=["cal.book"]))
        cmd, _ = claude_code.command(t, "sonnet", "claude")
        perms = json.loads(cmd[cmd.index("--settings") + 1])["permissions"]
        self.assertIn(f"Edit(/{WS}/data/memory/**)", perms["allow"])
        self.assertIn("mcp__cal", perms["allow"])
        self.assertIn("WebSearch", perms["allow"])
        self.assertIn("Bash", perms["deny"])
        self.assertIn("mcp__cal__book", perms["deny"])
        self.assertEqual(cmd[cmd.index("--append-system-prompt") + 1], "persona")
        mcp = json.loads(cmd[cmd.index("--mcp-config") + 1])["mcpServers"]["cal"]
        self.assertEqual((mcp["command"], mcp["args"], mcp["env"]), ("/py", ["/srv.py"], {"A": "1"}))

    def test_read_paths_are_pinned_and_added(self):
        t = Task(prompt="p", workdir=WS, tools=ToolPolicy(read_paths=["/repo/x"]))
        cmd, _ = claude_code.command(t, "opus", "claude")
        allow = json.loads(cmd[cmd.index("--settings") + 1])["permissions"]["allow"]
        for tool in ("Read", "Glob", "Grep"):
            self.assertIn(f"{tool}(//repo/x/**)", allow)
        self.assertEqual(cmd[cmd.index("--add-dir") + 1], "/repo/x")

    def test_env_strips_api_keys(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "x"}):
            self.assertNotIn("ANTHROPIC_API_KEY", claude_code.env_for())

    def test_neutral_names(self):
        self.assertEqual(claude_code.neutral_tool_name("mcp__marketdata-db__get_eod"), "mcp:marketdata-db.get_eod")
        self.assertEqual(claude_code.neutral_tool_name("Write"), "edit")


class OpenCodeTranslationTest(unittest.TestCase):
    def test_permissions_deny_by_default(self):
        t = Task(prompt="p", workdir=WS, tools=ToolPolicy(write_paths=[f"{WS}/data/memory"],
                                                          mcp_servers={"cal": CAL}, mcp_deny=["cal.book"]))
        cfg = opencode.config_for(t, "openai/x")
        p = cfg["permission"]
        self.assertEqual(p["*"], "deny")
        self.assertEqual(p["bash"], "deny")
        self.assertEqual(p["cal_*"], "allow")           # MCP must be allowed by name
        self.assertEqual(p["cal_book"], "deny")
        self.assertEqual(p["edit"]["*"], "deny")
        self.assertEqual(p["edit"]["data/memory/**"], "allow")  # relative form for in-workspace paths
        self.assertEqual(cfg["mcp"]["cal"]["command"], ["/py", "/srv.py"])
        self.assertEqual(cfg["mcp"]["cal"]["environment"], {"A": "1"})
        self.assertNotIn("agent", cfg)

    def test_isolated_env(self):
        t = Task(prompt="p", system="persona")
        import tempfile
        env = opencode.env_for(t, "openai/x", tempfile.mkdtemp())
        for k in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "OPENCODE_CONFIG_CONTENT",
                  "OPENCODE_DISABLE_CLAUDE_CODE", "OPENCODE_DISABLE_EXTERNAL_SKILLS"):
            self.assertIn(k, env)
        self.assertIn("estate", json.loads(env["OPENCODE_CONFIG_CONTENT"])["agent"])

    def test_no_model_is_an_error_not_a_guess(self):
        self.assertTrue(opencode.run(Task(prompt="p")).error)


class EvalScoringTest(unittest.TestCase):
    def test_infer_schema_and_contracts(self):
        ref = {"say": "x", "positions": [{"id": 1, "text": "t"}]}
        s = infer_schema(ref)
        self.assertEqual(s["required"], ["say", "positions"])
        case = {"contract": "structured", "reference_structured": ref}
        ok, _ = contract_check(case, Result(structured={"say": "y", "positions": [{"id": 2, "text": "u"}]}))
        self.assertTrue(ok)
        ok, _ = contract_check(case, Result(structured={"say": "y"}))
        self.assertFalse(ok)
        self.assertTrue(contract_check({"contract": "wrapper-json"},
                                       Result(text='{"action":"text_response","message":"hi"}'))[0])
        self.assertTrue(contract_check({"contract": "wrapper-json"}, Result(text="plain"))[0])  # wrapper fallback
        self.assertFalse(contract_check({"contract": "wrapper-json"}, Result(text="  "))[0])
        self.assertTrue(contract_check({"contract": "door"}, Result(text="DOOR: ignore\nfine"))[0])
        self.assertFalse(contract_check({"contract": "door"}, Result(error="timeout"))[0])


class CompareTest(unittest.TestCase):
    def test_compare_marks_deltas(self):
        import tempfile
        from pathlib import Path
        from estate_brain.evals.runner import compare
        base, cand = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        (base / "summary.json").write_text(json.dumps(
            [{"case": "a", "judge": {"overall": 4}}, {"case": "b", "judge": {"overall": 3}}]))
        (cand / "summary.json").write_text(json.dumps(
            [{"case": "a", "judge": {"overall": 2}}, {"case": "b", "judge": {}}]))
        out = compare(base, cand)
        self.assertIn("-2.0  ▼", out)
        self.assertIn("1 clearly worse", out)


class ResolveAndLedgerTest(unittest.TestCase):
    def test_resolution_order(self):
        from estate_brain.resolve import resolve
        t = Task(prompt="p", app="photo-ocr", model="sonnet")
        self.assertEqual(resolve(t, environ={}), ("claude-code", "sonnet"))
        self.assertEqual(resolve(t, "opencode", "openai/x", environ={}), ("opencode", "openai/x"))
        env = {"BRAIN_ENGINE": "opencode", "BRAIN_MODEL": "openai/a"}
        self.assertEqual(resolve(t, environ=env), ("opencode", "openai/a"))
        env["BRAIN_PHOTO_OCR_MODEL"] = "opus"          # per-app beats estate-wide
        self.assertEqual(resolve(t, environ=env), ("claude-code", "opus"))
        # an engine switch never drags a Claude alias along
        self.assertEqual(resolve(t, environ={"BRAIN_ENGINE": "opencode"}), ("opencode", None))

    def test_ledger_writes_metadata_only(self):
        import os, tempfile
        from unittest import mock
        from estate_brain import ledger
        p = os.path.join(tempfile.mkdtemp(), "l.jsonl")
        with mock.patch.dict(os.environ, {"BRAIN_LEDGER": p}):
            ledger.record(Task(prompt="SECRET", app="a"), Result(engine="e", model="m", text="REPLY"))
        line = open(p).read()
        self.assertIn('"app": "a"', line)
        self.assertNotIn("SECRET", line)
        self.assertNotIn("REPLY", line)
        with mock.patch.dict(os.environ, {"BRAIN_LEDGER": "off"}):
            self.assertIsNone(ledger.path())

    def test_image_bytes_and_unknown_engine(self):
        from estate_brain import run
        t = Task(prompt="p", images=[b"\xff\xd8jpeg"], tools=ToolPolicy(files=False))
        _, stdin = claude_code.command(t, "sonnet", "claude")
        msg = json.loads(stdin)["message"]["content"]
        self.assertEqual(msg[0]["source"]["media_type"], "image/jpeg")
        self.assertEqual(msg[-1]["text"], "p")
        self.assertIn("unknown engine", run(t, engine="nope", ledger=False).error)


if __name__ == "__main__":
    unittest.main()
