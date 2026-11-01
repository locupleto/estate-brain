"""claude_rules (permission files -> ToolPolicy) and `brain run`."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from estate_brain import Task, ToolPolicy
from estate_brain.claude_rules import PolicyError, load_rules, policy_from, roundtrip_problems
from estate_brain.engines import claude_code, opencode

FIX = Path(__file__).parent / "fixtures" / "claude_rules"
MCP = FIX / "mcp.json"
PROJECT = "/home/u/Public/ws"
ROOT = Path(__file__).resolve().parents[1]


def _agents_for(path: Path):
    a = path.with_name(path.name.split("__")[0] + "__agents-research.json")
    return json.loads(a.read_text()) if "research" in path.name and a.exists() else None


class RoundTripTest(unittest.TestCase):
    def test_every_real_permission_file_round_trips(self):
        files = sorted(p for p in FIX.glob("*.json") if "permissions" in p.name)
        self.assertGreaterEqual(len(files), 11)
        for f in files:
            with self.subTest(f.name):
                rules = load_rules(f, PROJECT)
                pol = policy_from(rules, MCP, _agents_for(f))
                settings = claude_code.settings_for(Task(prompt="x", tools=pol))
                self.assertEqual(roundtrip_problems(rules, pol, settings), [])

    def test_jarvis_is_an_admin(self):
        pol = policy_from(load_rules(FIX / "the-armory__bot_permissions_JARVIS.json", PROJECT), MCP)
        self.assertTrue(pol.shell and pol.web and pol.skills and pol.files)
        self.assertEqual(pol.write_paths, ["*"])
        allow = claude_code.settings_for(Task(prompt="x", tools=pol))["permissions"]["allow"]
        self.assertIn("Bash", allow)
        self.assertIn("Edit", allow)

    def test_librarian_tool_level_grant(self):
        pol = policy_from(load_rules(FIX / "second-brain__bot_permissions_LIBRARIAN.json", PROJECT), MCP)
        self.assertEqual(pol.mcp_allow, ["ai-search.search_x"])
        self.assertIn("ai-search", pol.mcp_servers)
        allow = claude_code.settings_for(Task(prompt="x", tools=pol))["permissions"]["allow"]
        self.assertIn("mcp__ai-search__search_x", allow)
        self.assertNotIn("mcp__ai-search", allow)

    def test_only_named_servers_are_mounted(self):
        pol = policy_from(load_rules(FIX / "the-armory__bot_permissions.json", PROJECT), MCP)
        self.assertNotIn("unused-server", pol.mcp_servers)
        self.assertEqual(pol.mcp_servers["telegram-chat-db"],
                         {"command": ["/bin/echo", "chat"], "env": {"X": "1"}})

    def test_single_files_stay_files_existing_or_not(self):
        pol = policy_from(load_rules(FIX / "assistant-lab__bot_permissions.json", PROJECT), MCP)
        self.assertIn(f"{PROJECT}/bots/USER.md.bak", pol.write_files)
        self.assertIn(f"{PROJECT}/data/memory/**", pol.write_paths)

    def test_research_subagent_keeps_model_and_effort(self):
        f = FIX / "the-armory__permissions-research-turn.json"
        pol = policy_from(load_rules(f, PROJECT), MCP, _agents_for(f))
        scout = pol.subagents["web-scout"]
        self.assertEqual((scout["tier"], scout.get("effort"), scout["web"]), ("standard", "medium", True))
        self.assertEqual(claude_code._agent(scout)["model"], "sonnet")
        self.assertEqual(claude_code._agent(scout)["effort"], "medium")

    def test_refusals(self):
        bad = [
            {"allow": ["Bash(ls:*)"]},
            {"allow": ["Read"]},                      # without Glob and Grep
            {"allow": ["WebFetch"]},
            {"allow": ["Edit"]},
            {"allow": ["Edit(relative/**)"]},
            {"allow": ["Agent"]},                     # no subagents to delegate to
            {"allow": ["Bash"], "deny": ["Bash"]},
            {"allow": ["Read", "Glob", "Grep"], "deny": ["Read"]},
            {"allow": [], "ask": ["Bash"]},
        ]
        for perms in bad:
            with self.subTest(perms):
                with self.assertRaises(PolicyError):
                    policy_from({"permissions": perms}, MCP)

    def test_opencode_expresses_the_new_grants(self):
        pol = ToolPolicy(files=True, write_paths=["*"], mcp_servers={"ai-search": {"command": ["x"]}},
                         mcp_allow=["ai-search.search_x"], write_files=["/w/bots/USER.md"])
        perm = opencode.config_for(Task(prompt="x", tools=pol, workdir="/w"), "m")["permission"]
        self.assertEqual(perm["edit"], "allow")
        self.assertEqual(perm.get("ai-search_search_x"), "allow")
        self.assertNotIn("ai-search_*", perm)


class EngineFlagsTest(unittest.TestCase):
    def test_persisted_turn_carries_only_the_retention_setting(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "settings.json").write_text(json.dumps(
                {"cleanupPeriodDays": 120, "hooks": {"Stop": []}, "model": "opus"}))
            os.environ["CLAUDE_CONFIG_DIR"] = d
            try:
                s = claude_code.settings_for(Task(prompt="x", persist=True))
                self.assertEqual(s["cleanupPeriodDays"], 120)
                self.assertNotIn("hooks", s)
                self.assertNotIn("cleanupPeriodDays", claude_code.settings_for(Task(prompt="x")))
                cmd, _ = claude_code.command(Task(prompt="x", tools=ToolPolicy(files=False), persist=True),
                                             "haiku", "claude")
                self.assertIn("--settings", cmd)
            finally:
                del os.environ["CLAUDE_CONFIG_DIR"]

    def test_debug_file(self):
        cmd, _ = claude_code.command(Task(prompt="x", engine_opts={"claude-code": {"debug_file": "/t/d.log"}}),
                                     "opus", "claude")
        self.assertEqual(cmd[cmd.index("--debug-file") + 1], "/t/d.log")


FAKE = textwrap.dedent("""\
    #!/bin/sh
    # fake claude: records argv, answers per $FAKE_MODE
    printf '%s\\n' "$@" > "$FAKE_ARGV"
    cat > /dev/null
    case "$FAKE_MODE" in
      ok)     echo '{"type":"result","is_error":false,"result":"{\\"action\\":\\"text_response\\"}","usage":{}}' ;;
      budget) echo '{"type":"result","is_error":true,"subtype":"error_max_budget_usd","result":"","usage":{}}' ;;
      busy)   echo '{"type":"result","is_error":true,"subtype":"success","result":"API Error: 529 overloaded","usage":{}}' ;;
      fail)   echo boom >&2; exit 1 ;;
      hang)   sleep 60 ;;
    esac
""")


class RunCliTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.fake = Path(self.d, "claude")
        self.fake.write_text(FAKE)
        self.fake.chmod(0o755)
        self.prompt = Path(self.d, "p.txt")
        self.prompt.write_text("hello")
        ws = Path(self.d, "ws")
        (ws / "bots").mkdir(parents=True)
        (ws / ".mcp.json").write_text(MCP.read_text())
        (ws / "bots" / "perm.json").write_text(
            (FIX / "the-armory__bot_permissions.json").read_text().replace(PROJECT, "{{PROJECT_DIR}}"))
        self.spec = Path(self.d, "spec.json")
        self.spec.write_text(json.dumps({"workdir": str(ws), "app": "bots", "role": "T", "model": "opus",
                                         "permissions_file": str(ws / "bots" / "perm.json")}))

    def _run(self, mode, *extra, wait=True):
        env = dict(os.environ, CLAUDE_BIN=str(self.fake), FAKE_MODE=mode, BRAIN_LEDGER="off",
                   FAKE_ARGV=str(Path(self.d, "argv")), PYTHONPATH=str(ROOT))
        args = [sys.executable, "-m", "estate_brain", "run", "--spec", str(self.spec),
                "--prompt-file", str(self.prompt), *extra]
        if not wait:
            return subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return subprocess.run(args, env=env, capture_output=True, text=True, timeout=60)

    def test_ok_prints_the_reply(self):
        r = self._run("ok")
        self.assertEqual((r.returncode, r.stdout), (0, '{"action":"text_response"}'))
        argv = Path(self.d, "argv").read_text().splitlines()
        self.assertIn("--strict-mcp-config", argv)
        self.assertEqual(argv[argv.index("--setting-sources") + 1], "project,local")
        self.assertNotIn("--no-session-persistence", argv)

    def test_exit_codes(self):
        self.assertEqual(self._run("budget").returncode, 4)
        self.assertEqual(self._run("busy").returncode, 5)
        self.assertEqual(self._run("fail").returncode, 1)

    def test_refused_spec_calls_no_model(self):
        self.spec.write_text(json.dumps({"workdir": self.d, "bogus": 1}))
        r = self._run("ok")
        self.assertEqual(r.returncode, 64)
        self.assertFalse(Path(self.d, "argv").exists())

    def test_sigterm_kills_the_engine_too(self):
        p = self._run("hang", wait=False)
        for _ in range(100):
            if Path(self.d, "argv").exists():
                break
            time.sleep(0.05)
        out = subprocess.run(["pgrep", "-f", str(self.fake)], capture_output=True, text=True).stdout.split()
        self.assertTrue(out)
        p.send_signal(signal.SIGTERM)
        self.assertEqual(p.wait(timeout=15), 143)
        time.sleep(0.3)
        alive = subprocess.run(["pgrep", "-f", str(self.fake)], capture_output=True, text=True).stdout.split()
        self.assertEqual(alive, [])


if __name__ == "__main__":
    unittest.main()
