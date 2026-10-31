"""Child-process handling shared by the CLI engines.

One rule set, learned the hard way in the bot wrapper: the child gets its
own process group (so a timeout kills the CLI AND its MCP children), stdin
is fed and closed (an open stdin can stall a CLI forever), and a watchdog
enforces the wall-clock limit.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
from typing import Iterator


class Run:
    """A streaming child process. Iterate `lines()` for stdout lines."""

    def __init__(self, cmd: list[str], *, env: dict, cwd: str | None,
                 stdin_text: str | None, timeout: float):
        self.timed_out = False
        if cwd:
            # Tools trust the inherited $PWD over the real cwd (OpenCode
            # resolved its project from it, 2026-11-01): keep them in step.
            env = dict(env, PWD=cwd)
        self.proc = subprocess.Popen(
            cmd, cwd=cwd or None, env=env,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)
        self._stderr: list[bytes] = []
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        if stdin_text is not None:
            threading.Thread(target=self._feed, args=(stdin_text,), daemon=True).start()
        self._timer = threading.Timer(timeout, self._expire)
        self._timer.daemon = True
        self._timer.start()

    def _feed(self, text: str) -> None:
        try:
            self.proc.stdin.write(text.encode())
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                self.proc.stdin.close()
            except OSError:
                pass

    def _drain_stderr(self) -> None:
        for chunk in iter(lambda: self.proc.stderr.read(4096), b""):
            self._stderr.append(chunk)
            if sum(map(len, self._stderr)) > 200_000:
                self._stderr = self._stderr[-10:]

    def _expire(self) -> None:
        self.timed_out = True
        self.kill()

    def lines(self) -> Iterator[str]:
        for raw in self.proc.stdout:
            yield raw.decode("utf-8", "replace")

    def kill(self) -> None:
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def close(self) -> int:
        self._timer.cancel()
        self.kill()
        try:
            return self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            return -9

    def stderr_tail(self, n: int = 600) -> str:
        return b"".join(self._stderr).decode("utf-8", "replace")[-n:].strip()


def find_binary(name: str, env_var: str, candidates: tuple[str, ...]) -> str | None:
    explicit = os.environ.get(env_var)
    if explicit:
        return explicit
    found = shutil.which(name)
    if found:
        return found
    for c in candidates:
        c = os.path.expanduser(c)
        if os.access(c, os.X_OK):
            return c
    return None
