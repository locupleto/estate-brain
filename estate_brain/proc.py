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
        self._stderr: list[bytes] = []
        if cwd:
            # Tools trust the inherited $PWD over the real cwd (OpenCode
            # resolved its project from it, 2026-11-01): keep them in step.
            env = dict(env, PWD=cwd)
            self._popen(cmd, env, cwd, stdin_text)
        else:
            # No cwd to set: posix_spawn, no fork. A caller that has loaded
            # non-fork-safe frameworks (CoreAudio in the voice assistant)
            # can crash in a forked child before exec.
            self._spawn(cmd, env, stdin_text)
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        if stdin_text is not None:
            threading.Thread(target=self._feed, args=(stdin_text,), daemon=True).start()
        self._timer = threading.Timer(timeout, self._expire)
        self._timer.daemon = True
        self._timer.start()

    def _popen(self, cmd, env, cwd, stdin_text) -> None:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)
        self.pid = proc.pid
        self._wait = proc.wait
        self._stdin, self._stdout, self._stderr_f = proc.stdin, proc.stdout, proc.stderr

    def _spawn(self, cmd, env, stdin_text) -> None:
        out_r, out_w = os.pipe()
        err_r, err_w = os.pipe()
        actions = [(os.POSIX_SPAWN_DUP2, out_w, 1), (os.POSIX_SPAWN_DUP2, err_w, 2),
                   (os.POSIX_SPAWN_CLOSE, out_r), (os.POSIX_SPAWN_CLOSE, err_r)]
        in_w = None
        if stdin_text is not None:
            in_r, in_w = os.pipe()
            actions += [(os.POSIX_SPAWN_DUP2, in_r, 0), (os.POSIX_SPAWN_CLOSE, in_w)]
        else:
            actions.append((os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0))
        exe = cmd[0] if os.path.isabs(cmd[0]) else (shutil.which(cmd[0], path=env.get("PATH")) or cmd[0])
        try:
            self.pid = os.posix_spawn(exe, cmd, env, file_actions=actions, setpgroup=0)
        finally:
            os.close(out_w)
            os.close(err_w)
            if stdin_text is not None:
                os.close(in_r)
        self._stdin = os.fdopen(in_w, "wb") if in_w is not None else None
        self._stdout = os.fdopen(out_r, "rb")
        self._stderr_f = os.fdopen(err_r, "rb")
        self._wait = self._waitpid

    def _waitpid(self, timeout: float | None = None) -> int:
        _, status = os.waitpid(self.pid, 0)
        return os.waitstatus_to_exitcode(status)

    def _feed(self, text: str) -> None:
        try:
            self._stdin.write(text.encode())
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                self._stdin.close()
            except OSError:
                pass

    def _drain_stderr(self) -> None:
        for chunk in iter(lambda: self._stderr_f.read(4096), b""):
            self._stderr.append(chunk)
            if sum(map(len, self._stderr)) > 200_000:
                self._stderr = self._stderr[-10:]

    def _expire(self) -> None:
        self.timed_out = True
        self.kill()

    def lines(self) -> Iterator[str]:
        for raw in self._stdout:
            yield raw.decode("utf-8", "replace")

    def kill(self) -> None:
        if getattr(self, "_reaped", False):
            return
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def close(self) -> int:
        self._timer.cancel()
        self.kill()
        try:
            rc = self._wait(timeout=10)
        except (subprocess.TimeoutExpired, ChildProcessError):
            rc = -9
        self._reaped = True
        try:
            self._stdout.close()
        except OSError:
            pass
        return rc

    def stderr_tail(self, n: int = 600) -> str:
        return b"".join(self._stderr).decode("utf-8", "replace")[-n:].strip()


def find_binary(name: str, env_var: str, candidates: tuple[str, ...]) -> str | None:
    explicit = os.environ.get(env_var)
    if explicit:
        # A path that is not there must read as "not found" (an error Result
        # the fallback chain can act on), never as a spawn exception later.
        return explicit if os.access(os.path.expanduser(explicit), os.X_OK) else None
    found = shutil.which(name)
    if found:
        return found
    for c in candidates:
        c = os.path.expanduser(c)
        if os.access(c, os.X_OK):
            return c
    return None
