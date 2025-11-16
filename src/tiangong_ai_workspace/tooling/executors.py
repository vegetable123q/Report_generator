"""
Execution utilities that expose controlled shell and Python runtimes.

These helpers allow higher-level agents (Codex, LangChain, workspace agent, etc.) to
reliably run commands while capturing structured outputs that can be surfaced to
callers or persisted for auditing.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import shlex
import signal
import subprocess
import textwrap
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

LOGGER = logging.getLogger(__name__)

__all__ = [
    "PythonExecutionResult",
    "PythonExecutor",
    "ShellExecutionResult",
    "ShellExecutor",
]


@dataclass(slots=True)
class ShellExecutionResult:
    """Structured result for shell command execution."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    cwd: Path
    duration: float
    timestamp: float

    def to_dict(self) -> MutableMapping[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "cwd": str(self.cwd),
            "duration": self.duration,
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class ShellExecutor:
    """
    Small wrapper around `subprocess.run` with predictable output structure.
    """

    workdir: Path = Path.cwd()
    default_timeout: int = 120
    allowed_binaries: tuple[str, ...] | None = None
    env: Mapping[str, str] | None = None

    def run(self, command: str, *, timeout: Optional[int] = None) -> ShellExecutionResult:
        self._validate_command(command)
        effective_timeout = timeout or self.default_timeout
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=self.workdir,
            timeout=effective_timeout,
            env=self._build_env(),
        )
        duration = time.perf_counter() - started
        timestamp = time.time()
        LOGGER.debug("ShellExecutor ran '%s' (timeout=%s, exit=%s)", command, effective_timeout, completed.returncode)
        return ShellExecutionResult(
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            cwd=self.workdir,
            duration=duration,
            timestamp=timestamp,
        )

    def _validate_command(self, command: str) -> None:
        if not self.allowed_binaries:
            return
        try:
            first_token = shlex.split(command, posix=True)[0]
        except (IndexError, ValueError) as exc:  # pragma: no cover - malformed command
            raise PermissionError("Unable to parse shell command.") from exc
        binary = Path(first_token).name
        if binary not in self.allowed_binaries:
            raise PermissionError(f"Command '{binary}' is not permitted in this workspace.")

    def _build_env(self) -> Mapping[str, str] | None:
        if not self.env:
            return None
        merged: MutableMapping[str, str] = dict(os.environ)
        merged.update(self.env)
        return merged


@dataclass(slots=True)
class PythonExecutionResult:
    """Structured result for arbitrary Python code execution."""

    code: str
    stdout: str
    stderr: str
    globals_used: Mapping[str, Any]
    duration: float
    timestamp: float
    timed_out: bool

    def to_dict(self) -> MutableMapping[str, Any]:
        return {
            "code": self.code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration": self.duration,
            "timestamp": self.timestamp,
            "timed_out": self.timed_out,
        }


class PythonExecutor:
    """
    Executes Python source strings in a constrained namespace.

    The executor captures stdout / stderr and surfaces them alongside any
    exceptions. Agents can reuse a single executor instance to preserve global
    state between invocations if desired.
    """

    def __init__(
        self,
        *,
        shared_globals: Optional[MutableMapping[str, Any]] = None,
        max_execution_seconds: Optional[int] = 90,
    ) -> None:
        self._globals: MutableMapping[str, Any] = shared_globals or {"__name__": "__agent_exec__"}
        self._max_execution_seconds = max_execution_seconds

    def run(self, code: str, *, locals_override: Optional[MutableMapping[str, Any]] = None) -> PythonExecutionResult:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        local_vars: MutableMapping[str, Any] = locals_override or {}

        compiled_code = compile(code, "<agent-python>", "exec")

        timed_out = False
        started = time.perf_counter()
        timestamp = time.time()

        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                self._execute_with_timeout(compiled_code, local_vars)
            except TimeoutError:
                timed_out = True
                stderr_buffer.write(f"Execution timed out after {self._max_execution_seconds} seconds.\n")
            except Exception:  # pragma: no cover - error text captured for caller
                traceback.print_exc(file=stderr_buffer)

        stdout_value = stdout_buffer.getvalue()
        stderr_value = stderr_buffer.getvalue()
        duration = time.perf_counter() - started
        return PythonExecutionResult(
            code=textwrap.dedent(code),
            stdout=stdout_value,
            stderr=stderr_value,
            globals_used=dict(self._globals),
            duration=duration,
            timestamp=timestamp,
            timed_out=timed_out,
        )

    def _execute_with_timeout(self, compiled_code: Any, local_vars: MutableMapping[str, Any]) -> None:
        if not self._max_execution_seconds or self._max_execution_seconds <= 0 or not hasattr(signal, "SIGALRM"):
            exec(compiled_code, self._globals, local_vars)
            return

        previous_handler = signal.getsignal(signal.SIGALRM)

        def _handle_timeout(signum, frame):  # pragma: no cover - exercised via TimeoutError path
            raise TimeoutError("PythonExecutor timed out")

        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self._max_execution_seconds)
        try:
            exec(compiled_code, self._globals, local_vars)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
