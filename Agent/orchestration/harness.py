"""Execution harness: every tool call goes through ``ToolRunner.run``.

It enforces, outside any prompt: per-agent tool allowlists, argument validation (context fields
are forbidden), a global executed-call budget, per-call timeouts and an end-to-end deadline,
context-identifier consistency of every result, and a complete trace. A result that arrives after
a timeout or cancellation is never read (late results are ignored).
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from pydantic import ValidationError

from Agent.schemas.contracts import TOOL_SPECS, RequestContext, TraceEvent
from Agent.tools.tools import ToolError, ToolSuite

log = logging.getLogger("agent.harness")
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")
ID_FIELDS = ("request_id", "engine_id", "dataset_split", "cutoff_cycle", "model_version", "corpus_version")


class HarnessError(RuntimeError):
    kind = "error"


class PermissionDenied(HarnessError):
    kind = "rejected"


class ArgumentRejected(HarnessError):
    kind = "rejected"


class BudgetExceeded(HarnessError):
    kind = "budget_exceeded"


class ToolTimeout(HarnessError):
    kind = "timeout"


class DeadlineExceeded(HarnessError):
    kind = "deadline"


class ContextMismatch(HarnessError):
    kind = "context_mismatch"


class ToolFailed(HarnessError):
    kind = "error"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe_args(args: Any) -> Dict[str, Any]:
    if not isinstance(args, dict):
        return {"_raw": str(args)[:120]}
    return {str(k): (v if len(str(v)) <= 120 else str(v)[:117] + "...") for k, v in args.items()}


class ToolRunner:
    def __init__(self, suite: ToolSuite, ctx: RequestContext, allowlists: Dict[str, Set[str]], budget: int,
                 tool_timeout_s: float, deadline_ts: float):
        self.suite, self.ctx, self.allowlists = suite, ctx, allowlists
        self.budget, self.tool_timeout_s, self.deadline_ts = budget, tool_timeout_s, deadline_ts
        self.events: List[TraceEvent] = []
        self.executed = 0
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def _record(self, agent: str, tool: str, args: Any, t0: float, status: str, result_ref: Optional[str] = None,
                cache: Optional[str] = None, error: Optional[str] = None, started: Optional[str] = None) -> None:
        self.events.append(TraceEvent(
            request_id=self.ctx.request_id, seq=len(self.events) + 1, agent=agent, tool=tool, args=_safe_args(args),
            started_utc=started or _utc_now(), duration_ms=round((time.monotonic() - t0) * 1000, 2), status=status,
            model_version=self.ctx.model_version, corpus_version=self.ctx.corpus_version, result_ref=result_ref,
            cache_status=cache, error=error))

    def run(self, agent: str, tool: str, raw_args: Optional[Dict[str, Any]] = None):
        raw_args = raw_args or {}
        t0, started = time.monotonic(), _utc_now()
        if self.cancelled:
            self._record(agent, tool, raw_args, t0, "rejected", error="request cancelled", started=started)
            raise HarnessError("request cancelled")
        if tool not in TOOL_SPECS or tool not in self.allowlists.get(agent, set()):
            self._record(agent, tool, raw_args, t0, "rejected", error="tool not allowed for this agent", started=started)
            raise PermissionDenied(f"Tool '{tool}' is not permitted for the {agent} agent.")
        spec = TOOL_SPECS[tool]
        try:
            args = spec.input_model.model_validate(raw_args)
        except ValidationError as e:
            msg = "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors())
            self._record(agent, tool, raw_args, t0, "rejected", error=f"invalid arguments: {msg}", started=started)
            raise ArgumentRejected(f"Invalid arguments for {tool}: {msg}")
        if self.executed >= self.budget:
            self._record(agent, tool, raw_args, t0, "budget_exceeded", error=f"budget of {self.budget} tool calls used", started=started)
            raise BudgetExceeded(f"Tool-call budget of {self.budget} is used up.")
        remaining = self.deadline_ts - time.monotonic()
        if remaining <= 0:
            self._record(agent, tool, raw_args, t0, "deadline", error="end-to-end deadline reached", started=started)
            raise DeadlineExceeded("The end-to-end deadline was reached.")
        self.executed += 1
        fut = _POOL.submit(self.suite.call, tool, self.ctx, args)
        try:
            out = fut.result(timeout=min(self.tool_timeout_s, remaining))
        except FutureTimeout:
            fut.cancel()  # a running call cannot be stopped, but its result is never read
            self._record(agent, tool, raw_args, t0, "timeout", error="tool call timed out", started=started)
            raise ToolTimeout(f"{tool} timed out.")
        except ToolError as e:
            self._record(agent, tool, raw_args, t0, "error", error=str(e), started=started)
            raise ToolFailed(str(e))
        except Exception as e:  # unexpected tool bug: surfaced, never swallowed into an answer
            self._record(agent, tool, raw_args, t0, "error", error=f"{type(e).__name__}: {e}", started=started)
            raise ToolFailed(f"{tool} failed unexpectedly: {type(e).__name__}")
        if self.cancelled:
            self._record(agent, tool, raw_args, t0, "rejected", error="late result ignored after cancellation", started=started)
            raise HarnessError("request cancelled")
        if not isinstance(out, spec.output_model):
            self._record(agent, tool, raw_args, t0, "error", error="malformed tool result", started=started)
            raise ToolFailed(f"{tool} returned a malformed result.")
        bad = [f for f in ID_FIELDS if getattr(out, f) != getattr(self.ctx, f)]
        if bad:
            log.error("context mismatch from tool %s for request %s: result differs in %s; result discarded", tool, self.ctx.request_id, bad)
            self._record(agent, tool, raw_args, t0, "context_mismatch", error=f"result identifiers differ from the request context: {bad}", started=started)
            raise ContextMismatch(f"{tool} returned a result for a different context ({', '.join(bad)}); it was discarded.")
        self._record(agent, tool, raw_args, t0, "ok", result_ref=f"{tool}#{self.executed}", cache=getattr(out, "cache_status", None), started=started)
        return out
