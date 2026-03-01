"""Tool loop detection — catches stuck LLM tool-call patterns early.

Adapted from OpenClaw's tool-loop-detection.ts, simplified for Syne.
Uses a sliding window of recent tool calls with 3 detection strategies:
  1. Generic repeat — same tool+args called consecutively
  2. Ping-pong — two tools alternating with same args
  3. Circuit breaker — one tool+args combo dominates the window
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

WINDOW_SIZE = 30


@dataclass
class ToolCallRecord:
    tool_name: str
    args_hash: str        # SHA-256[:16] of sorted JSON args
    result_hash: str = "" # SHA-256[:16] of result string (filled after execution)
    round_num: int = 0


@dataclass
class LoopDetectionResult:
    stuck: bool = False        # True = should abort
    level: str = "none"        # "none", "warning", "critical"
    detector: str = ""         # Which strategy triggered
    count: int = 0             # How many repetitions found
    message: str = ""          # Human-readable description


def _hash(data: str) -> str:
    """Short hash for dedup comparison."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _args_hash(args: dict) -> str:
    """Deterministic hash of tool arguments."""
    return _hash(json.dumps(args, sort_keys=True, default=str))


class ToolLoopDetector:
    """Sliding-window loop detector for LLM tool calls."""

    def __init__(self, window_size: int = WINDOW_SIZE):
        self._window_size = window_size
        self._history: list[ToolCallRecord] = []

    def record_call(self, tool_name: str, args: dict, round_num: int) -> ToolCallRecord:
        """Record a tool call BEFORE execution. Returns the record for later update."""
        record = ToolCallRecord(
            tool_name=tool_name,
            args_hash=_args_hash(args),
            round_num=round_num,
        )
        self._history.append(record)
        # Trim to window size
        if len(self._history) > self._window_size:
            self._history = self._history[-self._window_size:]
        return record

    def record_result(self, record: ToolCallRecord, result: str):
        """Update a record with the execution result hash."""
        record.result_hash = _hash(str(result))

    def detect(self) -> LoopDetectionResult:
        """Run all detection strategies. Call AFTER record_call, BEFORE execution."""
        if len(self._history) < 3:
            return LoopDetectionResult()

        # Run strategies in order of specificity
        for strategy in (self._detect_generic_repeat, self._detect_pingpong, self._detect_circuit_breaker):
            result = strategy()
            if result.level != "none":
                return result

        return LoopDetectionResult()

    def reset(self):
        """Clear history. Call at start of a new conversation turn."""
        self._history.clear()

    # ── Strategy 1: Generic repeat ──────────────────────────────

    def _detect_generic_repeat(self) -> LoopDetectionResult:
        """Same tool+args called consecutively."""
        if not self._history:
            return LoopDetectionResult()

        latest = self._history[-1]
        key = (latest.tool_name, latest.args_hash)

        count = 0
        for record in reversed(self._history):
            if (record.tool_name, record.args_hash) == key:
                count += 1
            else:
                break

        if count >= 8:
            return LoopDetectionResult(
                stuck=True,
                level="critical",
                detector="generic_repeat",
                count=count,
                message=f"Tool '{latest.tool_name}' called {count}× consecutively with identical arguments",
            )
        if count >= 5:
            return LoopDetectionResult(
                stuck=False,
                level="warning",
                detector="generic_repeat",
                count=count,
                message=f"Tool '{latest.tool_name}' called {count}× consecutively with identical arguments",
            )
        return LoopDetectionResult()

    # ── Strategy 2: Ping-pong ───────────────────────────────────

    def _detect_pingpong(self) -> LoopDetectionResult:
        """Two tools alternating: A→B→A→B with same args per tool."""
        if len(self._history) < 4:
            return LoopDetectionResult()

        latest = self._history[-1]
        prev = self._history[-2]

        # Need two different tools
        if latest.tool_name == prev.tool_name:
            return LoopDetectionResult()

        a_key = (latest.tool_name, latest.args_hash)
        b_key = (prev.tool_name, prev.args_hash)

        # Count alternations backwards
        alternations = 0
        for i in range(len(self._history) - 1, -1, -2):
            if i - 1 < 0:
                break
            rec_a = self._history[i]
            rec_b = self._history[i - 1]
            if (rec_a.tool_name, rec_a.args_hash) == a_key and \
               (rec_b.tool_name, rec_b.args_hash) == b_key:
                alternations += 1
            else:
                break

        if alternations >= 8:
            return LoopDetectionResult(
                stuck=True,
                level="critical",
                detector="pingpong",
                count=alternations,
                message=f"Ping-pong detected: '{latest.tool_name}' ↔ '{prev.tool_name}' alternating {alternations}× with identical arguments",
            )
        if alternations >= 5:
            return LoopDetectionResult(
                stuck=False,
                level="warning",
                detector="pingpong",
                count=alternations,
                message=f"Ping-pong detected: '{latest.tool_name}' ↔ '{prev.tool_name}' alternating {alternations}× with identical arguments",
            )
        return LoopDetectionResult()

    # ── Strategy 3: Circuit breaker ─────────────────────────────

    def _detect_circuit_breaker(self) -> LoopDetectionResult:
        """Fallback: one tool+args combo dominates the entire window."""
        if len(self._history) < self._window_size:
            return LoopDetectionResult()

        # Count occurrences of each (tool, args_hash) combo
        counts: dict[tuple[str, str], int] = {}
        for record in self._history:
            key = (record.tool_name, record.args_hash)
            counts[key] = counts.get(key, 0) + 1

        # Find the dominant combo
        max_key, max_count = max(counts.items(), key=lambda x: x[1])

        if max_count >= self._window_size:  # 30 out of 30
            return LoopDetectionResult(
                stuck=True,
                level="critical",
                detector="circuit_breaker",
                count=max_count,
                message=f"Circuit breaker: '{max_key[0]}' called {max_count}× in last {self._window_size} tool calls",
            )
        return LoopDetectionResult()
