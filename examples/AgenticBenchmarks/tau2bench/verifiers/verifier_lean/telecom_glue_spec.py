"""telecom_glue_spec.py

Glue layer between the Lean PolicyChecker daemon and the rest of the
verifier.

Scope (deliberately narrow):

* Lean handles ONLY the 9 proven DB-driven PRE rules. Each rule is a pure
  function of `(state, args)`. No hypotheses, no history, no POST coaching.
* Everything else (POST coaching, empirical arg-mismatch, hypothesis-driven
  PRE gates) lives in `telecom_python_rules.py`.

Public API (unchanged for `verifier.py`):

    check_all(tool_name, tool_args, conversation, db, **kwargs)
        -> str | list[str] | None

    check_all_results(tool_name, tool_args, result_content, db=None, **kwargs)
        -> list[str]
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from tau2.verifier.telecom_python_rules import PYTHON_PRE_RULES, PYTHON_POST_CHECKS

logger = logging.getLogger(__name__)


# LeanRunner : long-lived subprocess

_DEFAULT_LEAN_BINARY = "."


class _Sentinel:
    def __repr__(self) -> str:
        return "<LEAN_UNAVAILABLE>"


_LEAN_UNAVAILABLE = _Sentinel()


class LeanRunner:
    """Manages a single long-running Lean checker process.

    Lazy spawn on first `query()`, requests serialised by an internal lock,
    auto-restart on death, clean shutdown via atexit. On binary-not-found
    or repeated failures, drops to `available=False` and `query()` returns
    `_LEAN_UNAVAILABLE` for the rest of the process lifetime (one warning).
    """

    def __init__(
        self,
        lean_binary_path: Optional[str] = None,
        timeout_s: float = 5.0,
    ) -> None:
        self._binary_path = self._resolve_binary(lean_binary_path)
        self._timeout_s = timeout_s
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._available: bool = True
        self._unavailable_warned: bool = False
        self._query_count: int = 0
        self._deny_count: int = 0
        atexit.register(self.shutdown)

    @staticmethod
    def _resolve_binary(explicit: Optional[str]) -> Optional[str]:
        if explicit:
            return explicit
        env = os.environ.get("TAU2_LEAN_BINARY")
        if env:
            return env
        which = shutil.which("policychecker")
        if which:
            return which
        if _DEFAULT_LEAN_BINARY and os.path.exists(_DEFAULT_LEAN_BINARY):
            return _DEFAULT_LEAN_BINARY
        return None

    def _warn_unavailable_once(self, reason: str) -> None:
        if not self._unavailable_warned:
            logger.warning(
                "LeanRunner unavailable (%s); Lean rules will be skipped "
                "for the rest of this process.", reason)
            self._unavailable_warned = True
        self._available = False

    def _spawn(self) -> bool:
        if self._binary_path is None:
            self._warn_unavailable_once("no lean binary path resolved")
            return False
        try:
            self._proc = subprocess.Popen(
                [self._binary_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=True,
            )
        except (OSError, FileNotFoundError) as e:
            self._warn_unavailable_once(f"failed to spawn: {e!r}")
            self._proc = None
            return False
        logger.warning("LeanRunner spawned policychecker pid=%d at %s",
                       self._proc.pid, self._binary_path)
        return True

    def _ensure_alive(self) -> bool:
        if not self._available:
            return False
        if self._proc is None or self._proc.poll() is not None:
            return self._spawn()
        return True

    def query(self, request: dict) -> Any:
        """Send one request, return verdict (None=allow, str=deny, sentinel=skip)."""
        with self._lock:
            if not self._ensure_alive():
                return _LEAN_UNAVAILABLE
            assert self._proc is not None
            try:
                payload = json.dumps(request, default=_json_default) + "\n"
                self._proc.stdin.write(payload)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                logger.warning("Lean stdin write failed: %s", e)
                self._kill_proc()
                return _LEAN_UNAVAILABLE

            line = self._readline_with_timeout(self._timeout_s)
            if line is None:
                logger.warning("Lean read timed out (%.2fs); restarting.",
                               self._timeout_s)
                self._kill_proc()
                return _LEAN_UNAVAILABLE
            try:
                resp = json.loads(line)
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning("Lean returned non-JSON %r: %s", line, e)
                return _LEAN_UNAVAILABLE
            if not resp.get("ok", False):
                logger.warning("Lean reported error: %s", resp.get("error"))
                return _LEAN_UNAVAILABLE
            verdict = resp.get("verdict")
            self._query_count += 1
            if verdict is not None:
                self._deny_count += 1
            if self._query_count % 25 == 0:
                logger.warning("LeanRunner: %d queries, %d denials so far",
                               self._query_count, self._deny_count)
            return None if verdict is None else str(verdict)

    def _readline_with_timeout(self, timeout_s: float) -> Optional[str]:
        result_box: list = []
        def _reader():
            try:
                line = self._proc.stdout.readline()  
                result_box.append(line)
            except Exception as e: 
                result_box.append(e)
        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout_s)
        if t.is_alive() or not result_box:
            return None
        first = result_box[0]
        return None if isinstance(first, Exception) else first

    def _kill_proc(self) -> None:
        if self._proc is None:
            return
        try: self._proc.kill()
        except Exception: pass
        try: self._proc.wait(timeout=1.0)
        except Exception: pass
        self._proc = None

    def shutdown(self) -> None:
        with self._lock:
            if self._proc is None:
                return
            logger.warning("LeanRunner shutdown: %d total queries, %d denials",
                           self._query_count, self._deny_count)
            try:
                self._proc.stdin.write(json.dumps({"shutdown": True}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=1.0)
            except Exception:
                pass
            finally:
                self._kill_proc()


def _json_default(o: Any) -> Any:
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if is_dataclass(o):
        return asdict(o)
    if hasattr(o, "model_dump"):
        try: return o.model_dump()
        except Exception: pass
    if hasattr(o, "dict"):
        try: return o.dict()
        except Exception: pass
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


# LEAN_RULE_SPECS — proven DB-driven rules
# Each entry: {"tool": <name>, "rule": <lean_rule_id>}.

LEAN_RULE_SPECS: list[dict] = [
    # D.1 Customer lookup by name requires DOB.
    {"tool": "get_customer_by_name",   "rule": "check_nameLookupHasDOB"},

    # D.2 Customer must be identified before any state-mutating call.
    {"tool": "suspend_line",           "rule": "check_customerIdentified"},
    {"tool": "resume_line",            "rule": "check_customerIdentified"},
    {"tool": "send_payment_request",   "rule": "check_customerIdentified"},
    {"tool": "refuel_data",            "rule": "check_customerIdentified"},
    {"tool": "enable_roaming",         "rule": "check_customerIdentified"},
    {"tool": "disable_roaming",        "rule": "check_customerIdentified"},
    {"tool": "get_data_usage",         "rule": "check_customerIdentified"},
    {"tool": "get_bills_for_customer", "rule": "check_customerIdentified"},

    # D.3–D.5 send_payment_request.
    {"tool": "send_payment_request",   "rule": "check_billOverdue"},
    {"tool": "send_payment_request",   "rule": "check_noOtherAwaitingPayment"},
    {"tool": "send_payment_request",   "rule": "check_billBelongsToCustomer"},

    # D.7–D.8 resume_line.
    {"tool": "resume_line",            "rule": "check_noOverdueBillsForCustomer"},
    {"tool": "resume_line",            "rule": "check_contractNotExpired"},

    # D.10–D.11 refuel_data.
    {"tool": "refuel_data",            "rule": "check_refuelPositive"},
    {"tool": "refuel_data",            "rule": "check_refuelMaxGB"},
]


# LEAN_POST_RULE_SPECS — POST checks proved in PolicyChecker.lean
# Lean POST rules consume the raw tool result string + an extended
# AgentState (with `user_phone`).

LEAN_POST_RULE_SPECS: list[dict] = [
    # D.POST.1 get_data_usage → used ≥ limit + refuel  (Nat arithmetic in Lean)
    {"tool": "get_data_usage",   "rule": "check_result_dataUsage_exceeded"},

    # # D.POST.3 get_details_by_id (Line) → phone matches state.userPhone
    {"tool": "get_details_by_id", "rule": "check_result_linePhoneMatchesState"},

    # # D.POST.4 check_app_permissions (messaging) → storage AND sms granted
    {"tool": "check_app_permissions", "rule": "check_result_messagingPerms"},
]


# DB snapshot

def _coerce(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.toordinal()
    if isinstance(value, dict):
        return {str(k): _coerce(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_coerce(v) for v in value]
    if is_dataclass(value):
        return _coerce(asdict(value))
    if hasattr(value, "model_dump"):
        try: return _coerce(value.model_dump())
        except Exception: pass
    if hasattr(value, "dict"):
        try: return _coerce(value.dict())
        except Exception: pass
    if hasattr(value, "__dict__"):
        return _coerce({k: v for k, v in vars(value).items()
                        if not k.startswith("_")})
    return repr(value)


def _today_int() -> int:
    # Per policy header ("The current time is 2025-02-25 12:08:00 EST"),
    # all date-aware Lean rules anchor to this fixed reference date.
    return date(2025, 2, 25).toordinal()


def _empty_snapshot() -> dict:
    return {
        "today": _today_int(), "identified_customer": None,
        "customers": [], "lines": [], "bills": [],
    }


def _snapshot_db(db) -> dict:
    """Return a JSON-safe dict of TelecomDB matching Lean's expected schema."""
    customers = _coerce(getattr(db, "customers", []) or [])
    for c in customers:
        if isinstance(c, dict) and "status" not in c and "account_status" in c:
            c["status"] = c["account_status"]

    lines = _coerce(getattr(db, "lines", []) or [])
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        for fld in ("data_used_gb", "data_refueling_gb"):
            if fld in ln:
                try:
                    v = float(ln[fld] or 0)
                except (TypeError, ValueError):
                    v = 0.0
                ln[fld] = max(0, int(round(v)))
        if "owner_id" not in ln:
            ln["owner_id"] = ""

    return {
        "today": _today_int(),
        "identified_customer": getattr(db, "identified", None),
        "customers": customers,
        "lines": lines,
        "bills": _coerce(getattr(db, "bills", []) or []),
        "user_phone": "",
    }


# Identified-customer recovery

# The runtime DB has no `identified_customer` field; we recover it by
# scanning the conversation for the most recent successful identifying lookup.


def _msg_role(msg: Any) -> Optional[str]:
    return msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)


def _msg_content(msg: Any) -> Any:
    return msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)


def _msg_tool_calls(msg: Any) -> list:
    if isinstance(msg, dict):
        return msg.get("tool_calls") or []
    return getattr(msg, "tool_calls", None) or []


def _tc_name(tc: Any) -> str:
    if isinstance(tc, dict):
        return tc.get("name") or tc.get("function", {}).get("name", "") or ""
    return getattr(tc, "name", "") or ""


_IDENTIFYING_TOOLS = frozenset({
    "get_customer_by_id", "get_customer_by_phone", "get_customer_by_name",
})


def _infer_identified_customer(conversation: list) -> Optional[str]:
    """Return the customer_id of the most recent successful identifying lookup."""
    if not conversation:
        return None
    last_cid: Optional[str] = None
    pending_call: bool = False
    for msg in conversation:
        role = _msg_role(msg)
        if role == "assistant":
            pending_call = False
            for tc in _msg_tool_calls(msg):
                if _tc_name(tc) in _IDENTIFYING_TOOLS:
                    pending_call = True
                    break
            if not pending_call:
                content = _msg_content(msg)
                if isinstance(content, str) and "[Tool call:" in content:
                    for name in _IDENTIFYING_TOOLS:
                        if f"[Tool call: {name}(" in content:
                            pending_call = True
                            break
        elif role == "tool" and pending_call:
            pending_call = False
            content = _msg_content(msg) or ""
            if not isinstance(content, str):
                continue
            try:
                data = json.loads(content)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                cid = data.get("customer_id")
                if isinstance(cid, str) and cid:
                    last_cid = cid
        else:
            pending_call = False
    return last_cid


# Lean runner singleton

_LEAN_RUNNER: Optional[LeanRunner] = None
_LEAN_RUNNER_LOCK = threading.Lock()


def _get_runner() -> LeanRunner:
    global _LEAN_RUNNER
    if _LEAN_RUNNER is None:
        with _LEAN_RUNNER_LOCK:
            if _LEAN_RUNNER is None:
                _LEAN_RUNNER = LeanRunner()
    return _LEAN_RUNNER


# Per-rule telemetry

_STATS_LOCK = threading.Lock()
_STATS_T0 = time.monotonic()
_RULE_STATS: dict[tuple[str, str, str], dict] = {}


def _record_rule(tool: str, rule: str, phase: str, *,
                 fired: bool = False, error: bool = False,
                 unavailable: bool = False, feedback: Optional[str] = None) -> None:
    key = (tool or "<unknown>", rule or "<unknown>", phase or "<unknown>")
    with _STATS_LOCK:
        row = _RULE_STATS.get(key)
        if row is None:
            row = {"queries": 0, "fired": 0, "errors": 0,
                   "unavailable": 0, "last_feedback": None}
            _RULE_STATS[key] = row
        row["queries"] += 1
        if error: row["errors"] += 1
        if unavailable: row["unavailable"] += 1
        if fired:
            row["fired"] += 1
            if feedback:
                row["last_feedback"] = feedback[:500]


def get_rule_stats() -> list[dict]:
    with _STATS_LOCK:
        rows = [{"tool": t, "rule": r, "phase": p, **counts}
                for (t, r, p), counts in _RULE_STATS.items()]
    rows.sort(key=lambda r: (-r["fired"], -r["queries"], r["tool"], r["rule"]))
    return rows


def dump_stats(path: Optional[str] = None) -> Optional[str]:
    rows = get_rule_stats()
    if not rows:
        return None
    if path is None:
        out_dir = os.environ.get("TAU2_VERIFIER_STATS_DIR", "/tmp")
        try: os.makedirs(out_dir, exist_ok=True)
        except OSError: out_dir = "/tmp"
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        path = os.path.join(out_dir, f"verifier_stats_{os.getpid()}_{stamp}.json")
    payload = {"pid": os.getpid(),
               "wall_time_s": round(time.monotonic() - _STATS_T0, 3),
               "stats": rows}
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.warning("verifier rule stats written to %s (%d rules)",
                       path, len(rows))
        return path
    except OSError as e:
        logger.warning("failed to write verifier stats to %s: %s", path, e)
        return None


def _atexit_dump_stats() -> None:
    try: dump_stats()
    except Exception as e:  # pragma: no cover
        logger.warning("verifier stats dump failed: %s", e)


atexit.register(_atexit_dump_stats)


# Public API

def check_all(tool_name, tool_args, conversation, db, **kwargs):
    """Run Lean PRE rules then Python PRE rules.

    Short-circuits on first failure unless `kwargs['collect_all']` is set.
    """
    failures: list[str] = []
    collect_all = bool(kwargs.get("collect_all"))

    runner = _get_runner()
    db_snap = _snapshot_db(db) if db is not None else _empty_snapshot()
    identified = kwargs.get("identified_customer_id")
    if identified is None:
        identified = _infer_identified_customer(conversation)
    if identified is not None:
        db_snap["identified_customer"] = identified

    # Translate refuel_data's float `gb_amount` to Lean's Nat `gb_times_100`.
    if (tool_name == "refuel_data" and isinstance(tool_args, dict)
            and "gb_times_100" not in tool_args):
        try:
            gb = float(tool_args.get("gb_amount", 0) or 0)
            tool_args = {**tool_args,
                         "gb_times_100": max(0, int(round(gb * 100)))}
        except (TypeError, ValueError):
            pass

    for spec in LEAN_RULE_SPECS:
        if spec["tool"] != tool_name:
            continue
        try:
            req = {
                "id": uuid4().hex,
                "rule": spec["rule"],
                "tool_args": tool_args or {},
                "db_snapshot": db_snap,
            }
            verdict = runner.query(req)
            if verdict is _LEAN_UNAVAILABLE:
                _record_rule(tool_name, spec["rule"], "pre", unavailable=True)
                continue
            _record_rule(tool_name, spec["rule"], "pre",
                         fired=verdict is not None, feedback=verdict)
            if verdict is not None:
                if collect_all:
                    failures.append(verdict)
                else:
                    return verdict
        except Exception as e:
            logger.warning("Lean rule %s raised %s", spec["rule"], e)
            _record_rule(tool_name, spec["rule"], "pre", error=True)

    for rule in PYTHON_PRE_RULES:
        try:
            r = rule(tool_name, tool_args, conversation, db, **kwargs)
        except Exception as e:
            logger.warning("Python rule %s raised %s", rule.__name__, e)
            _record_rule(tool_name, rule.__name__, "pre_py", error=True)
            continue
        _record_rule(tool_name, rule.__name__, "pre_py",
                     fired=r is not None,
                     feedback=r if isinstance(r, str) else None)
        if r is not None:
            if collect_all:
                failures.append(r)
            else:
                return r

    return failures or None


def check_all_results(tool_name, tool_args, result_content, db=None, **kwargs):
    """Run Lean POST checks then any remaining Python POST checks.

    Returns a list of warning strings (possibly empty).
    """
    import re as _re
    warnings: list[str] = []

    # ---- Lean POST checks ------------------------------------------------
    runner = _get_runner()
    db_snap = _snapshot_db(db) if db is not None else _empty_snapshot()
    user_phone = kwargs.get("user_phone") or ""
    if user_phone:
        # Lean compares digits-only on both sides; normalise here so the
        # snapshot value matches whatever format `get_details_by_id` returns.
        db_snap["user_phone"] = _re.sub(r"\D", "", user_phone)
    identified = kwargs.get("identified_customer_id")
    if identified is not None:
        db_snap["identified_customer"] = identified

    post_args = {"result": result_content or ""}

    for spec in LEAN_POST_RULE_SPECS:
        if spec["tool"] != tool_name:
            continue
        try:
            req = {
                "id": uuid4().hex,
                "rule": spec["rule"],
                "tool_args": post_args,
                "db_snapshot": db_snap,
            }
            verdict = runner.query(req)
            if verdict is _LEAN_UNAVAILABLE:
                _record_rule(tool_name, spec["rule"], "post", unavailable=True)
                continue
            _record_rule(tool_name, spec["rule"], "post",
                         fired=verdict is not None, feedback=verdict)
            if verdict is not None:
                warnings.append(verdict)
        except Exception as e:
            logger.warning("Lean POST rule %s raised %s", spec["rule"], e)
            _record_rule(tool_name, spec["rule"], "post", error=True)

    # Python POST checks (if any)
    for chk in PYTHON_POST_CHECKS:
        try:
            w = chk(tool_name, tool_args, result_content, db=db, **kwargs)
        except Exception as e:
            logger.warning("Python post-check %s raised %s", chk.__name__, e)
            _record_rule(tool_name, chk.__name__, "post_py", error=True)
            continue
        _record_rule(tool_name, chk.__name__, "post_py",
                     fired=w is not None,
                     feedback=w if isinstance(w, str) else None)
        if w is not None:
            warnings.append(w)
    return warnings


__all__ = [
    "LeanRunner",
    "LEAN_RULE_SPECS",
    "LEAN_POST_RULE_SPECS",
    "check_all",
    "check_all_results",
    "get_rule_stats",
    "dump_stats",
]
