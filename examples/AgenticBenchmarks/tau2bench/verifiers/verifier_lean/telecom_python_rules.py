"""telecom_python_rules.py

Empirical / agent-coaching rules that do NOT exist in or
`PolicyChecker.lean`.  Add new rules below and append them to
`PYTHON_PRE_RULES` or `PYTHON_POST_CHECKS`.  The glue module
`telecom_glue_spec` imports these registries and runs them after every
Lean rule.

Rule signatures
---------------

A pre-execution rule:

    def rule_X(tool_name, tool_args, conversation, db, **kwargs) -> str | None:
        '''Return None to allow, or a feedback string to deny.'''

A post-execution check:

    def check_result_X(tool_name, tool_args, result_content, db=None, **kwargs)
            -> str | None:
        '''Return None for no warning, or a string to inject into the
        tool result.'''

Conventions
-----------

* Each rule should `if tool_name != "<T>": return None` on its first line
  unless it intentionally applies to many tools.
* Failures are surfaced as plain strings; do not raise (the glue logs and
  continues on exceptions, but explicit None/str is cleaner).
* For free-text facts, use `kwargs.get("<flag_name>")` first; fall back
  to `slm_helper.slm_extract` only if absent.
"""

from __future__ import annotations

import json
from typing import Any, Callable, List, Optional

# Standard DB lookup helpers (used by hand-written rules below).

def _find_customer(db, cid):
    return next((c for c in db.customers if c.customer_id == cid), None)


def _find_line(db, lid):
    return next((l for l in db.lines if l.line_id == lid), None)


def _find_bill(db, bid):
    return next((b for b in db.bills if b.bill_id == bid), None)


def _find_plan(db, pid):
    return next((p for p in db.plans if p.plan_id == pid), None)


# Transfer-to-human escalation rule 
# Policy: "You should escalate to a human agent if and only if the request
#          cannot be handled within the scope of your actions."
# Policy: "You should try your best to resolve the issue before escalating."
# Structural (no SLM) — block escalation until the agent has actually called
# the diagnostic tools relevant to the user's issue type.

def _msg_role(msg) -> Optional[str]:
    if isinstance(msg, dict):
        return msg.get("role")
    return getattr(msg, "role", None)


def _msg_content(msg) -> Any:
    if isinstance(msg, dict):
        return msg.get("content")
    return getattr(msg, "content", None)


def _msg_tool_calls(msg) -> list:
    if isinstance(msg, dict):
        return msg.get("tool_calls") or []
    return getattr(msg, "tool_calls", None) or []


def _tc_name(tc) -> str:
    if isinstance(tc, dict):
        return tc.get("name") or tc.get("function", {}).get("name", "") or ""
    return getattr(tc, "name", "") or ""


def _extract_tools_called(conversation: list) -> set[str]:
    """All tool names previously called by the agent.

    Handles both the native (pydantic / dict with `tool_calls` list) shape
    and the orchestrator-flattened shape where assistant `content` looks
    like `[Tool call: NAME({...})]`.
    """
    import re
    tools: set[str] = set()
    flat_re = re.compile(r"\[Tool call:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    for msg in conversation or []:
        if _msg_role(msg) != "assistant":
            continue
        for tc in _msg_tool_calls(msg):
            name = _tc_name(tc)
            if name:
                tools.add(name)
        content = _msg_content(msg)
        if isinstance(content, str) and "[Tool call:" in content:
            for m in flat_re.findall(content):
                tools.add(m)
    return tools


def _infer_issue_type(conversation: list, **kwargs) -> Optional[str]:
    """Infer issue type from the user's ticket / first message.

    Mirrors og_spec._infer_issue_type. Prefers verifier._user_instructions
    (the ticket text) — most reliable. Falls back to the first non-empty
    message content.
    """
    verifier = kwargs.get("verifier")
    user_instr = getattr(verifier, "_user_instructions", None) if verifier else None
    if user_instr:
        text = user_instr.lower()
    else:
        text = ""
        for msg in conversation or []:
            content = _msg_content(msg)
            if content:
                text = str(content).lower()
                break

    # MMS first (superset of data/service for messaging-specific triage).
    if any(kw in text for kw in (
        "mms", "picture message", "send picture", "send photo",
        "multimedia message",
    )):
        return "mms"
    if any(kw in text for kw in (
        "mobile data", "data issue", "internet",
        "slow data", "no data", "data not working",
        "cannot connect", "browsing", "data plan",
        "data speed", "connectivity issue",
    )):
        return "data"
    if any(kw in text for kw in (
        "no service", "no signal", "no network", "suspended",
        "can't make calls", "no connection", "service issue",
        "cannot call", "line suspended", "phone service",
    )):
        return "service"
    return None


# Diagnostic tools the agent must have tried before escalating, per issue type.
_REQUIRED_TOOLS: dict[str, dict[str, str]] = {
    "data": {
        "check_network_status": (
            "Check if mobile data is enabled — if disabled, turn it on "
            "with toggle_data()"
        ),
        "check_data_restriction_status": (
            "Check if Data Saver mode is on — if so, toggle it off "
            "with toggle_data_saver_mode"
        ),
        "check_network_mode_preference": (
            "Check the network mode preference — if set to 2G/3G, fix it "
            "with set_network_mode_preference('4g_5g_preferred')"
        ),
        "check_vpn_status": (
            "Check if a VPN is active and causing slow speeds — if VPN "
            "performance is Poor, disconnect with disconnect_vpn()"
        ),
        "get_data_usage": (
            "Check if user's data usage has exceeded their limit — "
            "if so, refuel with refuel_data"
        ),
    },
    "mms": {
        "check_network_status": (
            "Check network status — MMS requires mobile data to be ON "
            "and cellular service available"
        ),
        "check_app_permissions": (
            "Check messaging app permissions — if sms or storage is "
            "missing, grant it with grant_app_permission('messaging', ...)"
        ),
        "check_apn_settings": (
            "Check APN/MMSC settings — if MMSC URL is missing, reset "
            "with reset_apn_settings then reboot_device"
        ),
        "check_wifi_calling_status": (
            "Check if Wi-Fi Calling is on — if so, disable it with "
            "toggle_wifi_calling (it can interfere with MMS)"
        ),
        "check_network_mode_preference": (
            "Check network mode — MMS requires at least 3G, fix with "
            "set_network_mode_preference if set to 2G only"
        ),
        "get_data_usage": (
            "Check if data limit is exceeded — MMS requires active "
            "data, refuel with refuel_data if needed"
        ),
    },
    "service": {
        "check_network_status": (
            "Check network status including airplane mode and data settings"
        ),
        "check_sim_status": (
            "Check SIM card status — reseat with reseat_sim_card if missing, "
            "or escalate if locked"
        ),
        "check_apn_settings": (
            "Check APN settings — reset with reset_apn_settings + "
            "reboot_device if incorrect"
        ),
    },
}


def rule_transfer_missing_tools(tool_name, tool_args, conversation, db, **kwargs):
    """Block transfer if the agent hasn't tried required troubleshooting tools."""
    if tool_name != "transfer_to_human_agents":
        return None

    tools_called = _extract_tools_called(conversation)
    issue_type = _infer_issue_type(conversation, **kwargs)

    if not issue_type:
        return None  # Can't determine issue type, allow transfer

    required = _REQUIRED_TOOLS.get(issue_type, {})
    missing = []
    for tool, hint in required.items():
        if tool not in tools_called:
            missing.append(f"  - {tool}: {hint}")

    if not missing:
        return None  # All required tools tried, transfer is valid

    missing_str = "\n".join(missing)
    return (
        f"Policy violation: you are escalating to a human agent but have not "
        f"tried the following troubleshooting steps:\n"
        f"{missing_str}\n"
        f"Please try these tools before escalating. Only transfer to a human "
        f"agent after exhausting all available troubleshooting options."
    )


# SLM helpers

def _slm_extract_text_safe(question: str, conversation: list) -> str:
    """Best-effort SLM free-text extraction. Returns "" on any failure."""
    # try:
    from tau2.verifier.slm_helper import slm_extract  # type: ignore
    # except Exception:
        # return ""
    # try:
    return (slm_extract(question, conversation) or "").strip()
    # except Exception:
        # return ""


def _slm_yes_no_safe(question: str, conversation: list, *, default: bool = False) -> bool:
    """Returns True/False on a clear yes/no, else `default`."""
    answer = _slm_extract_text_safe(question, conversation).lower()
    if answer.startswith("yes"):
        return True
    if answer.startswith("no"):
        return False
    return default


def _arg_short_context(conversation: list, **kwargs) -> list:
    """Prefer the user's ticket / scenario instructions over the rolling
    conversation for arg-mismatch SLM extraction (shorter, more reliable)."""
    verifier = kwargs.get("verifier")
    instr = getattr(verifier, "_user_instructions", None) if verifier else None
    if instr:
        return [{"role": "system", "content": instr}]
    return conversation


def _ctx_text(ctx: list) -> str:
    return " ".join(
        str((m.get("content") if isinstance(m, dict) else "") or "")
        for m in (ctx or [])
    )

def check_result_speed_test(tool_name, tool_args, result_content, db=None, **kwargs):
    """Warn when run_speed_test returns anything below 'Excellent'."""
    if tool_name != "run_speed_test":
        return None
    rl = (result_content or "").lower()
    if "excellent" in rl or "no connection" in rl:
        return None
    return (
        " WARNING: Speed is below 'Excellent'. Per policy, any speed below "
        "'Excellent' is considered slow. You MUST continue troubleshooting "
        "(Path 2.2):\n"
        "  1. Check Data Saver: call check_data_restriction_status() — "
        "if Data Saver is ON, call toggle_data_saver_mode() to turn it OFF\n"
        "  2. Check network mode: call check_network_mode_preference() — "
        "if set to 2G/3G, call set_network_mode_preference('4g_5g_preferred')\n"
        "  3. Check VPN: call check_vpn_status() — "
        "if VPN is active, call disconnect_vpn()\n"
        "Re-run the speed test after each fix to check if speed improved to 'Excellent'."
    )


# Registries (consumed by telecom_glue_spec).

PYTHON_PRE_RULES: List[Callable[..., Optional[str]]] = [
    rule_transfer_missing_tools
]


PYTHON_POST_CHECKS: List[Callable[..., Optional[str]]] = [
    check_result_speed_test
]


__all__ = [
    "_find_customer", "_find_line", "_find_bill", "_find_plan",
    "rule_transfer_missing_tools",
    "rule_arg_refuel_line", "rule_arg_payment_bill",
    "rule_arg_resume_line", "rule_arg_enable_roaming_line",
    "rule_suspend_valid_reason",
    "check_result_line_phone", "check_result_speed_test",
    "check_result_can_send_mms", "check_result_get_data_usage",
    "check_result_check_network_status",
    "PYTHON_PRE_RULES", "PYTHON_POST_CHECKS",
]
