"""

Each rule is a function that takes:
  - tool_name: str           (the tool being called)
  - tool_args: dict          (the arguments passed to the tool)
  - conversation: list[dict] (recent message history for SLM extraction)
  - db: TelecomDB            (current database state for lookups)

And returns:
  - None            if the call is ALLOWED (rule passes or doesn't apply)
  - str             a feedback message explaining the violation
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Simulated date from telecom/utils.py
CURRENT_DATE = datetime.date(2025, 2, 25)


# ============================================================================
#  Helpers
# ============================================================================

def _find_customer(db, customer_id: str):
    """Find customer by ID from the list-based TelecomDB."""
    for c in db.customers:
        if c.customer_id == customer_id:
            return c
    return None


def _find_line(db, line_id: str):
    """Find line by ID."""
    for line in db.lines:
        if line.line_id == line_id:
            return line
    return None


def _find_bill(db, bill_id: str):
    """Find bill by ID."""
    for bill in db.bills:
        if bill.bill_id == bill_id:
            return bill
    return None


def _find_plan(db, plan_id: str):
    """Find plan by ID."""
    for plan in db.plans:
        if plan.plan_id == plan_id:
            return plan
    return None


def _get_customer_bills(db, customer_id: str):
    """Get all bills for a customer."""
    customer = _find_customer(db, customer_id)
    if not customer:
        return []
    return [_find_bill(db, bid) for bid in customer.bill_ids if _find_bill(db, bid)]


def _has_overdue_bills(db, customer_id: str) -> bool:
    """Check if customer has any overdue bills."""
    for bill in _get_customer_bills(db, customer_id):
        if bill and bill.status.value == "Overdue":
            return True
    return False


def _has_awaiting_payment_bill(db, customer_id: str) -> bool:
    """Check if customer already has a bill in AWAITING PAYMENT status."""
    for bill in _get_customer_bills(db, customer_id):
        if bill and bill.status.value == "Awaiting Payment":
            return True
    return False


# ============================================================================
#  REFUEL DATA rules
#  Policy: "The maximum amount of data that can be refueled is 2GB."
#  Policy: Line must be active to refuel.
#  Policy: "Know how much data they want to refuel" / "Confirm the price"
# ============================================================================

def rule_refuel_max_2gb(tool_name, tool_args, conversation, db):
    """Data refueling is capped at 2 GB per request."""
    if tool_name != "refuel_data":
        return None
    gb_amount = tool_args.get("gb_amount", 0)
    try:
        gb_amount = float(gb_amount)
    except (ValueError, TypeError):
        return None
    if gb_amount > 2.0:
        return (
            f"Policy violation: data refueling is limited to 2 GB per request, "
            f"but {gb_amount} GB was requested."
        )
    return None


def rule_refuel_line_active(tool_name, tool_args, conversation, db):
    """Data refueling requires an active line (tools.py has the check commented out)."""
    if tool_name != "refuel_data":
        return None
    line_id = tool_args.get("line_id", "")
    line = _find_line(db, line_id)
    if not line:
        return None
    if line.status.value != "Active":
        return (
            f"Policy violation: cannot refuel data on line {line_id} — "
            f"status is '{line.status.value}', must be 'Active'."
        )
    return None


# ============================================================================
#  SEND PAYMENT REQUEST rules
#  Policy: "Check the bill status to make sure it is overdue."
#  Policy: "A user can only have one bill in the AWAITING PAYMENT status at a time."
# ============================================================================

def rule_payment_bill_must_be_overdue(tool_name, tool_args, conversation, db):
    """Payment request should only be sent for overdue bills."""
    if tool_name != "send_payment_request":
        return None
    bill_id = tool_args.get("bill_id", "")
    bill = _find_bill(db, bill_id)
    if not bill:
        return None
    if bill.status.value != "Overdue":
        return (
            f"Policy violation: payment request should only be sent for overdue "
            f"bills, but bill {bill_id} has status '{bill.status.value}'."
        )
    return None


def rule_payment_no_duplicate_awaiting(tool_name, tool_args, conversation, db):
    """A user can only have one bill in AWAITING PAYMENT status at a time."""
    if tool_name != "send_payment_request":
        return None
    customer_id = tool_args.get("customer_id", "")
    if not customer_id:
        return None
    if _has_awaiting_payment_bill(db, customer_id):
        return (
            f"Policy violation: customer {customer_id} already has a bill in "
            f"'Awaiting Payment' status. Only one bill can be awaiting payment "
            f"at a time."
        )
    return None


# ============================================================================
#  RESUME LINE rules
#  Policy: "You are not allowed to lift the suspension if the line's contract
#           end date is in the past."
#  Policy: "You are allowed to lift the suspension after the user has paid
#           all their overdue bills."
# ============================================================================

def rule_resume_contract_not_expired(tool_name, tool_args, conversation, db):
    """Cannot resume a line if the contract has expired."""
    if tool_name != "resume_line":
        return None
    line_id = tool_args.get("line_id", "")
    line = _find_line(db, line_id)
    if not line:
        return None
    if line.contract_end_date and line.contract_end_date < CURRENT_DATE:
        return (
            f"Policy violation: cannot resume line {line_id} — "
            f"contract expired on {line.contract_end_date} "
            f"(current date is {CURRENT_DATE})."
        )
    return None


def rule_resume_all_bills_paid(tool_name, tool_args, conversation, db):
    """Cannot resume a line until ALL overdue bills are paid."""
    if tool_name != "resume_line":
        return None
    customer_id = tool_args.get("customer_id", "")
    if _has_overdue_bills(db, customer_id):
        return (
            f"Policy violation: cannot resume line — customer {customer_id} "
            f"still has overdue bills. All overdue bills must be paid first."
        )
    return None


# ============================================================================
#  SUSPEND LINE rules
#  Policy: "A line can be suspended for the following reasons:
#           - The user has an overdue bill.
#           - The line's contract end date is in the past."
# ============================================================================

def rule_suspend_valid_reason(tool_name, tool_args, conversation, db):
    """Suspension reason must be for overdue bill or expired contract."""
    if tool_name != "suspend_line":
        return None
    from tau2.verifier.slm_helper import slm_extract
    reason = tool_args.get("reason", "")
    answer = slm_extract(
        "Is this suspension reason related to either an overdue bill / "
        "non-payment, or an expired contract? "
        "Answer 'yes' or 'no'.",
        [{"role": "system", "content": f"Suspension reason: {reason}"}],
    )
    if answer.strip().lower().startswith("no"):
        return (
            f"Policy violation: line suspension reason '{reason}' is not a valid "
            f"policy reason. Lines can only be suspended for overdue bills or "
            f"expired contracts."
        )
    return None


# ============================================================================
#  ENABLE ROAMING rules
#  Policy: "If a user is traveling outside their home network, you should
#           check if the line is roaming enabled. If it is not, you should
#           enable it at no cost for the user."
#  (No rule needed to block enable_roaming — it's always free and allowed.
#   But disable_roaming should not happen during tech support for a traveling user.)
# ============================================================================

def rule_disable_roaming_not_while_traveling(tool_name, tool_args, conversation, db):
    """Should not disable roaming if the user is traveling and needs data."""
    if tool_name != "disable_roaming":
        return None
    from tau2.verifier.slm_helper import slm_extract
    answer = slm_extract(
        "Is the user currently traveling outside their home network or abroad "
        "and experiencing data connectivity issues? "
        "Answer 'yes' or 'no'.",
        conversation,
    )
    if answer.strip().lower().startswith("yes"):
        return (
            "Policy violation: should not disable roaming when the user is "
            "traveling outside their home network. Roaming should be enabled "
            "at no cost for traveling users."
        )
    return None


# ============================================================================
#  TRANSFER TO HUMAN AGENT rules
#  Policy: "You should escalate to a human agent if and only if the request
#           cannot be handled within the scope of your actions."
#  Policy: "You should try your best to resolve the issue before escalating."
# ============================================================================

def _extract_tools_called(conversation: list[dict]) -> set[str]:
    """Extract all tool names already called from conversation history."""
    tools = set()
    for msg in conversation:
        # Solo mode: tool calls are in assistant messages
        if msg.get("role") == "assistant":
            for tc in (msg.get("tool_calls") or []):
                name = tc.get("name", tc.get("function", {}).get("name", ""))
                if name:
                    tools.add(name)
    return tools


def _infer_issue_type(conversation: list[dict], **kwargs) -> str | None:
    """Infer the issue type from the user's initial ticket/instructions."""
    # Use user_instructions (ticket) if available — most reliable source.
    verifier = kwargs.get("verifier")
    user_instr = getattr(verifier, "_user_instructions", None) if verifier else None
    if user_instr:
        text = user_instr.lower()
    else:
        # Fallback: use first message (in solo mode this is the assistant's
        # initial thinking which quotes the ticket).
        text = ""
        for msg in conversation:
            if msg.get("content"):
                text = str(msg["content"]).lower()
                break

    # Check for MMS keywords first (MMS is a superset of data/service)
    if any(kw in text for kw in (
        "mms", "picture message", "send picture", "send photo",
        "multimedia message",
    )):
        return "mms"
    # Check for data issue keywords
    if any(kw in text for kw in (
        "mobile data", "data issue", "internet",
        "slow data", "no data", "data not working",
        "cannot connect", "browsing", "data plan",
        "data speed", "connectivity issue",
    )):
        return "data"
    # Check for service keywords
    if any(kw in text for kw in (
        "no service", "no signal", "no network", "suspended",
        "can't make calls", "no connection", "service issue",
        "cannot call", "line suspended", "phone service",
    )):
        return "service"
    return None


# Required troubleshooting tools per issue type.
# We require the DIAGNOSTIC checks (not the fix tools), since the agent
# must at least check each category before deciding it's not relevant.
# If a diagnostic reveals a problem, the agent should fix it.
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


# ============================================================================
#  SLM-based argument validation rules
# ============================================================================

def rule_arg_refuel_line(tool_name, tool_args, conversation, db):
    """Verify the refuel is being applied to the line the user discussed."""
    if tool_name != "refuel_data":
        return None
    from tau2.verifier.slm_helper import slm_extract
    line_id = tool_args.get("line_id", "")

    # Find the phone number for this line to check against conversation
    line = _find_line(db, line_id)
    if not line:
        return None

    answer = slm_extract(
        "What phone number or line does the user want to add data to? "
        "Reply with ONLY the phone number or line ID.",
        conversation,
    )
    raw_answer = answer.strip()
    mentioned = raw_answer.replace("-", "").replace(" ", "").lower()
    line_phone = line.phone_number.replace("-", "").replace(" ", "").lower()
    line_id_lower = line_id.lower()

    # Match either line_id or phone number anywhere in the SLM answer
    if (line_id_lower in mentioned or
            line_phone in mentioned or
            mentioned in line_phone or
            line_id_lower in raw_answer.lower()):
        return None

    # Fallback: check if the line_id appears in the conversation itself
    convo_text = " ".join(
        str(m.get("content", "")) for m in conversation
    ).lower()
    if line_id_lower in convo_text:
        return None

    return (
        f"Argument mismatch: refueling line {line_id} ({line.phone_number}) "
        f"but the user mentioned: {raw_answer}"
    )


def rule_arg_payment_bill(tool_name, tool_args, conversation, db):
    """Verify the payment request targets the bill the user discussed."""
    if tool_name != "send_payment_request":
        return None
    from tau2.verifier.slm_helper import slm_extract
    bill_id = tool_args.get("bill_id", "")
    answer = slm_extract(
        "What bill ID does the user want to pay? "
        "Reply with ONLY the bill ID (e.g. B1234321). "
        "Remove any hyphens or dashes from the ID.",
        conversation,
    )
    # Normalize both: strip hyphens, dashes, spaces for comparison
    mentioned = answer.strip().upper().replace(" ", "").replace("-", "")
    target = bill_id.upper().replace("-", "")
    if target and mentioned and target not in mentioned and mentioned not in target:
        # Fallback: check if the bill_id appears in the conversation itself
        convo_text = " ".join(
            str(m.get("content", "")) for m in conversation
        ).upper()
        if target in convo_text:
            return None
        return (
            f"Argument mismatch: sending payment for bill {bill_id} "
            f"but the user mentioned: {answer}"
        )
    return None


def rule_arg_resume_line(tool_name, tool_args, conversation, db):
    """Verify resume_line targets the correct line and customer."""
    if tool_name != "resume_line":
        return None
    from tau2.verifier.slm_helper import slm_extract
    line_id = tool_args.get("line_id", "")
    customer_id = tool_args.get("customer_id", "")

    line = _find_line(db, line_id)
    if not line:
        return None

    # Check the customer owns this line
    customer = _find_customer(db, customer_id)
    if customer and line_id not in customer.line_ids:
        return (
            f"Argument mismatch: line {line_id} does not belong to "
            f"customer {customer_id}."
        )

    answer = slm_extract(
        "What phone number or line does the user want to resume/unsuspend? "
        "Reply with ONLY the phone number or line ID.",
        conversation,
    )
    raw_answer = answer.strip()
    mentioned = raw_answer.replace("-", "").replace(" ", "").lower()
    line_phone = line.phone_number.replace("-", "").replace(" ", "").lower()
    line_id_lower = line_id.lower()

    if (line_id_lower in mentioned or
            line_phone in mentioned or
            mentioned in line_phone or
            line_id_lower in raw_answer.lower()):
        return None

    convo_text = " ".join(
        str(m.get("content", "")) for m in conversation
    ).lower()
    if line_id_lower in convo_text or line_phone in convo_text:
        return None

    return (
        f"Argument mismatch: resuming line {line_id} ({line.phone_number}) "
        f"but the user mentioned: {raw_answer}"
    )


def rule_arg_enable_roaming_line(tool_name, tool_args, conversation, db):
    """Verify enable_roaming targets the correct line."""
    if tool_name != "enable_roaming":
        return None
    from tau2.verifier.slm_helper import slm_extract
    line_id = tool_args.get("line_id", "")
    customer_id = tool_args.get("customer_id", "")

    line = _find_line(db, line_id)
    if not line:
        return None

    # Check the customer owns this line
    customer = _find_customer(db, customer_id)
    if customer and line_id not in customer.line_ids:
        return (
            f"Argument mismatch: line {line_id} does not belong to "
            f"customer {customer_id}."
        )

    answer = slm_extract(
        "What phone number or line is the user calling about? "
        "Reply with ONLY the phone number or line ID.",
        conversation,
    )
    raw_answer = answer.strip()
    mentioned = raw_answer.replace("-", "").replace(" ", "").lower()
    line_phone = line.phone_number.replace("-", "").replace(" ", "").lower()
    line_id_lower = line_id.lower()

    if (line_id_lower in mentioned or
            line_phone in mentioned or
            mentioned in line_phone or
            line_id_lower in raw_answer.lower()):
        return None

    convo_text = " ".join(
        str(m.get("content", "")) for m in conversation
    ).lower()
    if line_id_lower in convo_text or line_phone in convo_text:
        return None

    return (
        f"Argument mismatch: enabling roaming on line {line_id} ({line.phone_number}) "
        f"but the user mentioned: {raw_answer}"
    )


# ============================================================================
#  CUSTOMER LOOKUP rules
#  Policy: "For name lookup, date of birth is required for verification."
# ============================================================================

def rule_customer_lookup_name_requires_dob(tool_name, tool_args, conversation, db):
    """Name-based customer lookup must include date of birth."""
    if tool_name != "get_customer_by_name":
        return None
    dob = tool_args.get("dob", "")
    if not dob or not dob.strip():
        return (
            "Policy violation: looking up customer by name requires "
            "date of birth for verification purposes."
        )
    return None


# ============================================================================
#  Tech Support Workflow — Path 1: No Service
#  Policy (Step 1.4): "If the line is suspended ... follow the instructions
#          in the main policy for line suspension."
#  (resume_line rules already cover the main policy constraints.)
#
#  Policy (Step 1.2): "If SIM is LOCKED with PIN/PUK — Escalate to
#          technical support for assistance with SIM security."
#  (Transfer rule already prevents premature transfers; SIM lock is a valid
#   reason to escalate.)
# ============================================================================


# ============================================================================
#  Tech Support Workflow — Path 2: Data Issues
#  Policy (Step 2.1.4): "Check if user's data usage has exceeded their
#           data limit." If exceeded, refuel or change plan.
#  Policy: Refuel data max 2GB (already covered).
# ============================================================================

def rule_refuel_only_when_data_exceeded(tool_name, tool_args, conversation, db):
    """Data refueling should only be done when data usage exceeds the limit."""
    if tool_name != "refuel_data":
        return None
    line_id = tool_args.get("line_id", "")
    customer_id = tool_args.get("customer_id", "")

    line = _find_line(db, line_id)
    if not line:
        return None

    plan = _find_plan(db, line.plan_id)
    if not plan:
        return None

    total_available = plan.data_limit_gb + line.data_refueling_gb
    if line.data_used_gb <= total_available:
        # Data is not exceeded — refueling might still be requested by user
        # proactively, so only warn if usage is well under limit
        if line.data_used_gb < plan.data_limit_gb * 0.8:
            from tau2.verifier.slm_helper import slm_extract
            answer = slm_extract(
                "Did the user explicitly ask to add/refuel more data to their "
                "line, or is the agent doing it as part of troubleshooting a "
                "data connectivity issue? Answer 'user requested' or "
                "'troubleshooting'.",
                conversation,
            )
            if "troubleshooting" in answer.strip().lower():
                return (
                    f"Policy violation: data refueling line {line_id} but data "
                    f"usage ({line.data_used_gb} GB) is well below the limit "
                    f"({plan.data_limit_gb} GB). Data connectivity issues should "
                    f"be diagnosed through the troubleshooting workflow first."
                )
    return None


# ============================================================================
#  Rule registry & check_all
# ============================================================================

ALL_RULES = [
    # Refuel data
    rule_refuel_max_2gb,
    rule_refuel_line_active,
    rule_refuel_only_when_data_exceeded,
    rule_arg_refuel_line,
    # Payment
    rule_payment_bill_must_be_overdue,
    rule_payment_no_duplicate_awaiting,
    rule_arg_payment_bill,
    # Resume line
    rule_resume_contract_not_expired,
    rule_resume_all_bills_paid,
    rule_arg_resume_line,
    # Suspend line
    rule_suspend_valid_reason,
    # Roaming
    rule_disable_roaming_not_while_traveling,
    rule_arg_enable_roaming_line,
    # Customer lookup
    rule_customer_lookup_name_requires_dob,
    # Transfer
    rule_transfer_missing_tools,
]

CHEAP_RULES = [
    rule_refuel_max_2gb,
    rule_refuel_line_active,
    rule_payment_bill_must_be_overdue,
    rule_payment_no_duplicate_awaiting,
    rule_resume_contract_not_expired,
    rule_resume_all_bills_paid,
    rule_customer_lookup_name_requires_dob,
    rule_transfer_missing_tools,
]

SLM_RULES = [r for r in ALL_RULES if r not in CHEAP_RULES]

# Argument-accuracy rules: these only need the ticket/instructions context,
# not the full conversation. Passing a shorter context to the SLM yields
# more reliable extraction and is cheaper.
ARG_RULES = {
    rule_arg_refuel_line,
    rule_arg_payment_bill,
    rule_arg_resume_line,
    rule_arg_enable_roaming_line,
}

# Rules that need access to the verifier / kwargs (e.g., user_instructions).
_KWARGS_RULES = {
    rule_transfer_missing_tools,
}


def check_all(
    tool_name: str,
    tool_args: dict,
    conversation: list[dict],
    db,
    cheap_only: bool = False,
    **kwargs,
) -> str | None:
    """Run all applicable telecom policy rules against a tool call."""
    rules = CHEAP_RULES if cheap_only else ALL_RULES

    # Extract user instructions (ticket) from verifier for arg-accuracy rules.
    # This is much shorter than the full conversation and contains all the
    # key identifiers (phone number, customer name, etc.) upfront.
    verifier = kwargs.get("verifier")
    user_instructions = (
        getattr(verifier, "_user_instructions", None) if verifier else None
    )
    if user_instructions:
        short_context = [{"role": "system", "content": user_instructions}]
    else:
        short_context = conversation

    for rule_fn in rules:
        try:
            # Arg-accuracy rules use the short ticket context;
            # policy-constraint rules use the full conversation.
            ctx = short_context if rule_fn in ARG_RULES else conversation
            if rule_fn in _KWARGS_RULES:
                result = rule_fn(tool_name, tool_args, ctx, db, **kwargs)
            else:
                result = rule_fn(tool_name, tool_args, ctx, db)
            if result is not None:
                logger.info("Rule %s violated: %s", rule_fn.__name__, result)
                return result
        except Exception as e:
            logger.warning("Rule %s raised exception: %s", rule_fn.__name__, e)
            continue

    return None


#  POST-EXECUTION RESULT CHECKS

def check_result_line_phone(
    tool_name: str,
    tool_args: dict,
    result_content: str,
    user_phone: str | None,
) -> str | None:
    """After get_details_by_id returns a line, check if its phone matches the user's phone.

    If the agent looked up a line whose phone_number differs from the phone
    the user called with (captured from get_customer_by_phone), inject a
    warning so the agent knows to use a different line.
    """
    if tool_name != "get_details_by_id":
        return None
    if not user_phone:
        return None

    # Only applies to line lookups (line_id starts with L)
    lookup_id = tool_args.get("id", "")
    if not lookup_id.upper().startswith("L"):
        return None

    # Parse the result to extract the line's phone_number
    import json
    try:
        data = json.loads(result_content)
    except (json.JSONDecodeError, TypeError):
        return None

    line_phone = data.get("phone_number", "")
    if not line_phone:
        return None

    # Compare: if the line's phone matches the user's phone, no issue
    if line_phone.strip() == user_phone.strip():
        return None

    return (
        f"⚠️ WARNING: This line {lookup_id} has phone number {line_phone}, "
        f"which does NOT match the customer's contact phone {user_phone}. "
        f"This is likely NOT the correct line for the user's issue. "
        f"Look up the line whose phone number matches {user_phone} instead."
    )


def check_result_speed_test(
    tool_name: str,
    tool_args: dict,
    result_content: str,
) -> str | None:
    """After run_speed_test, warn the agent if speed is below 'Excellent'.

    Per policy: 'Any speed below Excellent is considered slow.'
    The agent must continue troubleshooting (Path 2.2): check Data Saver,
    network mode preference, and VPN before declaring the issue resolved.
    """
    if tool_name != "run_speed_test":
        return None

    result_lower = result_content.lower()

    # If speed is already excellent, no warning needed
    if "excellent" in result_lower:
        return None

    # If no connection, different problem — don't warn about speed
    if "no connection" in result_lower:
        return None

    return (
        "⚠️ WARNING: Speed is below 'Excellent'. Per policy, any speed below "
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


def check_result_can_send_mms(
    tool_name: str,
    tool_args: dict,
    result_content: str,
    last_tool_results: dict[str, str],
    called_tools: list[str],
) -> str | None:
    """After can_send_mms returns failure, analyze what the agent has checked
    so far and provide targeted feedback on what to fix.

    MMS requires ALL of:
      1. Mobile data working (data on + connected)
      2. Network >= 3G
      3. Wi-Fi calling OFF (carrier doesn't support MMS over Wi-Fi)
      4. MMSC URL configured in APN settings
      5. Messaging app has both 'sms' AND 'storage' permissions
    """
    if tool_name != "can_send_mms":
        return None

    # Only trigger on failure
    if "cannot" not in result_content.lower():
        return None

    hints = []

    # --- Check 1: App permissions ---
    perm_result = last_tool_results.get("check_app_permissions", "")
    if perm_result:
        perm_lower = perm_result.lower()
        missing_perms = []
        if "sms" not in perm_lower:
            missing_perms.append("sms")
        if "storage" not in perm_lower:
            missing_perms.append("storage")
        if missing_perms:
            hints.append(
                f"MISSING PERMISSIONS: The messaging app is missing "
                f"{', '.join(missing_perms)} permission(s). "
                f"Call grant_app_permission('messaging', '{missing_perms[0]}') to fix."
            )
    elif "check_app_permissions" not in called_tools:
        hints.append(
            "NOT CHECKED: You have not checked messaging app permissions yet. "
            "Call check_app_permissions('messaging') — MMS requires both 'sms' "
            "and 'storage' permissions."
        )

    # --- Check 2: Wi-Fi calling ---
    wifi_result = last_tool_results.get("check_wifi_calling_status", "")
    if wifi_result:
        if "on" in wifi_result.lower() and "off" not in wifi_result.lower():
            hints.append(
                "WI-FI CALLING IS ON: Wi-Fi Calling can interfere with MMS. "
                "Call toggle_wifi_calling() to turn it off."
            )
    elif "check_wifi_calling_status" not in called_tools:
        hints.append(
            "NOT CHECKED: You have not checked Wi-Fi calling status. "
            "Call check_wifi_calling_status() — if it's ON, it can block MMS."
        )

    # --- Check 3: APN/MMSC settings ---
    apn_result = last_tool_results.get("check_apn_settings", "")
    if apn_result:
        if "none" in apn_result.lower() and "mmsc" in apn_result.lower():
            hints.append(
                "MMSC URL MISSING: APN settings show no MMSC URL configured. "
                "Call reset_apn_settings() then reboot_device() to fix."
            )
    elif "check_apn_settings" not in called_tools:
        hints.append(
            "NOT CHECKED: You have not checked APN settings. "
            "Call check_apn_settings() — MMS requires a valid MMSC URL."
        )

    # --- Check 4: Network status / mobile data ---
    net_result = last_tool_results.get("check_network_status", "")
    if net_result:
        net_lower = net_result.lower()
        if "mobile data enabled: no" in net_lower:
            hints.append(
                "MOBILE DATA OFF: Mobile data is disabled. "
                "Call toggle_data() to enable it — MMS requires mobile data."
            )
    elif "check_network_status" not in called_tools:
        hints.append(
            "NOT CHECKED: You have not checked network status. "
            "Call check_network_status() — MMS requires mobile data to be ON."
        )

    # --- Check 5: Network mode (must be >= 3G for MMS) ---
    mode_result = last_tool_results.get("check_network_mode_preference", "")
    if mode_result:
        mode_lower = mode_result.lower()
        if "2g" in mode_lower and "3g" not in mode_lower and "4g" not in mode_lower and "5g" not in mode_lower:
            hints.append(
                "NETWORK MODE 2G: MMS requires at least 3G. "
                "Call set_network_mode_preference('4g_5g_preferred') to upgrade."
            )
    elif "check_network_mode_preference" not in called_tools:
        hints.append(
            "NOT CHECKED: You have not checked network mode preference. "
            "Call check_network_mode_preference() — MMS requires at least 3G."
        )

    # --- Check 6: Data exhaustion (blocks MMS even if everything else is fine) ---
    data_result = last_tool_results.get("get_data_usage", "")
    if data_result:
        import json as _json
        try:
            data_info = _json.loads(data_result)
            used = float(data_info.get("data_used_gb", 0))
            limit = float(data_info.get("data_limit_gb", 999))
            refueled = float(data_info.get("data_refueling_gb", 0))
            if used >= limit + refueled:
                hints.append(
                    "DATA EXHAUSTED: Data usage ({:.1f} GB) exceeds limit "
                    "({:.1f} GB + {:.1f} GB refueled). MMS requires data. "
                    "Call refuel_data(customer_id, line_id, gb_amount=2.0) to restore.".format(
                        used, limit, refueled
                    )
                )
        except (ValueError, TypeError, _json.JSONDecodeError):
            pass
    elif "get_data_usage" not in called_tools:
        hints.append(
            "NOT CHECKED: You have not checked data usage. "
            "Call get_data_usage(customer_id, line_id) — if data is exhausted, "
            "MMS will fail even if all other settings are correct."
        )

    # --- Check 7: Device-level roaming (user abroad needs toggle_roaming) ---
    net_result_roam = last_tool_results.get("check_network_status", "")
    if net_result_roam and "data roaming enabled: no" in net_result_roam.lower():
        hints.append(
            "DEVICE ROAMING OFF: Data Roaming is disabled on the device. "
            "If the user is abroad, call enable_roaming(customer_id, line_id) "
            "AND toggle_roaming() to enable roaming on both account and device."
        )

    if not hints:
        return None

    return (
        "⚠️ MMS CANNOT BE SENT. Based on your previous checks, here are "
        "the issues to fix:\n  " + "\n  ".join(hints)
    )


def check_result_get_data_usage(
    tool_name: str,
    tool_args: dict,
    result_content: str,
) -> str | None:
    """After get_data_usage, warn if data usage exceeds the plan limit.

    When data_used_gb >= data_limit_gb, the user's data is exhausted and
    connectivity is lost. The agent must refuel data to restore service.
    """
    if tool_name != "get_data_usage":
        return None

    import json
    try:
        data = json.loads(result_content)
    except (json.JSONDecodeError, TypeError):
        return None

    try:
        used = float(data.get("data_used_gb", 0))
        limit = float(data.get("data_limit_gb", 999))
        refueled = float(data.get("data_refueling_gb", 0))
    except (ValueError, TypeError):
        return None

    # If usage exceeds limit (even with refueling counted), data is exhausted
    if used >= limit + refueled:
        return (
            "⚠️ WARNING: Data usage ({:.1f} GB) has EXCEEDED the plan limit "
            "({:.1f} GB + {:.1f} GB refueled = {:.1f} GB available). "
            "The user's data connectivity is LOST. You MUST call "
            "refuel_data(customer_id, line_id, gb_amount=2.0) to restore "
            "data service. Maximum refuel is 2 GB per call.".format(
                used, limit, refueled, limit + refueled
            )
        )

    return None


def check_result_check_network_status(
    tool_name: str,
    tool_args: dict,
    result_content: str,
    called_tools: list[str],
) -> str | None:
    """After check_network_status, warn about device-level roaming if disabled.

    When the user is abroad and 'Data Roaming Enabled: No' appears, the agent
    must call toggle_roaming() on the device AND enable_roaming() on the account.
    """
    if tool_name != "check_network_status":
        return None

    result_lower = result_content.lower()

    hints = []

    # Check for device-level roaming disabled
    if "data roaming enabled: no" in result_lower:
        hints.append(
            "DATA ROAMING DISABLED ON DEVICE: The device has Data Roaming "
            "turned OFF. If the user is abroad/traveling, you MUST:\n"
            "    1. Call enable_roaming(customer_id, line_id) to enable roaming on the account\n"
            "    2. Call toggle_roaming() to enable roaming on the DEVICE\n"
            "  Both steps are required — account-level and device-level are separate controls."
        )

    if not hints:
        return None

    return "⚠️ WARNING:\n  " + "\n  ".join(hints)


def check_result_line_suspended(
    tool_name: str,
    tool_args: dict,
    result_content: str,
    called_tools: list[str],
) -> str | None:
    """After get_details_by_id returns a line with 'Suspended' status,
    remind the agent of the full service restoration workflow.

    After resuming a suspended line, the agent MUST also troubleshoot the
    device (check_network_status, check_sim_status, reboot_device, etc.)
    because physical/device issues may co-exist with the suspension.
    """
    if tool_name != "get_details_by_id":
        return None

    # Only applies to line lookups
    lookup_id = tool_args.get("id", "")
    if not lookup_id.upper().startswith("L"):
        return None

    import json
    try:
        data = json.loads(result_content)
    except (json.JSONDecodeError, TypeError):
        return None

    status = data.get("status", "")
    if status != "Suspended":
        return None

    return (
        "⚠️ WARNING: This line is SUSPENDED. To fully restore service you must:\n"
        "  1. Check for overdue bills → pay them → resume_line\n"
        "  2. AFTER resuming, the user must reboot their device (call reboot_device)\n"
        "  3. Then do FULL device troubleshooting: check_network_status, check_sim_status,\n"
        "     toggle_airplane_mode (if ON), reseat_sim_card (if SIM issues), reset_apn_settings + reboot\n"
        "  Do NOT stop after resume_line — the device may still have issues that need fixing."
    )


