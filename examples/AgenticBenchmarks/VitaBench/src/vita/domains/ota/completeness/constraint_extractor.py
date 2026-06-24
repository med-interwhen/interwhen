"""
LLM-based completeness constraint extractor for OTA tasks.

Extracts minimal constraints (counts, cities, routes, dates) needed to
verify that all required bookings were made. Attribute-level details
(room type, seat class, etc.) are left to the soundness judge.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from vita.config import DEFAULT_LLM_EVALUATOR, models
from vita.data_model.message import SystemMessage, UserMessage
from vita.data_model.tasks import Task
from vita.domains.ota.completeness.schema import CompletenessConstraints
from vita.domains.ota.verifier.utils import _extract_json
from vita.prompts import get_prompts
from vita.utils.llm_utils import generate

logger = logging.getLogger(__name__)


def _format_available_cities(environment: dict) -> str:
    """Extract unique flight/train city names from the environment."""
    cities: set[str] = set()
    for flight in environment.get("flights", {}).values():
        f = flight if isinstance(flight, dict) else vars(flight)
        for key in ("departure_city", "arrival_city"):
            if v := f.get(key):
                cities.add(v)
    for train in environment.get("trains", {}).values():
        t = train if isinstance(train, dict) else vars(train)
        for key in ("departure_city", "arrival_city"):
            if v := t.get(key):
                cities.add(v)
    if not cities:
        return "(none)"
    return ", ".join(sorted(cities))


def _format_existing_orders(environment: dict) -> str:
    """Format pre-existing orders for the LLM prompt."""
    orders = environment.get("orders", {})
    if not orders:
        return "(no existing orders)"

    lines = []
    for order_id, order in orders.items():
        # Handle both dict and object forms
        if isinstance(order, dict):
            otype = order.get("order_type", "unknown")
            status = order.get("status", "unknown")
            products = order.get("products", [])
        else:
            otype = getattr(order, "order_type", "unknown")
            status = getattr(order, "status", "unknown")
            products = getattr(order, "products", [])

        product_summary = []
        for p in products:
            if isinstance(p, dict):
                date = p.get("date", "N/A")
                qty = p.get("quantity", 1)
            else:
                date = getattr(p, "date", "N/A")
                qty = getattr(p, "quantity", 1)
            product_summary.append(f"date={date}, qty={qty}")

        lines.append(
            f"- order_id={order_id}, type={otype}, status={status}, "
            f"products=[{'; '.join(product_summary)}]"
        )
    return "\n".join(lines)


def extract_completeness_constraints(
    task: Task,
    llm_model: Optional[str] = None,
    llm_args: Optional[dict] = None,
    language: str = "english",
) -> CompletenessConstraints:
    """
    Extract completeness constraints from a task's instructions via a single LLM call.

    Args:
        task: The Task object (instructions + environment).
        llm_model: LLM model name for the extraction call.
        llm_args: Model-specific arguments (temperature, base_url, headers, etc.).
        language: Prompt language ("english" or "chinese").

    Returns:
        CompletenessConstraints with all booking count requirements.
    """
    if llm_model is None:
        llm_model = DEFAULT_LLM_EVALUATOR
    if llm_args is None:
        llm_args = dict(models.get(llm_model, models.get("default", {})))

    # Ensure enough output tokens for thinking + JSON response
    llm_args["max_tokens"] = max(llm_args.get("max_tokens", 0), 16384)

    env = task.environment or {}
    system_time = env.get("time", "unknown")

    user_profile = {}
    if task.user_scenario and task.user_scenario.user_profile:
        up = task.user_scenario.user_profile
        user_profile = up if isinstance(up, dict) else {}

    user_historical_behaviors = env.get("user_historical_behaviors", {})
    existing_orders = _format_existing_orders(env)
    available_cities = _format_available_cities(env)

    # Build the JSON schema for the LLM to follow
    json_schema = json.dumps(
        CompletenessConstraints.model_json_schema(),
        indent=2,
        ensure_ascii=False,
    )

    # Load the prompt template
    prompts = get_prompts(language)
    system_prompt = prompts.completeness_extraction_template.format(
        system_time=system_time,
        user_profile=json.dumps(user_profile, indent=2, ensure_ascii=False),
        user_historical_behaviors=json.dumps(
            user_historical_behaviors, indent=2, ensure_ascii=False
        ),
        existing_orders=existing_orders,
        available_cities=available_cities,
        json_schema=json_schema,
    )

    user_prompt = (
        f"Task ID: {task.id}\n\n"
        f"Instructions:\n{task.instructions}"
    )

    messages = [
        SystemMessage(role="system", content=system_prompt),
        UserMessage(role="user", content=user_prompt),
    ]

    # Single LLM call
    logger.info("Extracting completeness constraints for task %s with %s", task.id, llm_model)
    response = generate(model=llm_model, messages=messages, enable_think=True, **llm_args)

    # Parse and validate the response
    raw_content = response.content or ""
    try:
        parsed = _extract_json(raw_content)
        if parsed is None:
            raise ValueError("No JSON object found in LLM response")
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        if isinstance(parsed, dict) and "task_id" not in parsed:
            parsed["task_id"] = task.id
        constraints = CompletenessConstraints.model_validate(parsed)
    except Exception as e:
        raise ValueError(
            "LLM extraction failed for task %s: %s. Raw: %s" % (task.id, e, raw_content)
        )

    return constraints
