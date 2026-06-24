"""
Order-based completeness checker for OTA tasks.

Checks final order state against completeness constraints to verify
that all required bookings exist. Does NOT check attribute correctness
(room type, seat class, etc.) — that's the soundness judge's job.

Operates on Order objects (Pydantic models) from the live DB, pre-split
into old_states / new_states by the orchestrator's get_states().
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from vita.domains.ota.completeness.schema import (
    AttractionCompleteness,
    CancelCompleteness,
    CompletenessConstraints,
    FlightCompleteness,
    HotelCompleteness,
    ModifyCompleteness,
    TrainCompleteness,
)

logger = logging.getLogger(__name__)


def _parse_date(date_str: str) -> datetime | None:
    """Try to parse a date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _get(obj: Any, key: str, default: Any = "") -> Any:
    """Attribute or dict access — works with Order objects and plain dicts."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_products(order: Any) -> list:
    """Get the products list from an order (object or dict)."""
    prods = _get(order, "products", [])
    return prods if prods else []


def check_completeness(
    new_orders: list,
    old_orders: list,
    constraints: CompletenessConstraints,
    environment: dict | None = None,
) -> list[str]:
    """
    Check that all completeness constraints are satisfied by the final orders.

    Args:
        new_orders: Orders created/modified during the simulation (from get_states new_states).
        old_orders: Pre-existing orders (from get_states old_states).
        constraints: Completeness constraints (conditional bucket is excluded).
        environment: Task environment dict for looking up flight/train cities by store_id.

    Returns:
        List of missing-booking descriptions. Empty = all complete.
    """
    environment = environment or {}
    missing: list[str] = []

    # Active (non-cancelled) new orders
    active_new = [o for o in new_orders if _get(o, "status") != "cancelled"]

    all_orders = list(old_orders) + list(new_orders)

    for con in constraints.hotel:
        missing.extend(_check_hotel(con, active_new))
    for con in constraints.flight:
        missing.extend(_check_flight(con, active_new, environment))
    for con in constraints.train:
        missing.extend(_check_train(con, active_new, environment))
    for con in constraints.attraction:
        missing.extend(_check_attraction(con, active_new))
    for con in constraints.cancel:
        missing.extend(_check_cancel(con, all_orders))
    for con in constraints.modify:
        missing.extend(_check_modify(con, new_orders, old_orders))

    return missing


_ORDER_TYPE_TO_CATALOG = {"flight": "flights", "train": "trains"}


def _resolve_city(order: Any, environment: dict, field: str) -> str:
    """Look up a city field from the environment via order's store_id."""
    catalog_key = _ORDER_TYPE_TO_CATALOG.get(_get(order, "order_type", ""), "")
    if not catalog_key:
        return ""
    catalog = environment.get(catalog_key, {})
    store_id = _get(order, "store_id", "")
    entry = catalog.get(store_id)
    if entry is None:
        return ""
    return _get(entry, field, "")


# ──────────────────────────────────────────────
# Hotel completeness
# ──────────────────────────────────────────────

def _check_hotel(con: HotelCompleteness, active_orders: list) -> list[str]:
    """
    Check hotel completeness: right date range, enough room-nights.
    City is not checked (hotel addresses are unstructured).
    """
    hotel_orders = [o for o in active_orders if _get(o, "order_type") == "hotel"]
    num_nights = con.num_nights or 1
    required_room_nights = con.num_rooms * num_nights

    expected_dates: set[str] | None = None
    if con.check_in_date:
        checkin = _parse_date(con.check_in_date)
        if checkin:
            expected_dates = {
                (checkin + timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(num_nights)
            }

    matched_count = 0
    for order in hotel_orders:
        for product in _get_products(order):
            product_date = _get(product, "date", "")
            if expected_dates and product_date not in expected_dates:
                continue
            matched_count += _get(product, "quantity", 1)

    if matched_count >= required_room_nights:
        return []

    prefix = f"[{con.id}]"
    detail = (
        f"check-in={con.check_in_date or 'any'}, "
        f"{con.num_rooms} room(s) x {num_nights} night(s)"
    )
    if not hotel_orders:
        return [f"{prefix} No hotel booked ({detail})"]
    return [
        f"{prefix} Hotel: "
        f"{matched_count}/{required_room_nights} room-night orders ({detail})"
    ]


# ──────────────────────────────────────────────
# Flight completeness
# ──────────────────────────────────────────────

def _check_flight(con: FlightCompleteness, active_orders: list, environment: dict) -> list[str]:
    """Check flight completeness: right route, right date, enough total quantity."""
    flight_orders = [o for o in active_orders if _get(o, "order_type") == "flight"]

    matched_qty = 0
    for order in flight_orders:
        if con.departure_city:
            dep = _resolve_city(order, environment, "departure_city")
            if dep.lower() != con.departure_city.lower():
                continue
        if con.arrival_city:
            arr = _resolve_city(order, environment, "arrival_city")
            if arr.lower() != con.arrival_city.lower():
                continue
        for product in _get_products(order):
            if con.date and _get(product, "date", "") != con.date:
                continue
            matched_qty += _get(product, "quantity", 0)

    if matched_qty >= con.quantity:
        return []

    prefix = f"[{con.id}]"
    route = f"{con.departure_city or '*'} -> {con.arrival_city or '*'}"
    detail = f"route={route}, date={con.date or 'any'}, qty={con.quantity}"
    if not flight_orders:
        return [f"{prefix} No flight booked ({detail})"]
    return [f"{prefix} Flight {route} on {con.date}: {matched_qty}/{con.quantity} tickets ({detail})"]


# ──────────────────────────────────────────────
# Train completeness
# ──────────────────────────────────────────────

def _check_train(con: TrainCompleteness, active_orders: list, environment: dict) -> list[str]:
    """Check train completeness: right route, right date, enough total quantity."""
    train_orders = [o for o in active_orders if _get(o, "order_type") == "train"]

    matched_qty = 0
    for order in train_orders:
        if con.departure_city:
            dep = _resolve_city(order, environment, "departure_city")
            if dep.lower() != con.departure_city.lower():
                continue
        if con.arrival_city:
            arr = _resolve_city(order, environment, "arrival_city")
            if arr.lower() != con.arrival_city.lower():
                continue
        for product in _get_products(order):
            if con.date and _get(product, "date", "") != con.date:
                continue
            matched_qty += _get(product, "quantity", 0)

    if matched_qty >= con.quantity:
        return []

    prefix = f"[{con.id}]"
    route = f"{con.departure_city or '*'} -> {con.arrival_city or '*'}"
    detail = f"route={route}, date={con.date or 'any'}, qty={con.quantity}"
    if not train_orders:
        return [f"{prefix} No train booked ({detail})"]
    return [f"{prefix} Train {route} on {con.date}: {matched_qty}/{con.quantity} tickets ({detail})"]


# ──────────────────────────────────────────────
# Attraction completeness
# ──────────────────────────────────────────────

def _check_attraction(con: AttractionCompleteness, active_orders: list) -> list[str]:
    """Check attraction completeness: right date, enough total quantity."""
    attr_orders = [o for o in active_orders if _get(o, "order_type") == "attraction"]

    matched_qty = 0
    for order in attr_orders:
        for product in _get_products(order):
            if con.date and _get(product, "date", "") != con.date:
                continue
            matched_qty += _get(product, "quantity", 0)

    if matched_qty >= con.quantity:
        return []

    prefix = f"[{con.id}]"
    detail = f"date={con.date or 'any'}, qty={con.quantity}"
    if not attr_orders:
        return [f"{prefix} No attraction booked ({detail})"]
    return [f"{prefix} Attraction on {con.date}: insufficient quantity ({detail})"]


# ──────────────────────────────────────────────
# Cancel completeness
# ──────────────────────────────────────────────

def _check_cancel(con: CancelCompleteness, all_orders: list) -> list[str]:
    """Check that the specified pre-existing order was cancelled."""
    for order in all_orders:
        if _get(order, "order_id") == con.order_id:
            status = _get(order, "status")
            if status == "cancelled":
                return []
            return [
                f"[{con.id}] Order {con.order_id} ({con.entity_type}) should be cancelled "
                f"but has status '{status}'"
            ]

    return []


# ──────────────────────────────────────────────
# Modify completeness
# ──────────────────────────────────────────────

def _check_modify(con: ModifyCompleteness, new_orders: list, old_orders: list) -> list[str]:
    """
    Check that the specified pre-existing order was modified.
    If it moved to new_orders (update_time changed), it was modified.
    """
    
    # search in new_orders first - if it's there, it's modified, no need to check old_orders
    for order in new_orders:
        if _get(order, "order_id") == con.order_id:
            return []
    
    # if it's not in new_orders, check if it exists in old_orders - if it does, then it was not modified
    for order in old_orders:
        if _get(order, "order_id") == con.order_id:
            return [
                f"[{con.id}] Order {con.order_id} ({con.entity_type}) should be modified "
                f"but was not updated"
            ]

    return []
