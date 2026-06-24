"""
Pydantic models for OTA completeness constraints.

These are intentionally minimal — only fields needed to verify that
the right *number* of bookings exist. Attribute-level correctness
(room type, seat type, ticket type, etc.) is handled by the LLM
soundness judge.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Booking completeness constraints
# ──────────────────────────────────────────────

class HotelCompleteness(BaseModel):
    """Hotel booking completeness requirement."""

    id: str = Field(description="Unique constraint ID, e.g. 'hotel_1'")
    check_in_date: Optional[str] = Field(
        default=None,
        description="Check-in date in YYYY-MM-DD format",
    )
    num_rooms: int = Field(default=1, description="Number of rooms to book")
    num_nights: Optional[int] = Field(default=None, description="Number of nights to stay")
    description: str = ""


class FlightCompleteness(BaseModel):
    """Flight booking completeness requirement."""

    id: str = Field(description="Unique constraint ID, e.g. 'flight_1'")
    departure_city: Optional[str] = None
    arrival_city: Optional[str] = None
    date: Optional[str] = Field(default=None, description="Departure date YYYY-MM-DD")
    quantity: int = Field(default=1, description="Number of tickets")
    description: str = ""


class TrainCompleteness(BaseModel):
    """Train booking completeness requirement."""

    id: str = Field(description="Unique constraint ID, e.g. 'train_1'")
    departure_city: Optional[str] = None
    arrival_city: Optional[str] = None
    date: Optional[str] = Field(default=None, description="Departure date YYYY-MM-DD")
    quantity: int = Field(default=1, description="Number of tickets")
    description: str = ""


class AttractionCompleteness(BaseModel):
    """Attraction ticket completeness requirement."""

    id: str = Field(description="Unique constraint ID, e.g. 'attraction_1'")
    date: Optional[str] = Field(default=None, description="Visit date YYYY-MM-DD")
    quantity: int = Field(default=1, description="Number of tickets")
    description: str = ""


# ──────────────────────────────────────────────
# Cancel / modify completeness constraints
# ──────────────────────────────────────────────

class CancelCompleteness(BaseModel):
    """Requirement that a pre-existing order be cancelled."""

    id: str
    entity_type: Literal["hotel", "flight", "train", "attraction"]
    order_id: str = Field(
        description="Order ID of the pre-existing order that should be cancelled",
    )
    description: str = ""


class ModifyCompleteness(BaseModel):
    """Requirement that a pre-existing order be modified."""

    id: str
    entity_type: Literal["flight", "train"]
    order_id: str = Field(
        description="Order ID of the pre-existing order that should be modified",
    )
    description: str = ""


# ──────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────

AnyBookingCompleteness = Union[
    HotelCompleteness,
    FlightCompleteness,
    TrainCompleteness,
    AttractionCompleteness,
]

AnyCompleteness = Union[
    HotelCompleteness,
    FlightCompleteness,
    TrainCompleteness,
    AttractionCompleteness,
    CancelCompleteness,
    ModifyCompleteness,
]


# ──────────────────────────────────────────────
# Conditional (excluded from completeness checking)
# ──────────────────────────────────────────────

# Bookings that depend on ANY runtime condition (weather, price,
# availability, distance, etc.) are placed here and excluded from
# completeness checking. The soundness judge handles whether the
# agent took the correct branch.


# ──────────────────────────────────────────────
# Top-level extraction result
# ──────────────────────────────────────────────

class CompletenessConstraints(BaseModel):
    """Completeness constraints extracted from task instructions.

    Only captures *what must exist* at the end of the conversation
    (counts, cities, routes, dates). Attribute correctness (room type,
    seat type, etc.) is left to the soundness judge.

    Bookings that depend on any runtime condition should go in
    ``conditional`` so they are excluded from completeness checking.
    """

    task_id: str

    hotel: List[HotelCompleteness] = Field(default_factory=list)
    flight: List[FlightCompleteness] = Field(default_factory=list)
    train: List[TrainCompleteness] = Field(default_factory=list)
    attraction: List[AttractionCompleteness] = Field(default_factory=list)

    cancel: List[CancelCompleteness] = Field(default_factory=list)
    modify: List[ModifyCompleteness] = Field(default_factory=list)

    conditional: List[AnyBookingCompleteness] = Field(
        default_factory=list,
        description="Bookings that depend on any runtime condition "
        "(weather, price, availability, distance, etc.). "
        "These are EXCLUDED from completeness checking.",
    )
