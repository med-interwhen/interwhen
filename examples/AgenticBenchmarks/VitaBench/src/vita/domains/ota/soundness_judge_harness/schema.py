"""
Schema for the NL-based soundness judge harness.

Constraints are extracted preserving natural language descriptions,
categorized by type for structured reasoning.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ConstraintCategory(str, Enum):
    DATE = "date"
    PRICE = "price"
    CITY = "city"
    QUANTITY = "quantity"
    DURATION = "duration"
    ENTITY_ATTRIBUTE = "entity_attribute"  # room type, seat class, ticket type, etc.
    OTHER = "other"


class EntityType(str, Enum):
    HOTEL = "hotel"
    FLIGHT = "flight"
    TRAIN = "train"
    ATTRACTION = "attraction"


class Constraint(BaseModel):
    """A single constraint extracted from user instructions."""

    id: str = Field(description="Unique ID, e.g. 'c1', 'c2'")
    category: ConstraintCategory
    entity_type: Optional[EntityType] = Field(
        default=None,
        description="Which booking entity this applies to, if specific",
    )
    description: str = Field(
        description="Natural language description of the constraint, "
        "preserving the user's intent as closely as possible",
    )


class ExtractedConstraintSet(BaseModel):
    """Full set of constraints extracted from a task's instructions."""

    task_id: str
    constraints: List[Constraint]


# ──────────────────────────────────────────────
# Memory store schema
# ──────────────────────────────────────────────


class MemoryEntry(BaseModel):
    """A single fact recorded from a read tool call."""

    source_tool: str = Field(description="Tool call that produced this info, e.g. 'search_hotels'")
    summary: str = Field(description="Key information relevant to constraint evaluation")


class TaskMemory(BaseModel):
    """Runtime memory accumulator for a single task simulation."""

    entries: List[MemoryEntry] = Field(default_factory=list)

    def append(self, source_tool: str, summary: str):
        self.entries.append(MemoryEntry(source_tool=source_tool, summary=summary))

    def render(self) -> str:
        if not self.entries:
            return "(no observations recorded yet)"
        lines = []
        for i, e in enumerate(self.entries, 1):
            lines.append(f"[{i}] ({e.source_tool}) {e.summary}")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Judgment schema
# ──────────────────────────────────────────────


class ConstraintVerdict(str, Enum):
    VIOLATED = "violated"
    CONSISTENT = "consistent"
    UNDETERMINED = "undetermined"


class ConstraintJudgment(BaseModel):
    """Judgment for a single constraint against a write tool call."""

    constraint_id: str
    verdict: ConstraintVerdict
    reasoning: str = Field(description="Brief explanation for the verdict")


class JudgmentResult(BaseModel):
    """Aggregated judgment across all constraints for one write tool call."""

    tool_call_name: str
    tool_call_args: dict
    judgments: List[ConstraintJudgment]

    @property
    def has_violation(self) -> bool:
        return any(j.verdict == ConstraintVerdict.VIOLATED for j in self.judgments)

    @property
    def violated_constraints(self) -> List[ConstraintJudgment]:
        return [j for j in self.judgments if j.verdict == ConstraintVerdict.VIOLATED]
