"""Manifest schema — the contract between spec, runner, and glue.

The LLM emits ``manifest.json`` alongside ``PolicyChecker.lean``.  Stages 2
and 3 read **only** the manifest to template the runner and the Python
glue; they never re-parse Lean source.

Schema (validated at load time):

.. code-block:: json

    {
      "domain": "telecom",
      "namespace": "Telecom",
      "data_models": [
        {
          "name": "Customer",
          "fields": [["customerId", "CustomerId"], ["fullName", "String"]],
          "snapshot_key": "customers",
          "snapshot_singular": false
        }
      ],
      "agent_state_fields": [
        ["customers", "List Customer", "customers"],
        ["identifiedCustomer", "Option CustomerId", "identified_customer"],
        ["today", "Int", "today"],
        ["userPhone", "String", "user_phone"]
      ],
      "id_types": ["CustomerId", "LineId", "BillId", "PlanId"],
      "enums": [
        {"name": "BillStatus", "ctors": ["overdue", "paid", "awaitingPayment"],
         "string_map": {"Overdue": "overdue", "Paid": "paid",
                        "Awaiting Payment": "awaitingPayment"}}
      ],
      "hyp_fields": [
        {"name": "travelling", "type": "Bool",
         "slm_question": "Is the user currently traveling outside their home network?"}
      ],
      "actions": [
        {"name": "SendPaymentRequest", "tool": "send_payment_request",
         "args": [["customerId", "CustomerId", "customer_id"],
                  ["billId", "BillId", "bill_id"]]}
      ],
      "rules": [
        {
          "name": "billOverdue",
          "phase": "pre",
          "tool": "send_payment_request",
          "source": "db",
          "inputs": [["s", "AgentState"], ["b", "BillId"]],
          "args_from": [["b", "args.bill_id", "BillId"]],
          "feedback_args": ["s", "b"]
        }
      ],
      "stuck_rules": [
        {"name": "someTrickyOne", "reason": "proof timed out 5 retries"}
      ]
    }

The Lean side names: ``check_<name>``, ``spec_<name>``, ``feedback_<name>``
for ``phase == "pre"``;  ``check_result_<name>`` etc. for ``phase == "post"``.

Field types use the literal Lean syntax.  The Python glue side only cares
about the structural fields (``snapshot_key``, ``args_from``, etc.) — the
type strings are echoed back verbatim into Lean parsers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Field-level dataclasses 

@dataclass
class DataModel:
    name: str
    fields: list[list[str]]               # [[fieldName, leanType, jsonKey?], ...]
    snapshot_key: str                      # JSON key in db_snapshot
    snapshot_singular: bool = False        # True ⇒ single record (e.g. Plan)

    @classmethod
    def from_dict(cls, d: dict) -> "DataModel":
        # Normalise each field to a 3-tuple [name, type, jsonKey].
        # If jsonKey is missing, fall back to camel→snake on the name.
        import re as _re
        def _camel_to_snake(name: str) -> str:
            s = _re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
            return _re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower()
        norm: list[list[str]] = []
        for fld in d.get("fields", []):
            f = list(fld)
            if len(f) == 2:
                f.append(_camel_to_snake(f[0]))
            elif len(f) >= 3 and not f[2]:
                f[2] = _camel_to_snake(f[0])
            norm.append(f[:3])
        return cls(
            name=d["name"],
            fields=norm,
            snapshot_key=d.get("snapshot_key") or d.get("key") or d["name"].lower() + "s",
            snapshot_singular=bool(d.get("snapshot_singular", False)),
        )


@dataclass
class EnumDef:
    name: str
    ctors: list[str]
    string_map: dict[str, str]             # policy string → ctor name

    @classmethod
    def from_dict(cls, d: dict) -> "EnumDef":
        ctors = d.get("ctors") or d.get("constructors") or d.get("cases") or d.get("values") or []
        return cls(
            name=d["name"],
            ctors=list(ctors),
            string_map=dict(d.get("string_map") or d.get("strings") or {}),
        )


@dataclass
class HypField:
    name: str
    type: str = "Bool"
    slm_question: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "HypField":
        return cls(
            name=d["name"],
            type=d.get("type", "Bool"),
            slm_question=d.get("slm_question") or d.get("question") or "",
        )


@dataclass
class Action:
    name: str
    tool: str
    args: list[list[str]]                  # [[leanName, leanType, jsonKey]]

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        return cls(
            name=d["name"],
            tool=d.get("tool", ""),
            args=[list(x) for x in d.get("args", [])],
        )


@dataclass
class Rule:
    name: str
    phase: str                             # "pre" | "post"
    tool: str                              # tool name (or "" for orphan)
    source: str                            # "db" | "args" | "history" | "context" | combos
    inputs: list[list[str]]                # [[leanName, leanType]]
    args_from: list[list[str]]             # [[leanName, jsonPath, leanType]]
    feedback_args: list[str] = field(default_factory=list)
    quote: str = ""                        # policy excerpt (for feedback / docs)
    section: str = ""


@dataclass
class StuckRule:
    name: str
    reason: str


# Top-level manifest 


@dataclass
class Manifest:
    domain: str
    namespace: str
    data_models: list[DataModel] = field(default_factory=list)
    agent_state_fields: list[list[str]] = field(default_factory=list)
    id_types: list[str] = field(default_factory=list)
    enums: list[EnumDef] = field(default_factory=list)
    hyp_fields: list[HypField] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    stuck_rules: list[StuckRule] = field(default_factory=list)
    # Per-data-model snapshot field aliases.  Lets the runtime DB use a
    # different JSON key than the manifest's data-model field.  Shape:
    #   { "<DataModelName>": { "<lean_field_jsonKey>": ["alt1", "alt2"] } }
    # The first existing alias wins.  Useful when the db emits e.g.
    # "account_status" but the spec expects "status".
    snapshot_remap: dict = field(default_factory=dict)

    # IO 

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Manifest":
        m = cls(domain=d.get("domain", ""), namespace=d.get("namespace", ""))
        m.data_models    = [DataModel.from_dict(x) for x in d.get("data_models", [])]
        m.agent_state_fields = [list(x) for x in d.get("agent_state_fields", [])]
        m.id_types       = list(d.get("id_types", []))
        m.enums          = [EnumDef.from_dict(x) for x in d.get("enums", [])]
        m.hyp_fields     = [HypField.from_dict(x) for x in d.get("hyp_fields", [])]
        m.actions        = [Action.from_dict(x) for x in d.get("actions", [])]
        m.rules          = [Rule(**x) for x in d.get("rules", [])]
        m.stuck_rules    = [StuckRule(**x) for x in d.get("stuck_rules", [])]
        m.snapshot_remap = dict(d.get("snapshot_remap", {}))
        return m

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        return cls.from_dict(json.loads(path.read_text()))

    # Query helpers used by the renderers

    def rules_for_tool(self, tool: str, phase: str) -> list[Rule]:
        return [r for r in self.rules if r.tool == tool and r.phase == phase]

    def all_pre_rules(self) -> list[Rule]:
        return [r for r in self.rules if r.phase == "pre"]

    def all_post_rules(self) -> list[Rule]:
        return [r for r in self.rules if r.phase == "post"]

    def stuck_count(self) -> int:
        return len(self.stuck_rules)
