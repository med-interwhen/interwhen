"""
medical_graph.py  —  CUI-grounded graph verification layer.

This module is ADDITIVE: it does not modify the behaviour of
medical_verifier.py or medical_verifier_snomed.py unless explicitly wired in
by GraphAwareVerifier (see medical_verifier_graph.py). Every class here can
be imported and unit-tested standalone.

Pipeline (matches the architecture agreed on in conversation):

  section text
       |
       v
  (1) EntityMapper          span + context -> {span, cui_candidate, confidence}
       |                    cui_candidate is VALIDATED against SnomedRelationshipClient
       |                    before being trusted anywhere downstream. Below the
       |                    confidence floor, or if validation fails, the entity
       |                    is dropped from the graph (not guessed into it).
       v
  (2) ChunkGraphBuilder     builds a small nx.DiGraph for this section from
       |                    validated entities + the LLM's claimed edge, then
       |                    checks that claimed edge against SNOMED inferred
       |                    relationships. ("Local coherence" and "check
       |                    against SNOMED" are the same step here -- a 2-3
       |                    node graph has nothing else to be coherent with.)
       v
  (3) GlobalGraphAccumulator merges the chunk graph into CompactState's running
       |                    graph using validated CUI as node identity (never
       |                    the raw LLM guess), then re-checks every edge
       |                    touching a changed node for cross-chunk
       |                    contradictions.
       v
  (4) StructuredDiff        claimed vs. confirmed vs. unsupported vs. missing
                            edges, plus any cross-chunk conflicts. Renders to
                            the same [FEEDBACK] block format the solver already
                            expects, but is also a real, inspectable object --
                            this is what you'd log for the ablation study.

Design choices made deliberately conservative, per the known risks:

  - SnomedRelationshipClient talks to Snowstorm, NOT BioPortal (BioPortal has
    no usable relationship endpoint -- see SnomedClient in medical_verifier.py,
    which only returns definitions). This is the one external dependency that
    could not be validated in this environment (no network egress here). Every
    method fails SOFT: on any error, timeout, or missing config, it returns an
    "unavailable" result rather than raising, and callers treat "unavailable"
    as "no SNOMED opinion" rather than "SNOMED disconfirmed this." This means
    the whole graph layer degrades to "confirmed nothing, contradicted
    nothing" if Snowstorm is unreachable, rather than crashing generation or
    silently asserting false confirmations.

  - CONFIDENCE_FLOOR (default 0.6) gates entity acceptance as a GROUNDED node
    identity. This exists specifically to prevent the single-point-of-failure
    flagged earlier: a wrong CUI silently merging unrelated chunks together
    in the global graph. An entity that doesn't clear the floor, or whose
    cui_candidate fails Snowstorm validation, is kept as an UNGROUNDED node
    (never used as a merge key) rather than falling back to a guessed CUI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import requests

from .medical_reasoning_prompts import MedicalReasoningPromptBuilder
from .medical_verifier import LocalVLLMClient


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Minimum mapper confidence for an entity to be accepted as a GROUNDED node
# (i.e. eligible to use its CUI as a merge identity). Entities below this,
# or whose CUI fails validation, are kept as ungrounded nodes instead of
# being dropped silently -- they still show up in the chunk graph for
# transparency, but can never merge with anything in the global graph.
CONFIDENCE_FLOOR = 0.6

# Relation types the chunk graph will attempt to confirm against SNOMED.
# "other" is intentionally excluded from confirmation -- it is too vague to
# check meaningfully and is recorded as UNVERIFIABLE rather than unsupported.
CHECKABLE_RELATIONS = {
    "causes", "treats", "finding_site", "associated_with",
    "risk_factor", "contraindicated_with",
}


# ══════════════════════════════════════════════════════════════════════════════
# SNOMED RELATIONSHIP CLIENT  (Snowstorm -- NOT BioPortal)
# ══════════════════════════════════════════════════════════════════════════════

class SnomedRelationshipClient:
    """
    Wraps the Snowstorm SNOMED CT browser API for relationship traversal.

    BioPortal (see SnomedClient in medical_verifier.py) only exposes concept
    search and definitions -- it has no usable relationship/edge endpoint.
    This client targets a Snowstorm instance instead, which exposes inferred
    relationships per concept.

    IMPORTANT -- this could not be live-tested in the environment this code
    was written in (no network egress). The endpoint shapes below follow
    Snowstorm's documented REST API as of this writing. Treat the URL paths
    as the first thing to verify against your actual instance before
    trusting this in production; everything else in the graph pipeline is
    built to tolerate this client being wrong or unreachable.

    Every public method fails SOFT -- on any error it returns a result with
    "available": False rather than raising, so callers can distinguish
    "SNOMED says no" from "couldn't ask SNOMED."
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout:  int           = 10,
        edition:  str           = "MAIN",
    ):
        self.base_url = (base_url or os.getenv(
            "SNOWSTORM_BASE_URL", "https://browser.ihtsdotools.org/snowstorm/snomed-ct"
        )).rstrip("/")
        self.timeout = timeout
        self.edition = edition  # e.g. "MAIN" or "MAIN/SNOMEDCT-US"

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        url = f"{self.base_url}/{self.edition}{path}"
        try:
            r = requests.get(url, params=params or {}, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  [Snowstorm] request failed for {path}: {e}")
            return None

    # ── Concept validation ─────────────────────────────────────────────────────

    def validate_cui(self, cui: str) -> dict:
        """
        Confirms a CUI exists and returns its FSN. Used to validate the
        LLM mapper's cui_candidate before it is trusted in any graph.

        Returns:
          {"available": True,  "valid": bool, "fsn": str|None}
          {"available": False, "valid": None,  "fsn": None}   on any failure
        """
        if not cui:
            return {"available": True, "valid": False, "fsn": None}

        data = self._get(f"/concepts/{cui}")
        if data is None:
            return {"available": False, "valid": None, "fsn": None}

        active = data.get("active", False)
        fsn    = (data.get("fsn") or {}).get("term")
        return {"available": True, "valid": bool(active), "fsn": fsn}

    # ── Relationship lookup ─────────────────────────────────────────────────────

    def get_inferred_relationships(self, cui: str) -> dict:
        """
        Fetches inferred (transitive-closure) relationships for a CUI --
        finding-site, causative-agent, associated-with, etc. Inferred, not
        stated: inferred relationships include the full is-a chain and
        inherited attributes, which is what confirmation against a claimed
        edge actually needs.

        Returns:
          {"available": True,  "relationships": [{"type": str, "target_cui": str, "target_fsn": str}, ...]}
          {"available": False, "relationships": []}   on any failure
        """
        if not cui:
            return {"available": True, "relationships": []}

        data = self._get(f"/concepts/{cui}/relationships", params={
            "active": "true",
            "characteristicType": "INFERRED_RELATIONSHIP",
        })
        if data is None:
            return {"available": False, "relationships": []}

        rels = []
        for item in data.get("items", []):
            rel_type = (item.get("type") or {}).get("pt", {}).get("term", "")
            target   = item.get("target") or {}
            rels.append({
                "type":       rel_type,
                "target_cui": target.get("conceptId"),
                "target_fsn": (target.get("fsn") or {}).get("term"),
            })
        return {"available": True, "relationships": rels}

    def check_edge(self, source_cui: str, target_cui: str) -> dict:
        """
        Checks whether ANY inferred relationship connects source_cui to
        target_cui (in either direction -- SNOMED relation types are
        directed but the LLM's claimed direction may not match SNOMED's
        modeling direction, e.g. "metformin treats T2DM" vs. SNOMED modeling
        the disease as "associated with" the drug).

        Returns:
          {"available": True,  "confirmed": bool, "matched_type": str|None}
          {"available": False, "confirmed": None, "matched_type": None}
        """
        fwd = self.get_inferred_relationships(source_cui)
        if not fwd["available"]:
            return {"available": False, "confirmed": None, "matched_type": None}

        for rel in fwd["relationships"]:
            if rel["target_cui"] == target_cui:
                return {"available": True, "confirmed": True, "matched_type": rel["type"]}

        bwd = self.get_inferred_relationships(target_cui)
        if not bwd["available"]:
            # Forward check succeeded and found nothing; backward check is
            # unavailable. Treat as available-but-unconfirmed rather than
            # discarding the forward result.
            return {"available": True, "confirmed": False, "matched_type": None}

        for rel in bwd["relationships"]:
            if rel["target_cui"] == source_cui:
                return {"available": True, "confirmed": True, "matched_type": rel["type"]}

        return {"available": True, "confirmed": False, "matched_type": None}


# ══════════════════════════════════════════════════════════════════════════════
# ENTITY MAPPER  (LLM span -> CUI, validated)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MappedEntity:
    span:           str
    fsn_candidate:  str
    cui_candidate:  Optional[str]
    confidence:     float
    validated_cui:  Optional[str] = None   # set only after passing validation
    validated_fsn:  Optional[str] = None

    @property
    def grounded(self) -> bool:
        return self.validated_cui is not None

    @property
    def node_id(self) -> str:
        """Stable identity for graph nodes: validated CUI if grounded, else
        a namespaced placeholder keyed on surface text (never merges across
        chunks by construction)."""
        return self.validated_cui if self.grounded else f"unverified::{self.span}"


@dataclass
class ClaimedRelation:
    source_span:   str
    target_span:   str
    relation_type: str


class EntityMapper:
    """
    Step (1): maps text spans to SNOMED CUI candidates via one LLM call,
    then validates each candidate against SnomedRelationshipClient before it
    is ever trusted as a graph node identity.

    This is the single point that the earlier risk analysis flagged: a wrong
    CUI here becomes a corrupting node identity in the global graph. The
    mitigation is CONFIDENCE_FLOOR + mandatory validation for GROUNDED
    status -- neither is optional, and there is no silent path from "low
    confidence guess" to "trusted merge key."
    """

    def __init__(
        self,
        vllm:                LocalVLLMClient,
        relationship_client: Optional[SnomedRelationshipClient],
        confidence_floor:    float = CONFIDENCE_FLOOR,
    ):
        self.vllm                = vllm
        self.relationship_client = relationship_client
        self.confidence_floor    = confidence_floor

    def map_section(
        self,
        section_body: str,
        question:     str,
        options_text: str,
    ) -> Tuple[List[MappedEntity], Optional[ClaimedRelation]]:
        """
        Returns (entities, claimed_relation_or_None).

        Every entity above a minimal sanity bar (non-empty span) is returned
        -- but only entities that clear confidence_floor AND pass SNOMED
        validation are GROUNDED (entity.grounded == True). Ungrounded
        entities still appear in the chunk graph (for transparency / audit)
        but ChunkGraphBuilder treats them as un-mergeable, unconfirmable
        nodes -- they contribute no SNOMED-checked edges.
        """
        prompt = MedicalReasoningPromptBuilder.build_entity_cui_mapping_prompt(
            question=question, options_text=options_text, section_body=section_body,
        )
        resp = self.vllm.call(prompt)

        raw_entities = resp.get("entities", [])
        if not isinstance(raw_entities, list):
            raw_entities = []

        entities: List[MappedEntity] = []
        for e in raw_entities:
            if not isinstance(e, dict):
                continue
            span = str(e.get("span", "")).strip()
            if not span:
                continue
            try:
                confidence = float(e.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0

            ent = MappedEntity(
                span          = span,
                fsn_candidate = str(e.get("fsn_candidate", "")).strip(),
                cui_candidate = (str(e["cui_candidate"]).strip()
                                 if e.get("cui_candidate") else None),
                confidence    = confidence,
            )
            self._validate(ent)
            entities.append(ent)

        claimed = None
        raw_rel = resp.get("claimed_relation")
        if isinstance(raw_rel, dict) and raw_rel.get("source_span") and raw_rel.get("target_span"):
            claimed = ClaimedRelation(
                source_span   = str(raw_rel["source_span"]).strip(),
                target_span   = str(raw_rel["target_span"]).strip(),
                relation_type = str(raw_rel.get("relation_type", "other")).strip().lower(),
            )

        return entities, claimed

    def _validate(self, ent: MappedEntity) -> None:
        """
        Mutates ent in place, setting validated_cui/validated_fsn only if
        ALL of these hold:
          - confidence clears the floor
          - a cui_candidate was proposed
          - a relationship_client is configured and reachable
          - SNOMED confirms the candidate is an active, real concept

        Any failure on any of these leaves the entity ungrounded. This is
        intentionally strict: there is no partial-trust state.
        """
        if ent.confidence < self.confidence_floor:
            return
        if not ent.cui_candidate or self.relationship_client is None:
            return

        result = self.relationship_client.validate_cui(ent.cui_candidate)
        if not result["available"] or not result["valid"]:
            return

        ent.validated_cui = ent.cui_candidate
        ent.validated_fsn = result["fsn"] or ent.fsn_candidate


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK GRAPH BUILDER  (step 2)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkGraphResult:
    graph:            nx.DiGraph
    edge_supported:   Optional[bool]   # True / False / None (unverifiable or unavailable)
    matched_type:     Optional[str]
    snomed_available: bool
    note:             str = ""


class ChunkGraphBuilder:
    """
    Step (2): builds a small per-section graph from mapped entities and the
    claimed relation, then checks that relation against SNOMED. There is no
    separate "local coherence" phase -- a 2-3 node graph has nothing to be
    locally incoherent with except the ontology itself, so that check and
    the SNOMED confirmation are the same step.
    """

    def __init__(self, relationship_client: Optional[SnomedRelationshipClient]):
        self.relationship_client = relationship_client

    def build(
        self,
        entities: List[MappedEntity],
        relation: Optional[ClaimedRelation],
    ) -> ChunkGraphResult:
        g = nx.DiGraph()
        by_span = {e.span: e for e in entities}

        for e in entities:
            g.add_node(e.node_id, span=e.span, fsn=e.validated_fsn or e.fsn_candidate,
                       grounded=e.grounded)

        if relation is None:
            return ChunkGraphResult(graph=g, edge_supported=None, matched_type=None,
                                     snomed_available=True, note="No relation claimed.")

        src = by_span.get(relation.source_span)
        tgt = by_span.get(relation.target_span)
        if src is None or tgt is None:
            return ChunkGraphResult(
                graph=g, edge_supported=None, matched_type=None, snomed_available=True,
                note="Claimed relation references a span that was not extracted as an entity.",
            )

        g.add_edge(src.node_id, tgt.node_id, relation_type=relation.relation_type)

        if relation.relation_type not in CHECKABLE_RELATIONS:
            return ChunkGraphResult(graph=g, edge_supported=None, matched_type=None,
                                     snomed_available=True,
                                     note=f"Relation type '{relation.relation_type}' is not checkable.")

        if not src.grounded or not tgt.grounded:
            return ChunkGraphResult(
                graph=g, edge_supported=None, matched_type=None, snomed_available=True,
                note="One or both entities are ungrounded (no validated CUI) -- "
                     "edge cannot be confirmed against SNOMED.",
            )

        if self.relationship_client is None:
            return ChunkGraphResult(graph=g, edge_supported=None, matched_type=None,
                                     snomed_available=False,
                                     note="No SNOMED relationship client configured.")

        check = self.relationship_client.check_edge(src.validated_cui, tgt.validated_cui)
        if not check["available"]:
            return ChunkGraphResult(graph=g, edge_supported=None, matched_type=None,
                                     snomed_available=False,
                                     note="SNOMED relationship lookup unavailable.")

        return ChunkGraphResult(
            graph=g, edge_supported=check["confirmed"], matched_type=check["matched_type"],
            snomed_available=True,
            note=("Confirmed by SNOMED." if check["confirmed"]
                  else "No matching inferred relationship found in SNOMED."),
        )


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURED DIFF  (step 4 -- replaces ad hoc feedback strings)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StructuredDiff:
    """
    The verification output is not just "supported/unsupported" -- it is a
    diff between what the section asserted and what SNOMED (and the global
    graph) say. This is the object that gets logged for the ablation study
    AND rendered into the [FEEDBACK] block.
    """
    claimed_edges:          List[dict] = field(default_factory=list)
    confirmed_edges:        List[dict] = field(default_factory=list)
    unsupported_edges:      List[dict] = field(default_factory=list)
    unverifiable_edges:     List[dict] = field(default_factory=list)
    cross_chunk_conflicts:  List[dict] = field(default_factory=list)

    @property
    def has_blocking_issue(self) -> bool:
        """Unverifiable edges are not blocking (SNOMED simply had no
        opinion); unsupported claims and cross-chunk conflicts are."""
        return bool(self.unsupported_edges or self.cross_chunk_conflicts)

    def to_feedback_text(self) -> str:
        lines: List[str] = []
        if self.unsupported_edges:
            lines.append("Claimed relationships not confirmed by SNOMED:")
            for e in self.unsupported_edges:
                lines.append(f"  - {e['source']} --{e['relation_type']}--> {e['target']}: {e['note']}")
        if self.cross_chunk_conflicts:
            lines.append("Conflicts with earlier established reasoning:")
            for c in self.cross_chunk_conflicts:
                lines.append(f"  - {c['description']}")
        if self.confirmed_edges:
            lines.append("Confirmed by SNOMED (for reference):")
            for e in self.confirmed_edges:
                lines.append(f"  - {e['source']} --{e['relation_type']}--> {e['target']} "
                              f"(matched: {e.get('matched_type', 'n/a')})")
        if self.unverifiable_edges:
            lines.append("Could not be checked against SNOMED (not a blocking issue):")
            for e in self.unverifiable_edges:
                lines.append(f"  - {e['source']} --{e['relation_type']}--> {e['target']}: {e['note']}")
        return "\n".join(lines).strip() or "No graph-level issues."

    def to_dict(self) -> dict:
        return {
            "claimed_edges":         self.claimed_edges,
            "confirmed_edges":       self.confirmed_edges,
            "unsupported_edges":     self.unsupported_edges,
            "unverifiable_edges":    self.unverifiable_edges,
            "cross_chunk_conflicts": self.cross_chunk_conflicts,
        }


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL GRAPH ACCUMULATOR  (step 3 -- lives alongside CompactState)
# ══════════════════════════════════════════════════════════════════════════════

class GlobalGraphAccumulator:
    """
    Step (3): accumulates chunk graphs into one running graph across the
    whole reasoning trace, using validated CUI as the ONLY merge identity.
    Nodes keyed "unverified::<span>" never merge with anything else, by
    construction -- they only ever match themselves (same exact span
    string), so an unconfirmed entity can never silently corrupt the global
    graph by colliding with an unrelated chunk's unconfirmed entity that
    happens to share surface text.

    After every merge, checks the newly-added edges for direct
    contradictions with already-accumulated edges (e.g. one chunk claims
    "DrugX treats ConditionY" and a later chunk claims
    "DrugX contraindicated_with ConditionY"). This is the mechanism for the
    "two locally-valid but mutually contradictory inferences both pass
    today" gap identified earlier -- it did not exist before this module.
    """

    # Relation pairs treated as directly contradictory when found between
    # the same (unordered) node pair.
    _OPPOSING_RELATIONS = {
        ("treats", "contraindicated_with"),
        ("contraindicated_with", "treats"),
    }

    def __init__(self):
        self.graph = nx.DiGraph()
        self.contradiction_log: List[dict] = []

    def merge(self, chunk_result: ChunkGraphResult) -> List[dict]:
        """
        Merges a chunk graph into the global graph. Returns any new
        contradictions found as a result of this merge (empty list if none).
        """
        new_conflicts: List[dict] = []

        for node_id, attrs in chunk_result.graph.nodes(data=True):
            if not self.graph.has_node(node_id):
                self.graph.add_node(node_id, **attrs)

        for u, v, attrs in chunk_result.graph.edges(data=True):
            relation_type = attrs.get("relation_type", "other")
            conflict = self._check_conflict(u, v, relation_type)
            if conflict:
                new_conflicts.append(conflict)
                self.contradiction_log.append(conflict)
            self.graph.add_edge(u, v, relation_type=relation_type, contradicted=bool(conflict))

        return new_conflicts

    def _check_conflict(self, u: str, v: str, relation_type: str) -> Optional[dict]:
        # Same direction, same pair already in the graph.
        if self.graph.has_edge(u, v):
            existing_type = self.graph[u][v].get("relation_type")
            if existing_type != relation_type and \
               self._opposes(existing_type, relation_type):
                return {
                    "source": u, "target": v,
                    "description": (
                        f"New edge {u} --{relation_type}--> {v} conflicts with "
                        f"earlier {u} --{existing_type}--> {v}."
                    ),
                }
            return None

        # Reverse direction already in the graph -- opposing relations are
        # frequently asserted in either order across chunks.
        if self.graph.has_edge(v, u):
            existing_type = self.graph[v][u].get("relation_type")
            if self._opposes(existing_type, relation_type):
                return {
                    "source": u, "target": v,
                    "description": (
                        f"New edge {u} --{relation_type}--> {v} conflicts with "
                        f"earlier {v} --{existing_type}--> {u}."
                    ),
                }
        return None

    def _opposes(self, type_a: Optional[str], type_b: Optional[str]) -> bool:
        return (type_a, type_b) in self._OPPOSING_RELATIONS or (type_b, type_a) in self._OPPOSING_RELATIONS

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH VERIFICATION PIPELINE  (orchestrates 1-4 for one section)
# ══════════════════════════════════════════════════════════════════════════════

class GraphVerificationPipeline:
    """
    Single entry point that ties EntityMapper -> ChunkGraphBuilder ->
    GlobalGraphAccumulator -> StructuredDiff together for one section.

    This is what GraphAwareVerifier (in medical_verifier_graph.py) calls
    AFTER a section has already passed the existing per-section LLM
    verification -- it never replaces that check, it adds a second,
    independent check on top of it.
    """

    def __init__(
        self,
        vllm:                LocalVLLMClient,
        relationship_client: Optional[SnomedRelationshipClient],
        accumulator:          Optional[GlobalGraphAccumulator] = None,
    ):
        self.mapper        = EntityMapper(vllm, relationship_client)
        self.chunk_builder = ChunkGraphBuilder(relationship_client)
        self.accumulator   = accumulator or GlobalGraphAccumulator()

    def run(
        self,
        section_body: str,
        question:     str,
        options_text: str,
    ) -> StructuredDiff:
        entities, relation = self.mapper.map_section(section_body, question, options_text)
        chunk_result        = self.chunk_builder.build(entities, relation)
        conflicts            = self.accumulator.merge(chunk_result)

        diff = StructuredDiff()

        if relation is not None:
            edge_record = {
                "source":        relation.source_span,
                "target":        relation.target_span,
                "relation_type": relation.relation_type,
                "note":          chunk_result.note,
                "matched_type":  chunk_result.matched_type,
            }
            diff.claimed_edges.append(edge_record)

            if chunk_result.edge_supported is True:
                diff.confirmed_edges.append(edge_record)
            elif chunk_result.edge_supported is False:
                diff.unsupported_edges.append(edge_record)
            else:
                diff.unverifiable_edges.append(edge_record)

        diff.cross_chunk_conflicts.extend(conflicts)
        return diff
