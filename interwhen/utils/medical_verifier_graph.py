"""
medical_verifier_graph.py  —  Graph-aware verifier (the new layer).

GraphAwareVerifier extends MedicalReasoningVerifierSnomedFirst the same way
that class extends MedicalReasoningVerifier — by overriding the three
section-verify methods and delegating to super() for everything that already
works (the per-section LLM grounding/validity/consistency checks). This file
adds exactly one new behaviour: after a section PASSES its existing checks,
it now also runs through GraphVerificationPipeline. That pipeline can:

  - confirm/disconfirm a claimed clinical relationship against SNOMED
  - detect that this section's claim contradicts an EARLIER section's
    claim, even though both were individually valid — the gap identified
    in the original design conversation, and the reason this module exists.

If the graph layer finds a cross-chunk contradiction, the section's result
is downgraded from PASS to FAIL, and the StructuredDiff becomes the feedback
shown to the solver (richer than the original wrong_claim/correction pair,
but rendered into the same [FEEDBACK] block format).

If the graph layer can't reach SNOMED, or has no opinion (e.g. a non-
checkable relation type, or the section made no clinical-relation claim at
all), the original LLM verifier's PASS/FAIL stands untouched. The graph
layer can only ADD a failure on top of an existing pass; it never overrides
a FAIL from the base checks into a PASS.

Where this plugs into the existing classes (no changes needed there):
  MedicalMonitor.__init__ builds:
      self.verifier = MedicalReasoningVerifierSnomedFirst(...)
  To turn the graph layer on, change exactly that one line to:
      self.verifier = GraphAwareVerifier(...)
  Everything else — step_extractor, verify(), fix(), decision_log — is
  untouched, because GraphAwareVerifier has the identical public interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .medical_graph import (
    GlobalGraphAccumulator,
    GraphVerificationPipeline,
    SnomedRelationshipClient,
    StructuredDiff,
)
from .medical_verifier import LocalVLLMClient, SnomedClient
from .medical_verifier_snomed import MedicalReasoningVerifierSnomedFirst, SnomedFirstConfig


@dataclass
class GraphVerifierConfig(SnomedFirstConfig):
    run_graph_checks:        bool = True
    # Sections where a graph-level cross-chunk contradiction should be able
    # to downgrade an otherwise-passing section. CONCLUSION is deliberately
    # excluded by default: contradicting the conclusion against the
    # accumulated graph is a much stronger claim and is better surfaced as
    # a diagnostic than as a silent retry loop at the very last section.
    graph_enforced_sections: Tuple[str, ...] = ("INFERENCE", "OPTION_COMPARISON")


class GraphAwareVerifier(MedicalReasoningVerifierSnomedFirst):
    """
    Adds CUI-grounded graph verification on top of the existing structured
    verifier, without changing any existing pass/fail behaviour on its own.
    """

    def __init__(
        self,
        vllm:                 LocalVLLMClient,
        snomed:                Optional[SnomedClient]             = None,
        config:                Optional[GraphVerifierConfig]      = None,
        compact_case:          str                                = "",
        snomed_cache:          Optional[Dict[str, str]]            = None,
        relationship_client:   Optional[SnomedRelationshipClient]  = None,
    ):
        super().__init__(
            vllm=vllm, snomed=snomed, config=config or GraphVerifierConfig(),
            compact_case=compact_case, snomed_cache=snomed_cache,
        )
        # relationship_client defaults to a real SnomedRelationshipClient
        # (Snowstorm). If construction fails (missing config) or the
        # caller passes None explicitly, the graph pipeline degrades to
        # "unverifiable everything" rather than raising — see medical_graph.py.
        self.relationship_client = (
            relationship_client if relationship_client is not None
            else self._try_build_relationship_client()
        )
        self.graph_pipeline = GraphVerificationPipeline(
            vllm=self.vllm,
            relationship_client=self.relationship_client,
        )
        # Exposed for callers/tests that want to inspect the running graph
        # directly (e.g. for the ablation study: node_count/edge_count/
        # contradiction_log over the course of a full trace).
        self.global_graph_accumulator: GlobalGraphAccumulator = self.graph_pipeline.accumulator
        self.graph_diff_log: List[dict] = []

    @staticmethod
    def _try_build_relationship_client() -> Optional[SnomedRelationshipClient]:
        try:
            return SnomedRelationshipClient()
        except Exception as e:
            print(f"  [GraphAwareVerifier] Relationship client unavailable: {e}")
            return None

    # ── Shared graph-check helper ────────────────────────────────────────────

    def _run_graph_check(
        self,
        section_type: str,
        paragraph:    str,
        options:      dict,
        question:     str,
    ) -> Tuple[bool, Optional[StructuredDiff]]:
        """
        Runs the graph pipeline for a section that already passed its base
        verification. Returns (graph_passed, diff). graph_passed is True
        unless run_graph_checks is off, the section type isn't enforced, or
        no blocking issue was found.
        """
        cfg = self.config
        if not getattr(cfg, "run_graph_checks", True):
            return True, None
        if section_type not in getattr(cfg, "graph_enforced_sections", ()):
            return True, None

        opts_text = self._options_text(options)
        diff = self.graph_pipeline.run(paragraph, question=question, options_text=opts_text)
        self.graph_diff_log.append({"section_type": section_type, **diff.to_dict()})

        return (not diff.has_blocking_issue), diff

    # ── Overrides ──────────────────────────────────────────────────────────────

    def _verify_inference(
        self,
        paragraph: str,
        options:   dict,
        question:  str  = "",
        _retried:  bool = False,
    ) -> Tuple[bool, Optional[str]]:
        passed, feedback = super()._verify_inference(paragraph, options, question, _retried=_retried)
        if not passed:
            return passed, feedback   # base check already failed; graph adds nothing here

        graph_ok, diff = self._run_graph_check("INFERENCE", paragraph, options, question)
        if graph_ok:
            return True, None

        # Base check passed, graph check did not: downgrade to FAIL with the
        # structured diff as feedback. This is the new behaviour — a section
        # that would have silently passed before now gets caught.
        return False, diff.to_feedback_text()

    def _verify_option_comparison(
        self,
        paragraph: str,
        options:   dict,
        question:  str  = "",
        _retried:  bool = False,
    ) -> Tuple[bool, Optional[str]]:
        passed, feedback = super()._verify_option_comparison(paragraph, options, question, _retried=_retried)
        if not passed:
            return passed, feedback

        graph_ok, diff = self._run_graph_check("OPTION_COMPARISON", paragraph, options, question)
        if graph_ok:
            return True, None
        return False, diff.to_feedback_text()

    def _verify_conclusion(
        self,
        paragraph: str,
        options:   dict,
        question:  str = "",
    ) -> Tuple[bool, Optional[str]]:
        passed, feedback = super()._verify_conclusion(paragraph, options, question)
        if not passed:
            return passed, feedback

        # CONCLUSION is excluded from graph_enforced_sections by default
        # (see GraphVerifierConfig docstring) — this call is a no-op unless
        # the caller explicitly opts in via config.
        graph_ok, diff = self._run_graph_check("CONCLUSION", paragraph, options, question)
        if graph_ok:
            return True, None
        return False, diff.to_feedback_text()

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def graph_summary(self) -> dict:
        """Convenience for logging/ablation: snapshot of the running global
        graph at any point in the trace."""
        return {
            "node_count":         self.global_graph_accumulator.node_count(),
            "edge_count":         self.global_graph_accumulator.edge_count(),
            "contradiction_count": len(self.global_graph_accumulator.contradiction_log),
            "contradictions":     list(self.global_graph_accumulator.contradiction_log),
        }
