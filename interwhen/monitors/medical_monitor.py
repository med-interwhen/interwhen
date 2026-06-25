"""
medical_monitor.py  —  VerifyMonitor for structured medical reasoning.

Key changes from the free-reasoning version:
  - step_extractor triggers on section CLOSE TAGS ([/OBSERVATION], [/INFERENCE],
    [/OPTION_COMPARISON], [/CONCLUSION]) instead of bare \\n\\n. This means the
    verifier fires exactly once per complete section — no more spurious triggers
    on blank lines inside sections, and no missed triggers when a section ends
    without a trailing blank line.
  - UNKNOWN mid-section trigger is still supported: if the solver writes
    UNKNOWN: … on its own line, we trigger immediately so the verifier can
    inject SNOMED enrichment before the section closes.
  - ALL_CLOSE_TAGS is imported from medical_prompts (single source of truth).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Dict, Any, Tuple

from .base import VerifyMonitor
from ..utils.medical_verifier import (
    LocalVLLMClient,
    SnomedClient,
    VerifierConfig,
    MedicalPreprocessor,
)
from ..utils.medical_verifier_snomed import (
    MedicalReasoningVerifierSnomedFirst,
    SnomedFirstConfig,
)
from ..utils.medical_verifier_graph import (
    GraphAwareVerifier,
    GraphVerifierConfig,
)
from ..utils.medical_graph import SnomedRelationshipClient
from ..utils.medical_prompts import ALL_CLOSE_TAGS   # e.g. ("[/OBSERVATION]", "[/INFERENCE]", …)

logger = logging.getLogger(__name__)

MAX_CORRECTIONS_SENTINEL = (
    "\n[STOPPED: maximum correction attempts reached without a passing verification.]"
)


class MedicalMonitor(VerifyMonitor):
    """
    VerifyMonitor for the Medical Reasoning (MedReason) dataset.

    What happens at construction
    ----------------------------
    __init__ pre-processes the sample immediately:
      1. extract_case_facts  — converts the raw question/case text to compact JSON
      2. prefetch_snomed     — fetches SNOMED definitions for all option terms
    Both results are passed to the verifier and available for every
    verification call throughout generation, at zero per-cycle cost.

    Trigger (step_extractor)
    -------------------------
    Fires on two signals (both inside the <think> block only):
      - Any section close tag ([/OBSERVATION], [/INFERENCE], …):
        a complete structured section just finished → verify it.
      - "UNKNOWN:" in the chunk:
        the solver signalled uncertainty mid-section → verify immediately
        so SNOMED enrichment can be injected before the section closes.

    Verification (verify)
    ----------------------
    The verifier reads the section tag directly, routes to the appropriate
    check, and returns (passed, feedback). On failure, feedback is injected
    and the solver retries. On max_corrections, generation stops.

    Graph verification (run_graph_checks, opt-in)
    ----------------------------------------------
    When run_graph_checks=True, the verifier is GraphAwareVerifier instead of
    MedicalReasoningVerifierSnomedFirst. It runs every existing check
    unchanged, then ALSO maps entities to validated SNOMED CUIs, builds a
    running cross-section concept graph, and can downgrade an otherwise-
    passing [INFERENCE] or [OPTION_COMPARISON] section if it contradicts an
    earlier section's claim. Off by default because it depends on a
    Snowstorm relationship endpoint (see medical_graph.py) that was not
    validated against a live instance when this was written — if that
    endpoint is unreachable, graph checks degrade to a harmless no-op rather
    than failing closed, but you should confirm connectivity before relying
    on it to catch anything.
    """

    def __init__(
        self,
        name:            str,
        instance:        Dict[str, Any],
        line_interval:   int  = 5,        # kept for interface parity; not used
        max_corrections: int  = 5,
        verifier_port:   int  = 8001,
        verifier_model:  str  = "medverifier",
        run_snomed:      bool = True,
        run_graph_checks: bool = False,   # NEW — opt-in, see class docstring
        snowstorm_base_url: Optional[str] = None,  # NEW — passed to SnomedRelationshipClient
        think_open_tag:  str  = "<think>",
        think_close_tag: str  = "</think>",
        priority:        int  = 0,
    ) -> None:
        super().__init__(name=name, priority=priority)

        self.instance        = instance
        self.max_corrections = max_corrections
        self.think_open_tag  = think_open_tag
        self.think_close_tag = think_close_tag

        # --------------- verifier wiring ---------------
        self.vllm_client = LocalVLLMClient(
            base_url = f"http://localhost:{verifier_port}/v1",
            model    = verifier_model,
        )
        self.snomed_client: Optional[SnomedClient] = None
        if run_snomed:
            try:
                self.snomed_client = SnomedClient()
            except ValueError as e:
                logger.warning("[MedicalMonitor] SNOMED disabled: %s", e)

        # --------------- pre-processing ----------------
        question = instance.get("question", "")
        options  = instance.get("options", {})
        prep     = MedicalPreprocessor(self.vllm_client, self.snomed_client)

        logger.info("[MedicalMonitor] Extracting compact case facts...")
        compact_case = prep.extract_case_facts(question)

        logger.info("[MedicalMonitor] Pre-fetching SNOMED for option terms...")
        snomed_cache = prep.prefetch_snomed(question, options)

        if run_graph_checks:
            relationship_client = self._try_build_relationship_client(snowstorm_base_url)
            self.verifier = GraphAwareVerifier(
                vllm                = self.vllm_client,
                snomed              = self.snomed_client,
                config              = GraphVerifierConfig(run_snomed=run_snomed,
                                                            run_graph_checks=True),
                compact_case        = compact_case,
                snomed_cache        = snomed_cache,
                relationship_client = relationship_client,
            )
        else:
            self.verifier = MedicalReasoningVerifierSnomedFirst(
                vllm         = self.vllm_client,
                snomed       = self.snomed_client,
                config       = SnomedFirstConfig(run_snomed=run_snomed),
                compact_case = compact_case,
                snomed_cache = snomed_cache,
            )

    @staticmethod
    def _try_build_relationship_client(
        base_url: Optional[str],
    ) -> Optional[SnomedRelationshipClient]:
        try:
            return SnomedRelationshipClient(base_url=base_url)
        except Exception as e:
            logger.warning("[MedicalMonitor] SnomedRelationshipClient unavailable: %s", e)
            return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_in_think_block(self, generated_text: str) -> bool:
        # vLLM strips the opening <think> tag from the SSE stream for Qwen3
        # thinking models, so rfind('<think>') always returns -1. Infer instead:
        # if </think> hasn't appeared yet, we are still inside the think block.
        return self.think_close_tag not in generated_text

    def _count_feedback_blocks(self, text: str) -> int:
        return text.count("[FEEDBACK]")

    def _call_verifier(self, text: str) -> Tuple[bool, Optional[str]]:
        question = self.instance.get("question", "")
        options  = self.instance.get("options", {})
        return self.verifier.verify_trace(text, question=question, options=options)

    # ── step_extractor ────────────────────────────────────────────────────────

    def step_extractor(self, chunk: str, generated_text: str) -> bool:
        """
        Triggers verification on:
          1. Any section close tag in the chunk — a complete structured section
             just arrived; verify it.
          2. "UNKNOWN:" in the chunk — solver flagged uncertainty mid-section;
             trigger early so SNOMED enrichment can be injected.

        Only fires inside the <think> block.

        Returns a bare bool, matching VerifyMonitor's documented contract
        ("Returns: Boolean indicating if more steps should be generated").
        The original version of this method returned a (bool, str|None)
        2-tuple; that is dropped here because a non-empty tuple is always
        truthy in Python regardless of its first element, so any caller
        that does `if monitor.step_extractor(...):` would have triggered
        verification on every chunk rather than only on real triggers.
        """
        if not self._is_in_think_block(generated_text):
            return False

        # Primary trigger: section close tag
        for close_tag in ALL_CLOSE_TAGS:
            if close_tag in chunk:
                logger.debug(
                    "[MedicalMonitor.step_extractor] Close tag '%s' detected.", close_tag
                )
                return True

        # Secondary trigger: UNKNOWN uncertainty signal
        if "UNKNOWN:" in chunk:
            logger.debug("[MedicalMonitor.step_extractor] UNKNOWN signal detected.")
            return True

        return False


    # ── verify ────────────────────────────────────────────────────────────────

    async def verify(
        self,
        chunk:       str,
        token_index: int,
        event:       asyncio.Event,
        event_info:  Dict[str, Any],
    ) -> None:
        if event.is_set():
            return

        num_prior_corrections = self._count_feedback_blocks(chunk)
        passed, feedback      = self._call_verifier(chunk)

        if passed:
            logger.debug("[MedicalMonitor.verify] PASS at token_index=%d", token_index)
            return

        logger.info(
            "[MedicalMonitor.verify] FAIL at token_index=%d "
            "(prior=%d/%d): %s",
            token_index, num_prior_corrections, self.max_corrections, feedback,
        )

        if num_prior_corrections >= self.max_corrections:
            if not event.is_set():
                event_info.update({
                    "generated_text":   chunk,
                    "feedback":         MAX_CORRECTIONS_SENTINEL,
                    "correction_index": token_index,
                    "gave_up":          True,
                    "decision_log":     self.verifier.decision_log,
                })
                event.set()
            return

        feedback_text    = feedback or "The verifier rejected this reasoning. Please reconsider."
        wrapped_feedback = f"\n\n[FEEDBACK]\n{feedback_text}\n[/FEEDBACK]\n\n"

        if not event.is_set():
            event_info.update({
                "generated_text":   chunk,
                "feedback":         wrapped_feedback,
                "correction_index": token_index,
                "gave_up":          False,
                "decision_log":     self.verifier.decision_log,
            })
            event.set()

    # ── fix ───────────────────────────────────────────────────────────────────

    async def fix(
        self,
        generated_text: str,
        event_info:     Dict[str, Any],
        fix_method:     Optional[Any] = None,
    ) -> str:
        text_so_far = event_info.get("generated_text", generated_text)
        feedback    = event_info.get("feedback", "")

        if event_info.get("gave_up"):
            event_info["phase"] = "final_answer_correct"

        return text_so_far + feedback
