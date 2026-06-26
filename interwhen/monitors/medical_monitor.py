"""
Medical Reasoning Monitor
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

logger = logging.getLogger(__name__)

MAX_CORRECTIONS_SENTINEL = "\n[STOPPED: maximum correction attempts reached without a passing verification.]"


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
    Fires on two signals:
      - \\n\\n  : a paragraph just completed — verify the last paragraph
      - UNKNOWN : the solver flagged uncertainty mid-paragraph — verify immediately

    Verification (verify)
    ----------------------
    The verifier classifies each paragraph, routes to the appropriate check
    (observation grounding / inference validity / conclusion consistency),
    and returns (passed, feedback). On failure, feedback is injected and
    the solver retries. On max_corrections, generation stops.
    """

    def __init__(
        self,
        name:            str,
        instance:        Dict[str, Any],
        line_interval:   int   = 5,       # kept for interface parity; trigger is now \n\n
        max_corrections: int   = 5,
        verifier_port:   int   = 8001,
        verifier_model:  str   = "medverifier",
        run_snomed:      bool  = True,
        think_open_tag:  str   = "<think>",
        think_close_tag: str   = "</think>",
        priority:        int   = 0,
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

        self.verifier = MedicalReasoningVerifierSnomedFirst(
            vllm         = self.vllm_client,
            snomed       = self.snomed_client,
            config       = SnomedFirstConfig(run_snomed=run_snomed),
            compact_case = compact_case,
            snomed_cache = snomed_cache,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _is_in_think_block(self, generated_text: str) -> bool:
        last_open  = generated_text.rfind(self.think_open_tag)
        last_close = generated_text.rfind(self.think_close_tag)
        return last_open != -1 and last_open > last_close

    def _count_feedback_blocks(self, text: str) -> int:
        return text.count("[FEEDBACK]")

    def _call_verifier(self, text: str) -> Tuple[bool, Optional[str]]:
        question = self.instance.get("question", "")
        options  = self.instance.get("options", {})
        return self.verifier.verify_trace(text, question=question, options=options)

    # ── step_extractor ────────────────────────────────────────────────────────

    def step_extractor(self, chunk: str, generated_text: str) -> Tuple[bool, Optional[str]]:
        """
        Triggers on:
          - UNKNOWN in the chunk: solver signaled uncertainty mid-paragraph
          - \\n\\n  in the chunk: a paragraph just completed

        Both signals carry the full accumulated generated_text so the verifier
        can extract the relevant paragraph via _split_latest.
        """
        if not self._is_in_think_block(generated_text):
            return False, None

        if "UNKNOWN" in chunk:
            return True, generated_text

        if "\n\n" in chunk:
            return True, generated_text

        return False, None

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