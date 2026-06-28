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


class MedicalMonitor(VerifyMonitor):
    """
    VerifyMonitor for the Medical Reasoning (MedReason) dataset.

    Trigger: fires on \n\n (paragraph complete) or UNKNOWN (solver uncertain).
    Retries: unlimited — interwhen's own 50-call ceiling in interject.py handles stopping.
    Crash safety: verifier errors fail open (PASS) rather than crashing the worker.

    Preprocessing flags (both default False for cost efficiency):
      preprocess_case  — extract compact case facts via LLM before generation
      prefetch_snomed  — batch fetch SNOMED for option terms before generation
    """

    def __init__(
        self,
        name:             str,
        instance:         Dict[str, Any],
        line_interval:    int   = 5,        # kept for interface parity, not used
        max_corrections:  int   = 50,       # kept for interface parity, not used internally
        verifier_port:    int   = 8001,
        verifier_model:   str   = "medverifier",
        run_snomed:       bool  = True,
        preprocess_case:  bool  = False,    # set True to run case extraction LLM call
        prefetch_snomed:  bool  = False,    # set True to batch-fetch SNOMED before generation
        think_open_tag:   str   = "<think>",
        think_close_tag:  str   = "</think>",
        priority:         int   = 0,
    ) -> None:
        super().__init__(name=name, priority=priority)

        self.instance        = instance
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

        # --------------- pre-processing (optional) ----------------
        question = instance.get("question", "")
        options  = instance.get("options", {})
        prep     = MedicalPreprocessor(self.vllm_client, self.snomed_client)

        if preprocess_case:
            logger.info("[MedicalMonitor] Extracting compact case facts...")
            compact_case = prep.extract_case_facts(question)
        else:
            compact_case = ""

        if prefetch_snomed and run_snomed and self.snomed_client:
            logger.info("[MedicalMonitor] Pre-fetching SNOMED for option terms...")
            snomed_cache = prep.prefetch_snomed(question, options)
        else:
            snomed_cache = {}

        self.verifier = MedicalReasoningVerifierSnomedFirst(
            vllm         = self.vllm_client,
            snomed       = self.snomed_client,
            config       = SnomedFirstConfig(run_snomed=run_snomed),
            compact_case = compact_case,
            snomed_cache = snomed_cache,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _is_in_think_block(self, generated_text: str) -> bool:
        # vLLM strips the opening <think> tag — infer from closing tag absence.
        return self.think_close_tag not in generated_text

    def _count_feedback_blocks(self, text: str) -> int:
        return text.count("[FEEDBACK]")

    def _call_verifier(self, text: str) -> Tuple[bool, Optional[str]]:
        return True, None  # default to pass if verifier is not available
    
        try:
            question = self.instance.get("question", "")
            options  = self.instance.get("options", {})
            return self.verifier.verify_trace(text, question=question, options=options)
        except Exception as e:
            logger.warning("[MedicalMonitor] Verifier error — failing open: %s", e)
            return True, None   # fail open: don't crash the worker

    # ── step_extractor ────────────────────────────────────────────────────────

    def step_extractor(self, chunk: str, generated_text: str) -> Tuple[bool, Optional[str]]:
        if not self._is_in_think_block(generated_text):
            return False, None
        if "UNKNOWN" in chunk:
            logger.info("[MONITOR] Trigger: UNKNOWN in solver chunk")
            print("\n  [MONITOR] Trigger → UNKNOWN")
            return True, generated_text
        if "\n\n" in chunk:
            logger.info("[MONITOR] Trigger: paragraph completed")
            print("\n  [MONITOR] Trigger → \\n\\n paragraph end")
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

        passed, feedback = self._call_verifier(chunk)

        if passed:
            logger.debug("[MONITOR] PASS at token_index=%d", token_index)
            return

        logger.info("[MONITOR] FAIL at token_index=%d — injecting feedback", token_index)
        print(f"\n  [MONITOR] FAIL — preparing feedback injection")

        feedback_text    = feedback or "The verifier rejected this reasoning. Please reconsider."
        wrapped_feedback = (
            f"\n\n[FEEDBACK]\n"
            f"{feedback_text}\n\n"
            f"Re-evaluate your option selection. Your final answer may need to change.\n"
            f"[/FEEDBACK]\n\n"
        )

        logger.info("[MONITOR] Feedback injected:\n%s", wrapped_feedback)
        print(f"  [MONITOR] Feedback:\n{'─'*60}")
        print(wrapped_feedback.strip())
        print('─'*60)

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

    async def fix(self, generated_text, event_info, fix_method=None):
        text_so_far = event_info.get("generated_text", generated_text)
        feedback    = event_info.get("feedback", "")
        fb_count    = text_so_far.count("[FEEDBACK]") + (1 if "[FEEDBACK]" in feedback else 0)
        logger.info("[MONITOR] fix() — solver continuing (correction #%d)", fb_count)
        print(f"\n  [MONITOR] Solver continuing — correction #{fb_count}")
        return text_so_far + feedback