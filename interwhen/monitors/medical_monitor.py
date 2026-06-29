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
    PubMedClient,
    VerifierConfig,
    MedicalPreprocessor,
)
from ..utils.medical_verifier_snomed import (
    MedicalReasoningVerifierSnomedFirst,
    SnomedFirstConfig,
)

logger = logging.getLogger(__name__)

MAX_CORRECTIONS_SENTINEL = (
    "\n[STOPPED: maximum verifier corrections reached for this sample.]\n"
)


class MedicalMonitor(VerifyMonitor):
    """
    VerifyMonitor for the Medical Reasoning (MedReason) dataset.

    Key params
    ----------
    evidence_source     : "pubmed" | "snomed" | "both" | "none"
    max_corrections     : feedback blocks before stopping (default 10)
    paragraph_interval  : trigger every N paragraphs (default 1 = every \\n\\n)
    verification_window : paragraphs sent per verification call (default 3)
    preprocess_case     : run case extraction LLM call before generation
    prefetch_snomed     : batch-fetch SNOMED for option terms before generation
    """

    def __init__(
        self,
        name:                str,
        instance:            Dict[str, Any],
        line_interval:       int   = 15,
        max_corrections:     int   = 10,
        verification_window: int   = 3,
        confidence_threshold:float = 0.9,
        verifier_port:       int   = 8001,
        verifier_model:      str   = "medverifier",
        evidence_source:     str   = "pubmed",
        run_snomed:          bool  = True,
        preprocess_case:     bool  = False,
        prefetch_snomed:     bool  = False,
        think_open_tag:      str   = "<think>",
        think_close_tag:     str   = "</think>",
        priority:            int   = 0,
    ) -> None:
        super().__init__(name=name, priority=priority)

        self.instance            = instance
        self.max_corrections     = max_corrections
        self.line_interval       = line_interval
        self.think_open_tag      = think_open_tag
        self.think_close_tag     = think_close_tag

        # line trigger counter — non-empty lines seen since last trigger
        self._line_count         = 0
        self._last_trigger_line  = 0

        # ── verifier LLM ──────────────────────────────────────────────────────
        self.vllm_client = LocalVLLMClient(
            base_url = f"http://localhost:{verifier_port}/v1",
            model    = verifier_model,
        )

        # ── SNOMED (optional) ─────────────────────────────────────────────────
        self.snomed_client: Optional[SnomedClient] = None
        if run_snomed and evidence_source in ("snomed", "both"):
            try:
                self.snomed_client = SnomedClient()
            except ValueError as e:
                logger.warning("[MedicalMonitor] SNOMED disabled: %s", e)

        # ── PubMed (optional) ─────────────────────────────────────────────────
        self.pubmed_client: Optional[PubMedClient] = None
        if evidence_source in ("pubmed", "both"):
            self.pubmed_client = PubMedClient()
            logger.info("[MedicalMonitor] PubMedClient initialised")

        # ── VerifierConfig ────────────────────────────────────────────────────
        config = SnomedFirstConfig(
            run_snomed              = run_snomed and evidence_source in ("snomed", "both"),
            evidence_source         = evidence_source,
            verification_window     = verification_window,
            max_feedback_per_sample = max_corrections,
            confidence_threshold    = confidence_threshold,
        )

        # ── Pre-processing (optional) ─────────────────────────────────────────
        question = instance.get("question", "")
        options  = instance.get("options", {})
        prep     = MedicalPreprocessor(self.vllm_client, self.snomed_client, self.pubmed_client)

        if preprocess_case:
            logger.info("[MedicalMonitor] Extracting compact case facts...")
            compact_case = prep.extract_case_facts(question)
        else:
            compact_case = ""

        if prefetch_snomed and self.snomed_client:
            logger.info("[MedicalMonitor] Pre-fetching SNOMED for option terms...")
            snomed_cache = prep.prefetch_snomed(question, options)
        else:
            snomed_cache = {}

        # ── Verifier ─────────────────────────────────────────────────────────
        self.verifier = MedicalReasoningVerifierSnomedFirst(
            vllm         = self.vllm_client,
            snomed       = self.snomed_client,
            pubmed       = self.pubmed_client,
            config       = config,
            compact_case = compact_case,
            snomed_cache = snomed_cache,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _is_in_think_block(self, generated_text: str) -> bool:
        return self.think_close_tag not in generated_text

    def _count_feedback_blocks(self, text: str) -> int:
        return text.count("[FEEDBACK]")

    def _call_verifier(self, text: str) -> Tuple[bool, Optional[str]]:
        try:
            question = self.instance.get("question", "")
            options  = self.instance.get("options", {})
            return self.verifier.verify_trace(text, question=question, options=options)
        except Exception as e:
            logger.warning("[MedicalMonitor] Verifier error — failing open: %s", e)
            return True, None

    # ── step_extractor ────────────────────────────────────────────────────────

    def step_extractor(self, chunk: str, generated_text: str) -> Tuple[bool, Optional[str]]:
        if not self._is_in_think_block(generated_text):
            return False, None

        if "UNKNOWN" in chunk:
            logger.info("[MONITOR] Trigger: UNKNOWN")
            print("\n  [MONITOR] Trigger → UNKNOWN")
            self._last_trigger_line = self._line_count
            return True, generated_text

        if "\n" in chunk:
            # Count newline characters directly — vLLM streams \n as its own
            # token so split("\n") always yields ["",""] giving 0 non-empty lines.
            self._line_count += chunk.count("\n")
            since = self._line_count - self._last_trigger_line
            if since >= self.line_interval:
                logger.info("[MONITOR] Trigger: %d lines accumulated", since)
                print(f"\n  [MONITOR] Trigger → {since} lines")
                self._last_trigger_line = self._line_count
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

        # Max corrections cap
        num_prior = self._count_feedback_blocks(chunk)
        if num_prior >= self.max_corrections:
            logger.info("[MONITOR] Max corrections (%d) reached — stopping", self.max_corrections)
            print(f"\n  [MONITOR] Max corrections ({self.max_corrections}) reached — stopping")
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

        passed, feedback = self._call_verifier(chunk)

        if passed:
            logger.debug("[MONITOR] PASS at token_index=%d", token_index)
            return

        logger.info("[MONITOR] FAIL at token_index=%d (feedback #%d)", token_index, num_prior + 1)
        print(f"\n  [MONITOR] FAIL — injecting feedback #{num_prior + 1}")

        feedback_text    = feedback or "Review this reasoning step before continuing."
        wrapped_feedback = f"\n\n[FEEDBACK]\n{feedback_text}\n[/FEEDBACK]\n\n"

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
