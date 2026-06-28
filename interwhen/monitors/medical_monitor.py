"""
Medical Reasoning Monitor — with confidence-gated verification.

Changes vs original
-------------------
* MedicalMonitor now accepts `conf_threshold` (float, default 0.75).
* step_extractor computes a paragraph-level confidence score before deciding
  whether to fire the verifier.
* Verifier is called ONLY when:
      confidence < conf_threshold   (model is uncertain)
      OR "UNKNOWN" is in the chunk  (model signalled explicit uncertainty)
* Logprob data can be fed into the monitor via push_logprob_chunk() if your
  stream runner exposes it; text-heuristic fallback is used otherwise.
* All decisions (confidence score, source, threshold, gate outcome) are written
  to the decision_log for post-hoc analysis.
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
from ..utils.confidence import (
    ParagraphConfidenceScorer,
    DEFAULT_LOGPROB_THRESHOLD,
    DEFAULT_HEURISTIC_THRESHOLD,
)

logger = logging.getLogger(__name__)


class MedicalMonitor(VerifyMonitor):
    """
    VerifyMonitor for the Medical Reasoning (MedReason) dataset.

    Trigger: fires on \\n\\n (paragraph complete) or UNKNOWN (solver uncertain),
             subject to a confidence gate — low-confidence paragraphs only.

    Confidence gate
    ---------------
    After each trigger, the monitor scores the new paragraph:
      - If logprob data has been pushed via push_logprob_chunk(), uses mean
        per-token probability (logprob-based).
      - Otherwise uses a text-heuristic scorer (hedging language, UNKNOWN,
        short length, uncommitted option comparisons).

    The verifier is called when:
      score < conf_threshold   OR   "UNKNOWN" in chunk

    This means high-confidence paragraphs are passed through without a
    verifier call, reducing both latency and false rejections.

    Parameters
    ----------
    conf_threshold : float
        Unified threshold used for both logprob and heuristic scores.
        Lower = call verifier less often (faster, more false positives).
        Higher = call verifier more often (safer, but risks over-rejection).
        Recommended starting point: 0.75.
        Tune up if accuracy is still below baseline; tune down if slow.

    All other parameters are identical to the original MedicalMonitor.
    """

    def __init__(
        self,
        name:             str,
        instance:         Dict[str, Any],
        line_interval:    int   = 5,
        max_corrections:  int   = 50,
        verifier_port:    int   = 8001,
        verifier_model:   str   = "medverifier",
        run_snomed:       bool  = True,
        preprocess_case:  bool  = False,
        prefetch_snomed:  bool  = False,
        think_open_tag:   str   = "<think>",
        think_close_tag:  str   = "</think>",
        priority:         int   = 0,
        # ── NEW ──────────────────────────────────────────────────
        conf_threshold:   float = 0.65,
    ) -> None:
        super().__init__(name=name, priority=priority)

        self.instance        = instance
        self.think_open_tag  = think_open_tag
        self.think_close_tag = think_close_tag
        self.conf_threshold  = conf_threshold

        # --------------- confidence scorer ---------------
        self._conf_scorer = ParagraphConfidenceScorer()

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

    # ── logprob feed (optional) ───────────────────────────────────────────────

    def push_logprob_chunk(self, token_logprob_dicts) -> None:
        """
        Feed per-token logprob data from the vLLM SSE stream into the confidence
        scorer.  Call this from your stream runner for each SSE chunk that
        carries logprob payloads.

        token_logprob_dicts: list of dicts in vLLM /v1/completions logprob format:
            [{"token": "...", "logprob": -0.23, "top_logprobs": {...}}, ...]
        """
        self._conf_scorer.push_logprob_chunk(token_logprob_dicts)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _is_in_think_block(self, generated_text: str) -> bool:
        return self.think_close_tag not in generated_text

    def _call_verifier(self, text: str) -> Tuple[bool, Optional[str]]:
        try:
            question = self.instance.get("question", "")
            options  = self.instance.get("options", {})
            return self.verifier.verify_trace(text, question=question, options=options)
        except Exception as e:
            logger.warning("[MedicalMonitor] Verifier error — failing open: %s", e)
            return True, None

    def _extract_new_paragraph(self, generated_text: str) -> str:
        """
        Extract the paragraph that just completed — mirrors the logic inside
        MedicalReasoningVerifier._split_latest so we score the same text that
        will be verified.
        """
        import re
        idx        = generated_text.rfind(self.think_open_tag)
        think_text = generated_text[idx + len(self.think_open_tag):] if idx != -1 else generated_text
        parts      = re.split(r"\[FEEDBACK\].*?\[/FEEDBACK\]", think_text, flags=re.DOTALL)
        since_fb   = parts[-1]
        paragraphs = [p.strip() for p in since_fb.split("\n\n") if p.strip()]
        return paragraphs[-1] if paragraphs else ""

    # ── confidence gate ───────────────────────────────────────────────────────

    def _should_verify(
        self,
        chunk:          str,
        generated_text: str,
        trigger_reason: str,
    ) -> Tuple[bool, float, str]:
        """
        Decide whether to call the verifier for the paragraph that just ended.

        Returns
        -------
        (should_verify, confidence_score, score_source)

        Logic
        -----
        1. UNKNOWN in chunk → always verify (explicit model uncertainty).
        2. Score the new paragraph.
        3. Verify iff score < conf_threshold.
        """
        # Hard-pass: UNKNOWN is an explicit uncertainty signal — always verify
        if "UNKNOWN" in chunk:
            # Drain the logprob buffer so it doesn't bleed into the next para
            score  = self._conf_scorer.score(self._extract_new_paragraph(generated_text))
            source = self._conf_scorer.last_source
            logger.info(
                "[CONFIDENCE] UNKNOWN trigger → forced verify | conf=%.4f src=%s",
                score, source,
            )
            print(f"  [CONFIDENCE] UNKNOWN → forced verify (conf={score:.4f}, src={source})")
            return True, score, source

        # Score the paragraph
        new_para = self._extract_new_paragraph(generated_text)
        score    = self._conf_scorer.score(new_para)
        source   = self._conf_scorer.last_source

        gate_open = score < self.conf_threshold

        logger.info(
            "[CONFIDENCE] trigger=%s | conf=%.4f | src=%s | threshold=%.4f | verify=%s",
            trigger_reason, score, source, self.conf_threshold, gate_open,
        )
        print(
            f"  [CONFIDENCE] {trigger_reason} | conf={score:.4f} "
            f"(src={source}) | τ={self.conf_threshold} | "
            f"→ {'VERIFY' if gate_open else 'SKIP'}"
        )
        return gate_open, score, source

    # ── step_extractor ────────────────────────────────────────────────────────

    def step_extractor(self, chunk: str, generated_text: str) -> Tuple[bool, Optional[str]]:
        """
        Called by interwhen on every streamed chunk.

        Original triggers (paragraph end, UNKNOWN) are preserved.
        Each trigger now runs through the confidence gate before returning True.
        """
        if not self._is_in_think_block(generated_text):
            return False, None

        trigger_reason: Optional[str] = None

        if "UNKNOWN" in chunk:
            trigger_reason = "UNKNOWN"
        elif "\n\n" in chunk:
            trigger_reason = "paragraph_end"

        if trigger_reason is None:
            return False, None

        logger.info("[MONITOR] Trigger: %s", trigger_reason)
        print(f"\n  [MONITOR] Trigger → {trigger_reason}")

        should_verify, conf, source = self._should_verify(chunk, generated_text, trigger_reason)

        if not should_verify:
            # Gate closed — log the skip and pass through
            if self.verifier.decision_log is not None:
                self.verifier.decision_log.append({
                    "type":              "GATED_SKIP",
                    "label":             "SKIP",
                    "paragraph_preview": self._extract_new_paragraph(generated_text)[:80],
                    "confidence":        conf,
                    "conf_source":       source,
                    "threshold":         self.conf_threshold,
                    "feedback":          None,
                })
            return False, None

        return True, generated_text

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

        # Attach confidence metadata to the verifier's last decision_log entry
        if self.verifier.decision_log:
            last = self.verifier.decision_log[-1]
            # _should_verify already ran and stored data; retrieve from
            # the GATED_SKIP logic — but here we are inside verify(), so
            # the gate was OPEN.  We annotate the verifier's entry instead.
            # (score is not re-computed here to avoid double-scoring;
            # it was already logged in step_extractor above.)
            last.setdefault("confidence", "see_step_extractor_log")

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