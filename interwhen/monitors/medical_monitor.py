"""
Medical Reasoning Monitor

A VerifyMonitor for the Medical Reasoning (MedReason) dataset integration.

What this monitor actually does (the "brain" loop)
----------------------------------------------------
1. step_extractor watches the streamed generation. While the model is inside
   a <think>...</think> block, every `line_interval` non-empty reasoning
   lines it hands the FULL generated-text-so-far to verify().

2. verify() sends that text to the verifier (today: a stub that always says
   "pass"; later: a real medical verifier). The verifier returns a
   pass/fail decision plus, on failure, a feedback string explaining what's
   wrong.

   - If the verifier says PASS: verify() returns without touching the
     asyncio.Event. Generation continues untouched.
   - If the verifier says FAIL: verify() sets the asyncio.Event. This is
     what actually stops the main LLM stream (stream_completion breaks out
     of the SSE loop as soon as the event is set, discarding any further
     tokens that were already in flight). event_info is populated with:
       - "generated_text": everything the model produced up to this point
       - "feedback": the verifier's own feedback string
     stream_completion then calls fix(), which appends the feedback to the
     generated text and re-prompts the model with that as prev_text — this
     is the actual "stop, show it what it built, show it the feedback,
     let it try again" cycle.

3. Giving up after K failed corrections. The monitor counts how many times
   feedback has already been injected (by counting "[FEEDBACK]" blocks in
   the text). If the verifier fails again after `max_corrections` feedback
   injections have already happened, the monitor does NOT inject yet another
   round of feedback and recurse forever. Instead it sets the event with a
   terminal sentinel feedback ("\\n[STOPPED: max corrections reached...]")
   and fix() returns the accumulated text with that sentinel appended,
   ending generation for good.

The verifier itself lives in medical_verifier.py
(MedicalReasoningVerifier.verify_trace), constructed below in __init__
from verifier_port/verifier_model/run_snomed. It judges the newest content
in the trace against the Observation/Inference/Evidence/Diagnosis/Plan
structure defined in medical_prompts.SYSTEM_PROMPT_MEDICAL, falling back
to a SNOMED CT lookup when it isn't confident before committing to a
final PASS/FAIL.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Dict, Any, Tuple

from .base import VerifyMonitor
from ..utils.medical_verifier import (
    LocalVLLMClient,
    SnomedClient,
    MedicalReasoningVerifier,
    VerifierConfig,
)

logger = logging.getLogger(__name__)

# Sentinel feedback text used when max_corrections is reached. fix() appends
# this to the generated text and stream_completion does not recurse again
# after that — see medical_example.py, which checks event_info["gave_up"]
# only for logging; the actual stop is just "the recursive call returns
# without setting the event again because generation has ended".
MAX_CORRECTIONS_SENTINEL = "\n[STOPPED: maximum correction attempts reached without a passing verification.]"


class MedicalMonitor(VerifyMonitor):
    """Monitor for the Medical Reasoning (MedReason) dataset.

    Args:
        name:            Monitor identifier.
        instance:        Problem dict for the current item (question, options,
                          answer, reasoning, id, etc. — see medical_helper.py).
        line_interval:   Trigger verification every N non-empty reasoning
                          lines generated inside the <think> block.
        max_corrections: Maximum number of feedback injections allowed before
                          the monitor gives up and stops generation for good.
        verifier_port:   Port of the LOCAL vLLM server running the verifier
                          model (separate from whatever server is serving the
                          solver).
        verifier_model:  --served-model-name of the verifier vLLM server.
        run_snomed:      Whether the verifier falls back to a SNOMED CT
                          lookup on UNKNOWN verdicts. Requires
                          BIOPORTAL_API_KEY in the environment; if that's
                          missing this is disabled with a warning rather
                          than raising, so a missing key never crashes the
                          monitor outright.
        think_open_tag:  Opening tag marking the start of the reasoning block.
        think_close_tag: Closing tag marking the end of the reasoning block.
        priority:        Monitor priority (unused by stream_completion today,
                          kept for interface parity with other monitors).
    """

    def __init__(
        self,
        name: str,
        instance: Dict[str, Any],
        line_interval: int = 5,
        max_corrections: int = 5,
        verifier_port: int = 8001,
        verifier_model: str = "medverifier",
        run_snomed: bool = True,
        think_open_tag: str = "<think>",
        think_close_tag: str = "</think>",
        priority: int = 0,
    ) -> None:
        super().__init__(name=name, priority=priority)

        self.instance = instance
        self.line_interval = line_interval
        self.max_corrections = max_corrections
        self.think_open_tag = think_open_tag
        self.think_close_tag = think_close_tag

        # --------------- verifier wiring ---------------
        self.vllm_client = LocalVLLMClient(
            base_url=f"http://localhost:{verifier_port}/v1",
            model=verifier_model,
        )
        self.snomed_client: Optional[SnomedClient] = None
        if run_snomed:
            try:
                self.snomed_client = SnomedClient()
            except ValueError as e:
                logger.warning("[MedicalMonitor] SNOMED disabled: %s", e)
        self.verifier = MedicalReasoningVerifier(
            vllm=self.vllm_client,
            snomed=self.snomed_client,
            config=VerifierConfig(run_snomed=run_snomed),
        )

        # --------------- internal state ---------------
        # Number of non-empty lines seen inside the think block, counted
        # cumulatively across the whole generation so far.
        self._line_count: int = 0
        # _line_count value at the last trigger, so we know when another
        # `line_interval` lines have accumulated.
        self._last_trigger_line_count: int = 0

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _is_in_think_block(self, generated_text: str) -> bool:
        """Return True if the most recent think tag is still open."""
        last_open = generated_text.rfind(self.think_open_tag)
        last_close = generated_text.rfind(self.think_close_tag)
        return last_open != -1 and last_open > last_close

    def _count_feedback_blocks(self, text: str) -> int:
        """Count how many times feedback has already been injected."""
        return text.count("[FEEDBACK]")

    def _call_verifier(self, text: str) -> Tuple[bool, Optional[str]]:
        """Send the accumulated reasoning to the verifier and get a verdict.

        Args:
            text: Full generated text so far (including the <think> block).

        Returns:
            (passed, feedback) where:
                passed:   True if the verifier accepts the reasoning so far.
                feedback: None if passed, else a string explaining what is
                          wrong and what the model should do about it.
        """
        question = self.instance.get("question", "")
        options = self.instance.get("options", {})
        return self.verifier.verify_trace(text, question=question, options=options)

    # ------------------------------------------------------------------
    # step_extractor
    # ------------------------------------------------------------------

    def step_extractor(self, chunk: str, generated_text: str) -> Tuple[bool, Optional[str]]:
        """Decide whether to fire the verifier for the current chunk.

        Counts non-empty lines added by ``chunk`` while the model is inside
        the <think>...</think> block. Every ``line_interval`` such lines,
        triggers a verification call.

        Args:
            chunk:          The most recently received token/text chunk.
            generated_text: The full generation so far (includes chunk).

        Returns:
            (trigger, text_to_verify) — text_to_verify is generated_text when
            trigger is True, else None.
        """
        if not self._is_in_think_block(generated_text):
            return False, None

        new_lines = sum(1 for line in chunk.split("\n") if line.strip())
        self._line_count += new_lines

        lines_since_last_trigger = self._line_count - self._last_trigger_line_count
        if lines_since_last_trigger >= self.line_interval:
            self._last_trigger_line_count = self._line_count
            logger.debug(
                "[MedicalMonitor.step_extractor] trigger: line_count=%d "
                "interval=%d lines_since_last=%d",
                self._line_count, self.line_interval, lines_since_last_trigger,
            )
            return True, generated_text

        return False, None

    # ------------------------------------------------------------------
    # verify — the actual "brain": ask the verifier, stop on failure
    # ------------------------------------------------------------------

    async def verify(
        self,
        chunk: str,
        token_index: int,
        event: asyncio.Event,
        event_info: Dict[str, Any],
    ) -> None:
        """Send the accumulated reasoning to the verifier; stop on failure.

        Args:
            chunk:       The full generated text at the time of triggering
                         (step_extractor passes generated_text here).
            token_index: Token/char position in the generation stream.
            event:       asyncio.Event. Setting this is what makes
                         stream_completion stop the in-flight LLM stream.
            event_info:  Dict consumed by fix(). On failure this is filled
                         with "generated_text" (what the model built so far)
                         and "feedback" (the verifier's own explanation).
        """
        if event.is_set():
            return  # another verify() call already triggered a stop

        # How many correction rounds have already happened on this trace.
        num_prior_corrections = self._count_feedback_blocks(chunk)

        passed, feedback = self._call_verifier(chunk)

        if passed:
            logger.debug(
                "[MedicalMonitor.verify] PASS at token_index=%d (line_count=%d)",
                token_index, self._line_count,
            )
            return  # nothing to do, let the model keep generating

        # ---- Verifier said FAIL ----
        logger.info(
            "[MedicalMonitor.verify] FAIL at token_index=%d "
            "(prior_corrections=%d/%d): %s",
            token_index, num_prior_corrections, self.max_corrections, feedback,
        )

        if num_prior_corrections >= self.max_corrections:
            # The model has already been given max_corrections chances and
            # still hasn't satisfied the verifier. Stop for good instead of
            # injecting yet another round of feedback.
            if not event.is_set():
                event_info["generated_text"] = chunk
                event_info["feedback"] = MAX_CORRECTIONS_SENTINEL
                event_info["correction_index"] = token_index
                event_info["gave_up"] = True
                event.set()
            return

        # Inject the verifier's feedback and let the model retry.
        feedback_text = feedback or "The verifier rejected this reasoning. Please reconsider."
        wrapped_feedback = f"\n\n[FEEDBACK]\n{feedback_text}\n[/FEEDBACK]\n\n"

        if not event.is_set():
            event_info["generated_text"] = chunk
            event_info["feedback"] = wrapped_feedback
            event_info["correction_index"] = token_index
            event_info["gave_up"] = False
            event.set()

    # ------------------------------------------------------------------
    # fix — stop the LLM, show it what it built + the feedback, let it retry
    # ------------------------------------------------------------------

    async def fix(
        self,
        generated_text: str,
        event_info: Dict[str, Any],
        fix_method: Optional[Any] = None,
    ) -> str:
        """Build the text the model resumes from after a failed verification.

        This is exactly "send what the LLM built until now along with the
        feedback the verifier gave": the text the model already generated,
        followed by the verifier's feedback. stream_completion then re-sends
        prompt + this text as the new prev_text and lets the model continue
        from there — UNLESS event_info["gave_up"] is True, in which case we
        also set event_info["phase"] = "final_answer_correct".

        That phase flag is not specific to "correct answers" — it is the one
        generic early-return switch stream_completion already checks
        (interject.py: `if stop_info.get("phase") == "final_answer_correct"`)
        to stop recursing and return immediately instead of calling the LLM
        again. We reuse it here purely as a stop signal so that hitting
        max_corrections actually ends generation, rather than silently
        falling through to stream_completion's unrelated hardcoded
        `num_calls_index >= 50` ceiling, which is not tied to this monitor's
        max_corrections at all.

        Args:
            generated_text: The full generated text at the time of the event
                             (fallback if event_info wasn't populated).
            event_info:      Dict populated by verify() with "generated_text",
                              "feedback", and "gave_up".
            fix_method:      Unused; present for interface compatibility.

        Returns:
            generated_text_so_far + feedback_text (whether that feedback is
            a retry prompt or the max-corrections give-up sentinel).
        """
        text_so_far = event_info.get("generated_text", generated_text)
        feedback = event_info.get("feedback", "")

        if event_info.get("gave_up"):
            # Signal stream_completion to stop recursing for good.
            event_info["phase"] = "final_answer_correct"

        return text_so_far + feedback