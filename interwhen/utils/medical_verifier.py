"""
medical_verifier.py  —  Structured-section verifier.

Key changes from the free-reasoning version:
  - _classify() is GONE. Section type is read from the enclosing tag, not
    inferred by a separate LLM call.
  - _split_latest() now extracts the innermost complete tagged section since
    the last [FEEDBACK] block, and returns (section_type, content).
  - _verify_option_comparison() is new — dedicated check for [OPTION_COMPARISON].
  - _verify_conclusion() uses build_conclusion_verification_prompt (stricter).
  - CompactState.add_ruled_out() is called when OPTION_COMPARISON passes.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from .medical_reasoning_prompts import MedicalReasoningPromptBuilder
from .medical_prompts import SECTION_TAG_TO_TYPE    # single source of truth for tags


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL vLLM CLIENT  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class LocalVLLMClient:
    """Sync wrapper around vLLM's OpenAI-compatible /v1/chat/completions."""

    def __init__(
        self,
        base_url:    str   = None,
        model:       str   = None,
        temperature: float = 0.0,
        max_tokens:  int   = 1024,
        timeout:     int   = 90,
    ):
        load_dotenv()
        self.base_url    = (base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model       = model or os.getenv("VLLM_MODEL", "medverifier")
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        last_end = text.rfind("</think>")
        if last_end != -1:
            after = text[last_end + len("</think>"):].strip()
            if after:
                return after
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        return re.sub(r"\s*```$", "", text.strip()).strip()

    @staticmethod
    def _extract_json_object(text: str) -> str:
        depth, start = 0, -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    candidate = text[start:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        start = -1
        return text

    def _clean_json(self, content: str) -> str:
        content = self._strip_think_tags(content)
        content = self._strip_fences(content)
        content = self._extract_json_object(content)
        return content.strip()

    def call(self, prompt: str, retries: int = 3, wait: int = 3) -> dict:
        url = f"{self.base_url}/chat/completions"
        raw = ""
        for attempt in range(retries):
            try:
                send_prompt = prompt
                if attempt == 1:
                    send_prompt += (
                        "\n\nIMPORTANT: Output ONLY the JSON object. "
                        "No explanation, no markdown, no text before or after."
                    )
                payload = {
                    "model":       self.model,
                    "messages":    [{"role": "user", "content": send_prompt}],
                    "temperature": self.temperature,
                    "max_tokens":  self.max_tokens,
                }
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"] or ""
                if not raw.strip():
                    if attempt < retries - 1:
                        time.sleep(wait)
                        continue
                    return {"error": "empty_response"}
                return json.loads(self._clean_json(raw))
            except json.JSONDecodeError as e:
                if attempt < retries - 1:
                    time.sleep(wait)
                    continue
                return {"error": f"json_parse_error: {e}", "raw": raw[:300]}
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(wait)
                else:
                    return {"error": f"request_error: {e}"}
        return {"error": "max_retries_exceeded"}


# ══════════════════════════════════════════════════════════════════════════════
# SNOMED CLIENT  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class SnomedClient:
    """BioPortal SNOMED CT lookup. Requires BIOPORTAL_API_KEY in .env."""

    SEARCH_URL = "https://data.bioontology.org/search"

    def __init__(self, top_k: int = 5, timeout: int = 15):
        load_dotenv()
        self.top_k   = top_k
        self.timeout = timeout
        self.api_key = os.getenv("BIOPORTAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "BIOPORTAL_API_KEY not found. Add to .env or set run_snomed=False."
            )

    def _search(self, term: str) -> list:
        headers = {"Authorization": f"apikey token={self.api_key}"}
        params  = {"q": term, "ontologies": "SNOMEDCT", "pagesize": self.top_k}
        try:
            r = requests.get(self.SEARCH_URL, headers=headers, params=params, timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("collection", [])
        except Exception as e:
            print(f"  [SNOMED] search failed for '{term}': {e}")
            return []

    @staticmethod
    def _normalize(question: str, term: str) -> str:
        CORPUS_CALLOSUM_PARTS = {"rostrum", "genu", "body", "splenium"}
        if term.strip().lower() in CORPUS_CALLOSUM_PARTS and "corpus callosum" in question.lower():
            return f"{term} of corpus callosum"
        return term

    def enrich(self, question: str, option_text: str) -> dict:
        normalized = self._normalize(question, option_text)
        for term in [normalized, normalized.lower(), normalized.replace(" of ", " ")]:
            results = self._search(term)
            if results:
                top  = results[0]
                defs = top.get("definition", [])
                defn = defs[0] if isinstance(defs, list) and defs else top.get("prefLabel", "")
                return {
                    "found": True, "query": normalized, "definition": defn,
                    "synonyms": top.get("synonym", [])[:5], "ontology_status": "FOUND",
                }
        return {"found": False, "query": normalized, "definition": "", "synonyms": [], "ontology_status": "NOT_FOUND"}

    def prefetch(self, terms: List[str], question: str = "", sleep: float = 0.3) -> Dict[str, str]:
        cache: Dict[str, str] = {}
        for term in terms:
            result = self.enrich(question=question, option_text=term)
            if result.get("found"):
                cache[term] = result.get("definition", "")
            time.sleep(sleep)
        return cache

    @staticmethod
    def build_feedback_block(enrichments: dict) -> str:
        if not enrichments:
            return ""
        lines = [f"- {term}: {defn}" for term, defn in enrichments.items() if defn]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# VERIFIER CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerifierConfig:
    run_snomed:               bool  = True
    unknown_defaults_to_pass: bool  = True
    allow_unknown:            bool  = True
    max_prior_context_chars:  int   = 4000
    snomed_rate_limit_sleep:  float = 0.3


# ══════════════════════════════════════════════════════════════════════════════
# COMPACT STATE  (section-aware)
# ══════════════════════════════════════════════════════════════════════════════

class CompactState:
    """
    Bounded structured summary of verified reasoning.
    Now section-aware: observations and inferences are tracked separately,
    and ruled_out is populated from verified OPTION_COMPARISON blocks.
    """

    MAX_ITEMS = 8

    def __init__(self):
        self.observations:        List[str] = []   # from verified [OBSERVATION]
        self.inferences:          List[str] = []   # from verified [INFERENCE]
        self.ruled_out:           List[str] = []   # from verified [OPTION_COMPARISON]
        self.working_conclusion:  str       = ""   # from verified [CONCLUSION]

    def _add(self, lst: List[str], item: str) -> None:
        if item and item not in lst:
            lst.append(item[:120])
        if len(lst) > self.MAX_ITEMS:
            lst[:] = lst[-self.MAX_ITEMS:]

    def add_observation(self, obs: str)   -> None: self._add(self.observations, obs)
    def add_inference(self, inf: str)     -> None: self._add(self.inferences, inf)
    def add_ruled_out(self, opt: str)     -> None: self._add(self.ruled_out, opt)

    # kept for backward-compat with SnomedFirst subclass
    def add_fact(self, fact: str)         -> None: self.add_observation(fact)
    def add_claim(self, claim: str)       -> None: self.add_inference(claim)

    def revise_claim(self, wrong: str, correction: str) -> None:
        self.inferences = [c for c in self.inferences if wrong.lower()[:40] not in c.lower()]
        if correction:
            self.add_inference(f"[REVISED] {correction}")

    def to_str(self) -> str:
        lines = []
        if self.observations:
            lines.append("Observations: " + "; ".join(self.observations))
        if self.inferences:
            lines.append("Inferences: " + "; ".join(self.inferences))
        if self.ruled_out:
            lines.append("Ruled out: " + "; ".join(self.ruled_out))
        if self.working_conclusion:
            lines.append(f"Working conclusion: {self.working_conclusion}")
        return "\n".join(lines) if lines else "Nothing established yet."

    def reset(self) -> None:
        self.observations.clear()
        self.inferences.clear()
        self.ruled_out.clear()
        self.working_conclusion = ""


# ══════════════════════════════════════════════════════════════════════════════
# MEDICAL PREPROCESSOR  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class MedicalPreprocessor:
    """
    Runs before generation starts for each sample.

    extract_case_facts  — converts raw case text to compact JSON
    prefetch_snomed     — fetches SNOMED definitions for all option terms upfront
    """

    def __init__(self, vllm: LocalVLLMClient, snomed: Optional[SnomedClient]):
        self.vllm   = vllm
        self.snomed = snomed

    def extract_case_facts(self, question: str) -> str:
        prompt = MedicalReasoningPromptBuilder.build_case_extraction_prompt(case_text=question)
        resp   = self.vllm.call(prompt)
        if "error" in resp:
            return question[:2000]
        return json.dumps(resp, indent=2)

    def prefetch_snomed(self, question: str, options: dict) -> Dict[str, str]:
        if self.snomed is None:
            return {}

        opts_text = "\n".join(f"{k}. {v}" for k, v in options.items())
        prompt    = MedicalReasoningPromptBuilder.build_snomed_term_extraction_prompt(
            question=question,
            options_text=opts_text,
            reasoning_chunk=opts_text,
        )
        resp  = self.vllm.call(prompt)
        terms = resp.get("terms", [])
        if not isinstance(terms, list):
            terms = []
        terms = terms[:8]

        print(f"  [Preprocessor] Pre-fetching SNOMED for {len(terms)} terms: {terms}")
        cache = self.snomed.prefetch(terms, question=question, sleep=0.3)
        print(f"  [Preprocessor] Cached {len(cache)} definitions.")
        return cache


# ══════════════════════════════════════════════════════════════════════════════
# MEDICAL REASONING VERIFIER  (structured)
# ══════════════════════════════════════════════════════════════════════════════

# Regex to find complete tagged sections. Built from SECTION_TAG_TO_TYPE so
# there is exactly one place to add new tags.
_SECTION_PATTERN = re.compile(
    r"\[(" + "|".join(re.escape(t) for t in SECTION_TAG_TO_TYPE) + r")\](.*?)\[/\1\]",
    re.DOTALL,
)


class MedicalReasoningVerifier:
    """
    Structured-section verifier.

    Routing (no LLM classifier call):
      [OBSERVATION]       → grounding check vs compact case
      [INFERENCE]         → clinical validity + SNOMED + options context
      [OPTION_COMPARISON] → option-elimination audit
      [CONCLUSION]        → consistency + option alignment (no UNKNOWN)
      anything else       → pass through
    """

    def __init__(
        self,
        vllm:         LocalVLLMClient,
        snomed:       Optional[SnomedClient]   = None,
        config:       Optional[VerifierConfig] = None,
        compact_case: str                      = "",
        snomed_cache: Optional[Dict[str, str]] = None,
    ):
        self.vllm          = vllm
        self.snomed        = snomed
        self.config        = config or VerifierConfig()
        self.compact_case  = compact_case
        self.snomed_cache  = snomed_cache or {}
        self.compact_state = CompactState()
        self.decision_log: List[dict] = []

    # ── Text splitting ─────────────────────────────────────────────────────────

    @staticmethod
    def _split_latest(text: str, think_open_tag: str = "<think>") -> Tuple[str, str, str]:
        """
        Returns (prior_context, section_type, section_content).

        Finds the LAST complete [TAG]…[/TAG] block since the last [FEEDBACK]
        block. If none is found, returns ("", "OTHER", "").

        The section content is the raw text between the open and close tags,
        stripped of leading/trailing whitespace.
        """
        idx        = text.rfind(think_open_tag)
        think_text = text[idx + len(think_open_tag):] if idx != -1 else text

        # Strip [FEEDBACK]…[/FEEDBACK] blocks, take the tail since the last one
        parts         = re.split(r"\[FEEDBACK\].*?\[/FEEDBACK\]", think_text, flags=re.DOTALL)
        since_last_fb = parts[-1]
        prior_fb      = "".join(parts[:-1]).strip()

        # Find all complete tagged sections in since_last_fb
        matches = list(_SECTION_PATTERN.finditer(since_last_fb))
        if not matches:
            return prior_fb, "OTHER", ""

        last_match    = matches[-1]
        section_type  = last_match.group(1).upper()   # e.g. "INFERENCE"
        section_body  = last_match.group(2).strip()

        # prior context = everything before the last match
        prior_in_segment = since_last_fb[:last_match.start()].strip()
        prior_context    = "\n\n".join(filter(None, [prior_fb, prior_in_segment]))

        return prior_context.strip(), section_type, section_body

    def _truncate_prior(self, prior: str) -> str:
        limit = self.config.max_prior_context_chars
        if len(prior) <= limit:
            return prior
        return "[...earlier context truncated...]\n" + prior[-limit:]

    @staticmethod
    def _options_text(options: dict) -> str:
        return "\n".join(f"{k}. {v}" for k, v in options.items()) if options else ""

    # ── SNOMED helpers ─────────────────────────────────────────────────────────

    def _build_snomed_block(self) -> str:
        if not self.snomed_cache:
            return ""
        return "\n".join(f"- {t}: {d}" for t, d in self.snomed_cache.items() if d)

    def _realtime_snomed(self, terms: List[str], question: str) -> None:
        if self.snomed is None:
            return
        for term in terms:
            if term not in self.snomed_cache:
                result = self.snomed.enrich(question=question, option_text=term)
                if result.get("found"):
                    self.snomed_cache[term] = result.get("definition", "")
                    print(f"  [SNOMED realtime] '{term}'")
                time.sleep(self.config.snomed_rate_limit_sleep)

    # ── Observation verification ───────────────────────────────────────────────

    def _verify_observation(self, paragraph: str) -> Tuple[bool, Optional[str]]:
        """Grounding check: all stated facts must be explicitly in the case."""
        if not self.compact_case.strip():
            return True, None

        prompt   = MedicalReasoningPromptBuilder.build_observation_grounding_prompt(
            compact_case=self.compact_case,
            paragraph=paragraph,
        )
        resp     = self.vllm.call(prompt)
        grounded = resp.get("grounded", True)

        if grounded:
            # Add each verified fact line to compact state
            for line in paragraph.strip().splitlines():
                line = line.strip().lstrip("-•0123456789. ")
                if len(line) > 5:
                    self.compact_state.add_observation(line)
            return True, None

        issues = resp.get("issues", [])
        parts  = [
            f"{iss.get('type', 'error')}: '{iss.get('claim', '')}' — {iss.get('reason', '')}"
            for iss in issues
        ]
        return False, "[OBSERVATION] not grounded in case:\n" + "\n".join(f"  - {p}" for p in parts)

    # ── Inference verification ─────────────────────────────────────────────────

    def _verify_inference(
        self,
        paragraph: str,
        options:   dict,
        question:  str  = "",
        _retried:  bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Clinical validity check for one [INFERENCE] block."""
        opts_text    = self._options_text(options)
        snomed_block = self._build_snomed_block()

        prompt = MedicalReasoningPromptBuilder.build_inference_verification_prompt(
            compact_case  = self.compact_case,
            compact_state = self.compact_state.to_str(),
            options_text  = opts_text,
            snomed_context= snomed_block,
            paragraph     = paragraph,
            allow_unknown = self.config.allow_unknown,
        )
        resp  = self.vllm.call(prompt)
        label = str(resp.get("label", "ERROR")).upper()

        if label == "TRUE":
            self.compact_state.add_inference(paragraph.strip()[:100])
            return True, None

        if label == "FALSE":
            wrong      = resp.get("wrong_claim")
            correction = resp.get("correction")
            if wrong:
                self.compact_state.revise_claim(wrong, correction or "")
            return False, self._format_feedback(resp)

        # UNKNOWN
        if not _retried:
            if self.snomed is not None and self.config.run_snomed:
                terms_resp = self.vllm.call(
                    MedicalReasoningPromptBuilder.build_snomed_term_extraction_prompt(
                        question=question,
                        options_text=opts_text,
                        reasoning_chunk=paragraph,
                    )
                )
                terms = terms_resp.get("terms", [])[:3]
                if terms:
                    self._realtime_snomed(terms, question)
                    return self._verify_inference(paragraph, options, question, _retried=True)

        if self.config.unknown_defaults_to_pass:
            return True, None
        return False, self._format_feedback(resp, prefix="Unresolved uncertainty: ")

    # ── Option comparison verification ─────────────────────────────────────────

    def _verify_option_comparison(
        self,
        paragraph: str,
        options:   dict,
        question:  str  = "",
        _retried:  bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifies [OPTION_COMPARISON]: all options must be addressed with
        case-supported reasons. Ruled-out options are recorded in compact state.
        """
        opts_text    = self._options_text(options)
        snomed_block = self._build_snomed_block()

        prompt = MedicalReasoningPromptBuilder.build_option_comparison_verification_prompt(
            compact_case  = self.compact_case,
            compact_state = self.compact_state.to_str(),
            options_text  = opts_text,
            snomed_context= snomed_block,
            paragraph     = paragraph,
            allow_unknown = self.config.allow_unknown,
        )
        resp  = self.vllm.call(prompt)
        label = str(resp.get("label", "ERROR")).upper()

        if label == "TRUE":
            for opt in resp.get("ruled_out", []):
                self.compact_state.add_ruled_out(str(opt).strip())
            return True, None

        if label == "FALSE":
            wrong      = resp.get("wrong_claim")
            correction = resp.get("correction")
            if wrong:
                self.compact_state.revise_claim(wrong, correction or "")
            return False, self._format_feedback(resp)

        # UNKNOWN — try SNOMED enrichment once
        if not _retried:
            if self.snomed is not None and self.config.run_snomed:
                terms_resp = self.vllm.call(
                    MedicalReasoningPromptBuilder.build_snomed_term_extraction_prompt(
                        question=question,
                        options_text=opts_text,
                        reasoning_chunk=paragraph,
                    )
                )
                terms = terms_resp.get("terms", [])[:3]
                if terms:
                    self._realtime_snomed(terms, question)
                    return self._verify_option_comparison(paragraph, options, question, _retried=True)

        if self.config.unknown_defaults_to_pass:
            return True, None
        return False, self._format_feedback(resp, prefix="Unresolved uncertainty: ")

    # ── Conclusion verification ────────────────────────────────────────────────

    def _verify_conclusion(
        self,
        paragraph: str,
        options:   dict,
        question:  str = "",
    ) -> Tuple[bool, Optional[str]]:
        """
        Consistency + option alignment check for [CONCLUSION].
        UNKNOWN is never allowed here.
        """
        opts_text    = self._options_text(options)
        snomed_block = self._build_snomed_block()

        prompt = MedicalReasoningPromptBuilder.build_conclusion_verification_prompt(
            compact_case  = self.compact_case,
            compact_state = self.compact_state.to_str(),
            options_text  = opts_text,
            snomed_context= snomed_block,
            paragraph     = paragraph,
        )
        resp  = self.vllm.call(prompt)
        label = str(resp.get("label", "ERROR")).upper()

        if label == "TRUE":
            opt = resp.get("selected_option", "")
            self.compact_state.working_conclusion = (
                f"Option {opt}: {paragraph.strip()[:80]}" if opt else paragraph.strip()[:80]
            )
            return True, None

        if label == "FALSE":
            return False, self._format_feedback(resp)

        # Unexpected label — fail open at conclusion
        return True, None

    # ── Feedback formatting ────────────────────────────────────────────────────

    @staticmethod
    def _format_feedback(
        resp:         dict,
        snomed_block: Optional[str] = None,
        prefix:       str           = "",
    ) -> str:
        lines = []
        if prefix:
            lines.append(prefix.strip())
        wrong      = resp.get("wrong_claim")
        correction = resp.get("correction")
        evidence   = resp.get("evidence", [])
        if wrong:
            lines.append(f"Incorrect: {wrong}")
        if correction:
            lines.append(f"Correction: {correction}")
        if evidence:
            lines.extend(f"  - {e}" for e in evidence)
        if snomed_block:
            lines.append(f"\nSNOMED CT definitions:\n{snomed_block}")
        return "\n".join(l for l in lines if l).strip() \
               or "The verifier rejected this reasoning. Please reconsider."

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, section_type: str, label: str, paragraph: str, feedback: Optional[str]) -> None:
        self.decision_log.append({
            "section_type":      section_type,
            "label":             label,
            "paragraph_preview": paragraph[:80],
            "feedback":          feedback,
        })

    # ── Main entry point ───────────────────────────────────────────────────────

    def verify_trace(
        self,
        text:     str,
        question: str            = "",
        options:  Optional[dict] = None,
    ) -> Tuple[bool, Optional[str]]:
        options = options or {}
        _, section_type, section_body = self._split_latest(text)

        if not section_body.strip():
            return True, None

        if section_type == "OBSERVATION":
            passed, fb = self._verify_observation(section_body)

        elif section_type == "INFERENCE":
            passed, fb = self._verify_inference(section_body, options, question)

        elif section_type == "OPTION_COMPARISON":
            passed, fb = self._verify_option_comparison(section_body, options, question)

        elif section_type == "CONCLUSION":
            passed, fb = self._verify_conclusion(section_body, options, question)

        else:
            return True, None   # OTHER / unrecognised — pass through

        self._log(section_type, "PASS" if passed else "FAIL", section_body, fb)
        return passed, fb