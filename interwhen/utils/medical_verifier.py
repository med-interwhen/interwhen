"""
medical_verifier.py
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


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL vLLM CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class LocalVLLMClient:
    """Sync wrapper around vLLM's OpenAI-compatible /v1/chat/completions.
    Separate server/port from the solver.
    """

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
# SNOMED CLIENT
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
        """Batch fetch definitions for a list of terms. Returns {term: definition}."""
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
    run_snomed:              bool  = True
    unknown_defaults_to_pass: bool = True
    allow_unknown:           bool  = True
    max_prior_context_chars: int   = 4000
    snomed_rate_limit_sleep: float = 0.3


# ══════════════════════════════════════════════════════════════════════════════
# COMPACT STATE
# ══════════════════════════════════════════════════════════════════════════════

class CompactState:
    """
    Bounded structured summary of verified reasoning content.
    Replaces raw text accumulation — avoids context length blowup
    while preserving all established facts and claims.
    """

    MAX_ITEMS = 8

    def __init__(self):
        self.facts:               List[str] = []
        self.claims:              List[str] = []
        self.ruled_out:           List[str] = []
        self.working_conclusion:  str       = ""

    def _add(self, lst: List[str], item: str) -> None:
        if item and item not in lst:
            lst.append(item[:120])
        if len(lst) > self.MAX_ITEMS:
            lst[:] = lst[-self.MAX_ITEMS:]

    def add_fact(self, fact: str)       -> None: self._add(self.facts, fact)
    def add_claim(self, claim: str)     -> None: self._add(self.claims, claim)
    def add_ruled_out(self, opt: str)   -> None: self._add(self.ruled_out, opt)

    def revise_claim(self, wrong: str, correction: str) -> None:
        self.claims = [c for c in self.claims if wrong.lower()[:40] not in c.lower()]
        if correction:
            self.add_claim(f"[REVISED] {correction}")

    def to_str(self) -> str:
        lines = []
        if self.facts:             lines.append("Facts: "              + "; ".join(self.facts))
        if self.claims:            lines.append("Claims: "             + "; ".join(self.claims))
        if self.ruled_out:         lines.append("Ruled out: "          + "; ".join(self.ruled_out))
        if self.working_conclusion: lines.append(f"Working conclusion: {self.working_conclusion}")
        return "\n".join(lines) if lines else "Nothing established yet."

    def reset(self) -> None:
        self.facts.clear(); self.claims.clear()
        self.ruled_out.clear(); self.working_conclusion = ""


# ══════════════════════════════════════════════════════════════════════════════
# MEDICAL PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class MedicalPreprocessor:
    """
    Runs before generation starts for each sample.

    extract_case_facts  — converts raw case text to compact JSON (avoids
                          re-sending long vignettes on every verification call)
    prefetch_snomed     — fetches SNOMED definitions for all option terms upfront
                          (zero per-cycle SNOMED cost for known concepts)
    """

    def __init__(self, vllm: LocalVLLMClient, snomed: Optional[SnomedClient]):
        self.vllm   = vllm
        self.snomed = snomed

    def extract_case_facts(self, question: str) -> str:
        """Convert raw case text to compact structured JSON string."""
        prompt = MedicalReasoningPromptBuilder.build_case_extraction_prompt(case_text=question)
        resp   = self.vllm.call(prompt)
        if "error" in resp:
            # Fallback: use truncated raw text rather than fail
            return question[:2000]
        return json.dumps(resp, indent=2)

    def prefetch_snomed(self, question: str, options: dict) -> Dict[str, str]:
        """Pre-fetch SNOMED definitions for option terms + key question terms."""
        if self.snomed is None:
            return {}

        opts_text = "\n".join(f"{k}. {v}" for k, v in options.items())
        prompt    = MedicalReasoningPromptBuilder.build_snomed_term_extraction_prompt(
            question=question,
            options_text=opts_text,
            reasoning_chunk=opts_text,   # extract from option texts themselves
        )
        resp  = self.vllm.call(prompt)
        terms = resp.get("terms", [])
        if not isinstance(terms, list):
            terms = []
        terms = terms[:8]  # cap prefetch at 8 terms

        print(f"  [Preprocessor] Pre-fetching SNOMED for {len(terms)} terms: {terms}")
        cache = self.snomed.prefetch(terms, question=question, sleep=self.snomed.top_k * 0.0 + 0.3)
        print(f"  [Preprocessor] Cached {len(cache)} definitions.")
        return cache


# ══════════════════════════════════════════════════════════════════════════════
# MEDICAL REASONING VERIFIER
# ══════════════════════════════════════════════════════════════════════════════

class MedicalReasoningVerifier:
    """
    Classify-then-verify verifier.

    Per paragraph trigger:
      1. Split out the new paragraph (last complete unit since the last feedback block)
      2. Classify: OBSERVATION / INFERENCE / OPTION_COMPARISON / CONCLUSION / OTHER
      3. Route:
           OBSERVATION      → grounding check against compact case facts
           INFERENCE /
           OPTION_COMPARISON → clinical validity check + options context + SNOMED cache
           CONCLUSION       → same as inference but allow_unknown=False
           OTHER            → pass through (transitions, revisions, etc.)
      4. Update compact state on success
      5. On UNKNOWN: fetch real-time SNOMED for the specific uncertain terms, retry once
    """

    def __init__(
        self,
        vllm:           LocalVLLMClient,
        snomed:         Optional[SnomedClient]  = None,
        config:         Optional[VerifierConfig] = None,
        compact_case:   str                      = "",
        snomed_cache:   Optional[Dict[str, str]] = None,
        think_open_tag: str                      = "<think>",
    ):
        self.vllm           = vllm
        self.snomed         = snomed
        self.config         = config or VerifierConfig()
        self.compact_case   = compact_case
        self.snomed_cache   = snomed_cache or {}
        self.compact_state  = CompactState()
        self.think_open_tag = think_open_tag
        self.decision_log: List[dict] = []

    # ── Text splitting ─────────────────────────────────────────────────────────

    @staticmethod
    def _split_latest(text: str, think_open_tag: str = "<think>") -> Tuple[str, str]:
        """
        Returns (prior_context, new_paragraph).
        Extracts the last complete paragraph (by \n\n) since the last
        [FEEDBACK] block. Each trigger delivers exactly one new paragraph
        so the verifier judges one logical unit at a time.
        """
        idx         = text.rfind(think_open_tag)
        think_text  = text[idx + len(think_open_tag):] if idx != -1 else text

        parts         = re.split(r"\[FEEDBACK\].*?\[/FEEDBACK\]", think_text, flags=re.DOTALL)
        since_last_fb = parts[-1]
        prior_fb      = "".join(parts[:-1]).strip()

        paragraphs = [p.strip() for p in since_last_fb.split("\n\n") if p.strip()]

        if not paragraphs:
            return prior_fb, ""

        new_paragraph   = paragraphs[-1]
        prior_paragraphs = "\n\n".join(paragraphs[:-1])
        prior_context   = "\n\n".join(filter(None, [prior_fb, prior_paragraphs]))
        return prior_context.strip(), new_paragraph

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
        """Fetch SNOMED for terms not already in cache; updates self.snomed_cache."""
        if self.snomed is None:
            return
        for term in terms:
            if term not in self.snomed_cache:
                result = self.snomed.enrich(question=question, option_text=term)
                if result.get("found"):
                    self.snomed_cache[term] = result.get("definition", "")
                    print(f"  [SNOMED realtime] '{term}'")
                time.sleep(self.config.snomed_rate_limit_sleep)

    # ── Classification ─────────────────────────────────────────────────────────

    def _classify(self, paragraph: str) -> str:
        prompt = MedicalReasoningPromptBuilder.build_paragraph_classifier_prompt(
            paragraph=paragraph
        )
        resp = self.vllm.call(prompt)
        return str(resp.get("class", "OTHER")).upper()

    # ── Observation verification ───────────────────────────────────────────────

    def _verify_observation(self, paragraph: str) -> Tuple[bool, Optional[str]]:
        """Grounding check: are all stated facts actually in the case?"""
        if not self.compact_case.strip():
            # No compact case available — can't ground check, pass through
            return True, None

        prompt = MedicalReasoningPromptBuilder.build_observation_grounding_prompt(
            compact_case=self.compact_case,
            paragraph=paragraph,
        )
        resp     = self.vllm.call(prompt)
        grounded = resp.get("grounded", True)

        if grounded:
            # Add verified facts to compact state
            for line in paragraph.strip().splitlines():
                line = line.strip().lstrip("-•0123456789. ")
                if len(line) > 5:
                    self.compact_state.add_fact(line)
            return True, None

        issues = resp.get("issues", [])
        parts  = [
            f"{iss.get('type', 'error')}: '{iss.get('claim', '')}' — {iss.get('reason', '')}"
            for iss in issues
        ]
        return False, "Observation not grounded in case:\n" + "\n".join(f"  - {p}" for p in parts)

    # ── Inference / conclusion verification ────────────────────────────────────

    def _verify_inference(
        self,
        paragraph:  str,
        options:    dict,
        question:   str  = "",
        _retried:   bool = False,    # prevents infinite recursion on UNKNOWN
    ) -> Tuple[bool, Optional[str]]:
        """Clinical validity check with options context and SNOMED cache."""
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
            self.compact_state.add_claim(paragraph.strip()[:100])
            return True, None

        if label == "FALSE":
            wrong      = resp.get("wrong_claim")
            correction = resp.get("correction")
            if wrong:
                self.compact_state.revise_claim(wrong, correction or "")
            return False, self._format_feedback(resp)

        if label == "UNKNOWN" and not _retried:
            if self.snomed is not None and self.config.run_snomed:
                terms_resp = self.vllm.call(
                    MedicalReasoningPromptBuilder.build_snomed_term_extraction_prompt(
                        question       = question,
                        options_text   = opts_text,
                        reasoning_chunk= paragraph,
                    )
                )
                terms = terms_resp.get("terms", [])[:3]
                if terms:
                    self._realtime_snomed(terms, question)
                    return self._verify_inference(paragraph, options, question, _retried=True)

        # UNKNOWN (no SNOMED, terms empty, or after retry) — force solver to commit.
        # Passing ambiguous reasoning silently was the main accuracy leak:
        # unknown_defaults_to_pass=True meant the verifier only caught
        # *confidently* wrong inferences, not uncertain ones.
        evidence = resp.get("evidence", [])
        hint     = evidence[0] if evidence else "insufficient reasoning detail"
        return False, self._format_feedback(
            resp,
            prefix=(
                "Your reasoning is ambiguous here — the verifier could not confirm "
                f"or refute this inference ({hint}). "
                "Make your clinical reasoning explicit: state which specific finding "
                "leads to which conclusion and why the other options are less likely."
            ),
        )
    def _verify_conclusion(
        self,
        paragraph: str,
        options:   dict,
        question:  str = "",
    ) -> Tuple[bool, Optional[str]]:
        """Consistency + option alignment check. No UNKNOWN allowed at conclusion."""
        opts_text    = self._options_text(options)
        snomed_block = self._build_snomed_block()

        prompt = MedicalReasoningPromptBuilder.build_inference_verification_prompt(
            compact_case  = self.compact_case,
            compact_state = self.compact_state.to_str(),
            options_text  = opts_text,
            snomed_context= snomed_block,
            paragraph     = paragraph,
            allow_unknown = False,
        )
        resp  = self.vllm.call(prompt)
        label = str(resp.get("label", "ERROR")).upper()

        if label == "TRUE":
            self.compact_state.working_conclusion = paragraph.strip()[:100]
            return True, None
        if label == "FALSE":
            return False, self._format_feedback(resp)
        return True, None

    # ── Feedback formatting ────────────────────────────────────────────────────

    @staticmethod
    def _format_feedback(
        resp:         dict,
        snomed_block: Optional[str] = None,
        prefix:       str           = "",
    ) -> str:
        """
        Plain text only — MedicalMonitor wraps this in [FEEDBACK]...[/FEEDBACK].
        Returns wrong_claim + correction + evidence for actionable feedback.
        """
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

    def _log(self, para_type: str, label: str, paragraph: str, feedback: Optional[str]) -> None:
        self.decision_log.append({
            "type":              para_type,
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
        _, new_paragraph = self._split_latest(text)

        if not new_paragraph.strip():
            return True, None

        para_type = self._classify(new_paragraph)

        if para_type == "OBSERVATION":
            passed, fb = self._verify_observation(new_paragraph)

        elif para_type in ("INFERENCE", "OPTION_COMPARISON"):
            passed, fb = self._verify_inference(new_paragraph, options, question)

        elif para_type == "CONCLUSION":
            passed, fb = self._verify_conclusion(new_paragraph, options, question)

        else:
            return True, None   # OTHER — transitions, revisions, etc.

        self._log(para_type, "PASS" if passed else "FAIL", new_paragraph, fb)
        return passed, fb