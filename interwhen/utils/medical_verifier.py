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
    def __init__(self, base_url=None, model=None, temperature=0.0, max_tokens=1024, timeout=90):
        load_dotenv()
        self.base_url    = (base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model       = model or os.getenv("VLLM_MODEL", "medverifier")
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout

    @staticmethod
    def _strip_think_tags(text):
        last_end = text.rfind("</think>")
        if last_end != -1:
            after = text[last_end + len("</think>"):].strip()
            if after:
                return after
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _strip_fences(text):
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        return re.sub(r"\s*```$", "", text.strip()).strip()

    @staticmethod
    def _extract_json_object(text):
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

    def _clean_json(self, content):
        content = self._strip_think_tags(content)
        content = self._strip_fences(content)
        content = self._extract_json_object(content)
        return content.strip()

    def call(self, prompt, retries=3, wait=3):
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
    SEARCH_URL = "https://data.bioontology.org/search"

    def __init__(self, top_k=5, timeout=15):
        load_dotenv()
        self.top_k   = top_k
        self.timeout = timeout
        self.api_key = os.getenv("BIOPORTAL_API_KEY")
        if not self.api_key:
            raise ValueError("BIOPORTAL_API_KEY not found.")

    def _search(self, term):
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
    def _normalize(question, term):
        CORPUS_CALLOSUM_PARTS = {"rostrum", "genu", "body", "splenium"}
        if term.strip().lower() in CORPUS_CALLOSUM_PARTS and "corpus callosum" in question.lower():
            return f"{term} of corpus callosum"
        return term

    def enrich(self, question, option_text):
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

    def prefetch(self, terms, question="", sleep=0.3):
        cache: Dict[str, str] = {}
        for term in terms:
            result = self.enrich(question=question, option_text=term)
            if result.get("found"):
                cache[term] = result.get("definition", "")
            time.sleep(sleep)
        return cache

    @staticmethod
    def build_feedback_block(enrichments):
        if not enrichments:
            return ""
        lines = [f"- {t}: {d}" for t, d in enrichments.items() if d]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerifierConfig:
    run_snomed:               bool  = True
    unknown_defaults_to_pass: bool  = True
    allow_unknown:            bool  = True
    max_prior_context_chars:  int   = 4000
    snomed_rate_limit_sleep:  float = 0.3


# ══════════════════════════════════════════════════════════════════════════════
# COMPACT STATE
# ══════════════════════════════════════════════════════════════════════════════

class CompactState:
    MAX_ITEMS = 8

    def __init__(self):
        self.facts:              List[str] = []
        self.claims:             List[str] = []
        self.ruled_out:          List[str] = []
        self.working_conclusion: str       = ""

    def _add(self, lst, item):
        if item and item not in lst:
            lst.append(item[:120])
        if len(lst) > self.MAX_ITEMS:
            lst[:] = lst[-self.MAX_ITEMS:]

    def add_fact(self, fact):    self._add(self.facts,   fact)
    def add_claim(self, claim):  self._add(self.claims,  claim)
    def add_ruled_out(self, opt): self._add(self.ruled_out, opt)

    def revise_claim(self, wrong, correction):
        self.claims = [c for c in self.claims if wrong.lower()[:40] not in c.lower()]
        if correction:
            self.add_claim(f"[REVISED] {correction}")

    def to_str(self):
        lines = []
        if self.facts:              lines.append("Facts: "              + "; ".join(self.facts))
        if self.claims:             lines.append("Claims: "             + "; ".join(self.claims))
        if self.ruled_out:          lines.append("Ruled out: "          + "; ".join(self.ruled_out))
        if self.working_conclusion: lines.append(f"Working conclusion: {self.working_conclusion}")
        return "\n".join(lines) if lines else "Nothing established yet."

    def reset(self):
        self.facts.clear(); self.claims.clear()
        self.ruled_out.clear(); self.working_conclusion = ""


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class MedicalPreprocessor:
    def __init__(self, vllm, snomed):
        self.vllm   = vllm
        self.snomed = snomed

    def extract_case_facts(self, question):
        prompt = MedicalReasoningPromptBuilder.build_case_extraction_prompt(case_text=question)
        resp   = self.vllm.call(prompt)
        if "error" in resp:
            return question[:2000]
        return json.dumps(resp, indent=2)

    def prefetch_snomed(self, question, options):
        if self.snomed is None:
            return {}
        opts_text = "\n".join(f"{k}. {v}" for k, v in options.items())
        prompt    = MedicalReasoningPromptBuilder.build_snomed_term_extraction_prompt(
            question="", options_text="", reasoning_chunk=opts_text,
        )
        resp  = self.vllm.call(prompt)
        terms = resp.get("terms", [])
        if not isinstance(terms, list) or not terms:
            print("  [Preprocessor] Term extraction empty, using option texts as terms")
            terms = list(options.values())
        terms = terms[:8]
        print(f"  [Preprocessor] Pre-fetching SNOMED for {len(terms)} terms: {terms}")
        cache = self.snomed.prefetch(terms, question=question, sleep=0.3)
        print(f"  [Preprocessor] Cached {len(cache)} definitions.")
        return cache


# ══════════════════════════════════════════════════════════════════════════════
# MEDICAL REASONING VERIFIER
# ══════════════════════════════════════════════════════════════════════════════

class MedicalReasoningVerifier:
    """
    Verifies reasoning paragraphs against the accumulated reasoning trace only.
    Does NOT send the original question or options to the verifier LLM —
    the verifier judges reasoning consistency, not factual knowledge.
    """

    def __init__(self, vllm, snomed=None, config=None, compact_case="", snomed_cache=None):
        self.vllm          = vllm
        self.snomed        = snomed
        self.config        = config or VerifierConfig()
        self.compact_case  = compact_case
        self.snomed_cache  = snomed_cache or {}
        self.compact_state = CompactState()
        self.decision_log: List[dict] = []
        self._pending_revision: Optional[Tuple[str, str]] = None

    # ── text splitting ────────────────────────────────────────────────────────

    @staticmethod
    def _split_latest(text, think_open_tag="<think>"):
        idx        = text.rfind(think_open_tag)
        think_text = text[idx + len(think_open_tag):] if idx != -1 else text

        parts         = re.split(r"\[FEEDBACK\].*?\[/FEEDBACK\]", think_text, flags=re.DOTALL)
        since_last_fb = parts[-1]
        prior_fb      = "".join(parts[:-1]).strip()

        paragraphs = [p.strip() for p in since_last_fb.split("\n\n") if p.strip()]

        if not paragraphs:
            return prior_fb, ""

        new_paragraph    = paragraphs[-1]
        prior_paragraphs = "\n\n".join(paragraphs[:-1])
        prior_context    = "\n\n".join(filter(None, [prior_fb, prior_paragraphs]))
        return prior_context.strip(), new_paragraph

    def _truncate_prior(self, prior):
        limit = self.config.max_prior_context_chars
        return prior if len(prior) <= limit else "[...earlier context truncated...]\n" + prior[-limit:]

    # ── knowledge question detection ─────────────────────────────────────────

    def _is_knowledge_question(self) -> bool:
        """True when there is no clinical vignette — just a factual MCQ."""
        try:
            case_data     = json.loads(self.compact_case)
            meaningful    = ("patient", "vitals", "chief_complaint", "labs", "imaging", "ecg")
            null_values   = {"null", "none", "none stated", "not stated", "n/a", ""}
            for k in meaningful:
                val = case_data.get(k)
                if val and str(val).strip().lower() not in null_values:
                    return False
            return True
        except Exception:
            return not bool(self.compact_case.strip())

    # ── SNOMED helpers ────────────────────────────────────────────────────────

    def _build_snomed_block(self):
        if not self.snomed_cache:
            return ""
        return "\n".join(f"- {t}: {d}" for t, d in self.snomed_cache.items() if d)

    def _realtime_snomed(self, terms, question=""):
        if self.snomed is None:
            return
        for term in terms:
            if term not in self.snomed_cache:
                result = self.snomed.enrich(question=question, option_text=term)
                if result.get("found"):
                    self.snomed_cache[term] = result.get("definition", "")
                    print(f"  [SNOMED realtime] '{term}'")
                time.sleep(self.config.snomed_rate_limit_sleep)

    def _fetch_snomed_for_claim(self, claim):
        """Extract terms from a claim and fetch SNOMED — used on FALSE."""
        if not self.snomed or not self.config.run_snomed or not claim:
            return None
        terms_resp = self.vllm.call(
            MedicalReasoningPromptBuilder.build_snomed_term_extraction_prompt(
                question="", options_text="", reasoning_chunk=claim,
            )
        )
        terms = terms_resp.get("terms", [])[:3]
        if not isinstance(terms, list) or not terms:
            return None
        self._realtime_snomed(terms)
        block = self._build_snomed_block()
        return block if block.strip() else None

    # ── classify ─────────────────────────────────────────────────────────────

    def _classify(self, paragraph):
        prompt = MedicalReasoningPromptBuilder.build_paragraph_classifier_prompt(
            paragraph=paragraph
        )
        resp = self.vllm.call(prompt)
        return str(resp.get("class", "OTHER")).upper()

    # ── build verification prompt ─────────────────────────────────────────────

    def _build_verify_prompt(self, prior_context, paragraph, allow_unknown=True):
        """Always uses reasoning-trace-only prompt — no question or options."""
        snomed_block = self._build_snomed_block()
        truncated    = self._truncate_prior(prior_context) if prior_context else "No prior reasoning."

        if snomed_block:
            return MedicalReasoningPromptBuilder.build_reasoning_hypothesis_snomed_prompt(
                reasoning_trace = truncated,
                hypothesis      = paragraph,
                snomed_context  = snomed_block,
                allow_unknown   = allow_unknown,
            )
        return MedicalReasoningPromptBuilder.build_reasoning_hypothesis_prompt(
            reasoning_trace = truncated,
            hypothesis      = paragraph,
            allow_unknown   = allow_unknown,
        )

    # ── verification methods ──────────────────────────────────────────────────

    def _verify_observation(self, paragraph):
        """Grounding check — skip entirely for knowledge MCQs with no vignette."""
        if self._is_knowledge_question() or not self.compact_case.strip():
            # No clinical vignette to ground against — pass through
            return True, None

        prompt   = MedicalReasoningPromptBuilder.build_observation_grounding_prompt(
            compact_case=self.compact_case, paragraph=paragraph,
        )
        resp     = self.vllm.call(prompt)
        grounded = resp.get("grounded", True)

        if grounded:
            for line in paragraph.strip().splitlines():
                line = line.strip().lstrip("-•0123456789. ")
                if len(line) > 5:
                    self.compact_state.add_fact(line)
            return True, None

        issues = resp.get("issues", [])
        parts  = [
            f"{iss.get('type','error')}: '{iss.get('claim','')}' — {iss.get('reason','')}"
            for iss in issues
        ]
        return False, "Observation not grounded in case:\n" + "\n".join(f"  - {p}" for p in parts)

    def _verify_inference(self, paragraph, prior_context, _retried=False):
        prompt = self._build_verify_prompt(prior_context, paragraph, allow_unknown=self.config.allow_unknown)
        resp   = self.vllm.call(prompt)
        label  = str(resp.get("label", "ERROR")).upper()

        if label == "TRUE":
            if self._pending_revision:
                self.compact_state.revise_claim(*self._pending_revision)
                self._pending_revision = None
            self.compact_state.add_claim(paragraph.strip()[:100])
            return True, None

        if label == "FALSE":
            wrong      = resp.get("wrong_claim")
            correction = resp.get("correction")
            if wrong:
                self._pending_revision = (wrong, correction or "")
            snomed_block = self._fetch_snomed_for_claim(wrong)
            return False, self._format_feedback(resp, snomed_block=snomed_block)

        if label == "UNKNOWN" and not _retried:
            if self.snomed and self.config.run_snomed:
                terms_resp = self.vllm.call(
                    MedicalReasoningPromptBuilder.build_snomed_term_extraction_prompt(
                        question="", options_text="", reasoning_chunk=paragraph,
                    )
                )
                terms = terms_resp.get("terms", [])[:3]
                if isinstance(terms, list) and terms:
                    self._realtime_snomed(terms)
                    return self._verify_inference(paragraph, prior_context, _retried=True)

            if self.config.unknown_defaults_to_pass:
                return True, None
            return False, self._format_feedback(resp, prefix="Unresolved uncertainty: ")

        # UNKNOWN after retry or unexpected label
        if self.config.unknown_defaults_to_pass:
            return True, None
        return False, self._format_feedback(resp, prefix="Unresolved uncertainty: ")

    def _verify_conclusion(self, paragraph, prior_context):
        prompt = self._build_verify_prompt(prior_context, paragraph, allow_unknown=False)
        resp   = self.vllm.call(prompt)
        label  = str(resp.get("label", "ERROR")).upper()

        if label == "TRUE":
            self.compact_state.working_conclusion = paragraph.strip()[:100]
            return True, None
        if label == "FALSE":
            wrong        = resp.get("wrong_claim")
            snomed_block = self._fetch_snomed_for_claim(wrong)
            return False, self._format_feedback(resp, snomed_block=snomed_block)
        return True, None

    # ── feedback formatting ───────────────────────────────────────────────────

    @staticmethod
    def _format_feedback(resp, snomed_block=None, prefix=""):
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

    # ── logging ───────────────────────────────────────────────────────────────

    def _log(self, para_type, label, paragraph, feedback):
        self.decision_log.append({
            "type":              para_type,
            "label":             label,
            "paragraph_preview": paragraph[:80],
            "feedback":          feedback,
        })

    # ── main entry point ──────────────────────────────────────────────────────

    def verify_trace(self, text, question="", options=None):
        """
        question and options are accepted for interface compatibility
        but NOT sent to the verifier LLM — the verifier judges reasoning
        consistency only.
        """
        prior_context, new_paragraph = self._split_latest(text)

        if not new_paragraph.strip():
            return True, None

        para_type = self._classify(new_paragraph)

        if para_type == "OBSERVATION":
            passed, fb = self._verify_observation(new_paragraph)
        elif para_type in ("INFERENCE", "OPTION_COMPARISON"):
            passed, fb = self._verify_inference(new_paragraph, prior_context)
        elif para_type == "CONCLUSION":
            passed, fb = self._verify_conclusion(new_paragraph, prior_context)
        else:
            return True, None

        self._log(para_type, "PASS" if passed else "FAIL", new_paragraph, fb)
        return passed, fb