"""
medical_verifier.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from .medical_reasoning_prompts import MedicalReasoningPromptBuilder

logger = logging.getLogger(__name__)


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
# PUBMED CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class PubMedClient:
    """
    Fetches PubMed evidence for claim grounding.
    No API key required (rate-limited to 3 req/s without key).
    Set NCBI_API_KEY in environment for 10 req/s.

    Replaces SNOMED for comparative/ranking claims where definitions
    don't help — meta-analyses and systematic reviews give actual
    sensitivity/specificity numbers and clinical rankings.
    """
    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, max_results: int = 3, max_chars: int = 2000, timeout: int = 15):
        load_dotenv()
        self.max_results = max_results
        self.max_chars   = max_chars
        self.timeout     = timeout
        self.api_key     = os.getenv("NCBI_API_KEY", "")

    def _search(self, query: str) -> List[str]:
        params = {
            "db":      "pubmed",
            "term":    query,
            "retmax":  self.max_results,
            "sort":    "relevance",
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            r = requests.get(self.SEARCH_URL, params=params, timeout=self.timeout)
            r.raise_for_status()
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            logger.info("[PubMed] Query: %s | Found %d PMIDs", query[:80], len(ids))
            return ids
        except Exception as e:
            logger.warning("[PubMed] Search failed: %s", e)
            print(f"  [PubMed] Search failed: {e}")
            return []

    def _fetch(self, pmids: List[str]) -> str:
        if not pmids:
            return ""
        params = {
            "db":      "pubmed",
            "id":      ",".join(pmids),
            "rettype": "abstract",
            "retmode": "text",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            r = requests.get(self.FETCH_URL, params=params, timeout=self.timeout)
            r.raise_for_status()
            text = r.text[:self.max_chars]
            logger.info("[PubMed] Fetched %d chars from %d papers", len(text), len(pmids))
            return text
        except Exception as e:
            logger.warning("[PubMed] Fetch failed: %s", e)
            print(f"  [PubMed] Fetch failed: {e}")
            return ""

    def get_evidence(self, claim: str) -> str:
        """
        Search PubMed for evidence relevant to claim.
        Tries high-quality filter first (meta-analyses, systematic reviews, guidelines),
        falls back to plain search if no results.
        """
        print(f"  [PubMed] Searching for: {claim[:70]}")
        # High-quality filter first
        query_hq = (
            f"({claim}) AND "
            f"(meta-analysis[pt] OR systematic review[pt] OR practice guideline[pt])"
        )
        pmids = self._search(query_hq)
        if not pmids:
            pmids = self._search(claim)
        if not pmids:
            print(f"  [PubMed] No results")
            return ""
        print(f"  [PubMed] Found {len(pmids)} paper(s), fetching abstracts...")
        text = self._fetch(pmids[:self.max_results])
        print(f"  [PubMed] {len(text)} chars of evidence retrieved")
        return text


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
    # Evidence source: "snomed", "pubmed", "both", "none"
    # "pubmed" — PubMed abstracts for all claims
    # "snomed" — SNOMED definitions (skipped for comparative claims)
    # "both"   — SNOMED for terminology + PubMed for evidence
    # "none"   — no external evidence lookup
    evidence_source:          str   = "pubmed"
    # How many paragraphs to include per verification call.
    # 1 = just the latest paragraph (original behavior)
    # 3 = last 3 paragraphs — more context, fewer edge cases
    verification_window:      int   = 3
    # Max feedback blocks per sample before stopping
    max_feedback_per_sample:  int   = 10


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

    def add_fact(self, fact):     self._add(self.facts,    fact)
    def add_claim(self, claim):   self._add(self.claims,   claim)
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
    def __init__(self, vllm, snomed=None, pubmed=None):
        self.vllm   = vllm
        self.snomed = snomed
        self.pubmed = pubmed

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
    Classify-then-verify with configurable evidence source.

    evidence_source controls what external knowledge is fetched on FALSE:
      "pubmed" — PubMed abstracts (best for comparative/ranking claims)
      "snomed" — SNOMED CT definitions (best for terminology/anatomy)
      "both"   — both sources
      "none"   — no external lookup, verifier uses parametric knowledge only

    verification_window controls how many paragraphs are sent per call.
    """

    def __init__(
        self,
        vllm,
        snomed      = None,
        pubmed      = None,
        config      = None,
        compact_case= "",
        snomed_cache= None,
    ):
        self.vllm          = vllm
        self.snomed        = snomed
        self.pubmed        = pubmed
        self.config        = config or VerifierConfig()
        self.compact_case  = compact_case
        self.snomed_cache  = snomed_cache or {}
        self.compact_state = CompactState()
        self.decision_log: List[dict] = []
        self._pending_revision: Optional[Tuple[str, str]] = None
        self._corrected_topics: set = set()

    # ── text splitting ────────────────────────────────────────────────────────

    @staticmethod
    def _split_latest(text, think_open_tag="<think>", window=1):
        """
        Returns (prior_context, new_content).
        new_content = last `window` paragraphs since the last [FEEDBACK] block.
        Increasing window sends more context per verification call.
        """
        idx        = text.rfind(think_open_tag)
        think_text = text[idx + len(think_open_tag):] if idx != -1 else text

        parts         = re.split(r"\[FEEDBACK\].*?\[/FEEDBACK\]", think_text, flags=re.DOTALL)
        since_last_fb = parts[-1]
        prior_fb      = "".join(parts[:-1]).strip()

        paragraphs = [p.strip() for p in since_last_fb.split("\n\n") if p.strip()]

        if not paragraphs:
            return prior_fb, ""

        # Take last `window` paragraphs as the content to verify
        new_paras  = paragraphs[-window:]
        prior_paras= paragraphs[:-window]

        new_content   = "\n\n".join(new_paras)
        prior_context = "\n\n".join(filter(None, [prior_fb] + prior_paras))
        return prior_context.strip(), new_content

    def _truncate_prior(self, prior):
        limit = self.config.max_prior_context_chars
        return prior if len(prior) <= limit else "[...earlier context truncated...]\n" + prior[-limit:]

    @staticmethod
    def _options_text(options: dict) -> str:
        return "\n".join(f"{k}. {v}" for k, v in options.items()) if options else ""

    # ── knowledge question detection ──────────────────────────────────────────

    def _is_knowledge_question(self) -> bool:
        try:
            case_data   = json.loads(self.compact_case)
            null_values = {"null", "none", "none stated", "not stated", "n/a", ""}
            meaningful  = ("patient", "vitals", "chief_complaint", "labs", "imaging", "ecg")
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
                    defn = result.get("definition", "")
                    self.snomed_cache[term] = defn
                    logger.info("[SNOMED] '%s' → %s", term, defn[:120])
                    print(f"  [SNOMED realtime] '{term}' → {defn[:80]}")
                else:
                    logger.info("[SNOMED] '%s' → NOT FOUND", term)
                time.sleep(self.config.snomed_rate_limit_sleep)

    # ── Evidence fetching — routes by evidence_source config ─────────────────

    def _fetch_evidence(self, claim: str) -> Optional[str]:
        """
        Fetch external evidence for a claim based on config.evidence_source.
        No claim-type routing LLM call — source is purely the config param.
        SNOMED is skipped for comparative/ranking claims (regex guard only).
        """
        if not claim or self.config.evidence_source == "none":
            return None

        results = []

        if self.config.evidence_source in ("snomed", "both"):
            if self.snomed:
                terms_resp = self.vllm.call(
                    MedicalReasoningPromptBuilder.build_snomed_term_extraction_prompt(
                        question="", options_text="", reasoning_chunk=claim,
                    )
                )
                terms = terms_resp.get("terms", [])[:3]
                if isinstance(terms, list) and terms:
                    self._realtime_snomed(terms)
                    block = self._build_snomed_block()
                    if block:
                        results.append(f"SNOMED CT Definitions:\n{block}")

        if self.config.evidence_source in ("pubmed", "both"):
            if self.pubmed:
                text = self.pubmed.get_evidence(claim)
                if text:
                    results.append(f"PubMed Evidence:\n{text}")

        return "\n\n".join(results) if results else None

    # ── classify ─────────────────────────────────────────────────────────────

    def _classify(self, content):
        logger.info("[VERIFIER] Classifying (%d chars): %s...", len(content), content[:60])
        prompt = MedicalReasoningPromptBuilder.build_paragraph_classifier_prompt(
            paragraph=content
        )
        resp   = self.vllm.call(prompt)
        result = str(resp.get("class", "OTHER")).upper()
        reason = resp.get("reason", "")
        logger.info("[VERIFIER] Classification → %s | %s", result, reason)
        print(f"  [VERIFIER] Paragraph type: {result} | {reason}")
        return result

    # ── build verification prompt ─────────────────────────────────────────────

    def _build_verify_prompt(self, prior_context, content, allow_unknown=True):
        snomed_block = self._build_snomed_block()
        truncated    = self._truncate_prior(prior_context) if prior_context else "No prior reasoning."

        if snomed_block:
            return MedicalReasoningPromptBuilder.build_reasoning_hypothesis_snomed_prompt(
                reasoning_trace = truncated,
                hypothesis      = content,
                snomed_context  = snomed_block,
                allow_unknown   = allow_unknown,
            )
        return MedicalReasoningPromptBuilder.build_reasoning_hypothesis_prompt(
            reasoning_trace = truncated,
            hypothesis      = content,
            allow_unknown   = allow_unknown,
        )

    # ── verification methods ──────────────────────────────────────────────────

    def _verify_observation(self, content):
        if self._is_knowledge_question() or not self.compact_case.strip():
            logger.info("[VERIFIER] OBSERVATION → SKIP (knowledge MCQ)")
            print("  [VERIFIER] OBSERVATION: skipped (no clinical vignette)")
            return True, None

        logger.info("[VERIFIER] OBSERVATION grounding check...")
        prompt   = MedicalReasoningPromptBuilder.build_observation_grounding_prompt(
            compact_case=self.compact_case, paragraph=content,
        )
        resp     = self.vllm.call(prompt)
        grounded = resp.get("grounded", True)

        if grounded:
            logger.info("[VERIFIER] OBSERVATION → PASS")
            print("  [VERIFIER] OBSERVATION: PASS")
            for line in content.strip().splitlines():
                line = line.strip().lstrip("-•0123456789. ")
                if len(line) > 5:
                    self.compact_state.add_fact(line)
            return True, None

        issues = resp.get("issues", [])
        parts  = [
            f"{iss.get('type','error')}: '{iss.get('claim','')}' — {iss.get('reason','')}"
            for iss in issues
        ]
        fb = "Observation not grounded in case:\n" + "\n".join(f"  - {p}" for p in parts)
        logger.info("[VERIFIER] OBSERVATION → FAIL | %d issues", len(issues))
        print(f"  [VERIFIER] OBSERVATION: FAIL — {len(issues)} issue(s)")
        return False, fb

    def _verify_inference(self, content, prior_context, _retried=False):
        snomed_count = len(self.snomed_cache)
        logger.info("[VERIFIER] INFERENCE verify | %d chars | SNOMED cache: %d terms | retried: %s",
                    len(content), snomed_count, _retried)
        print(f"  [VERIFIER] INFERENCE: verifying ({len(content)} chars, {snomed_count} SNOMED terms)")

        prompt = self._build_verify_prompt(prior_context, content, allow_unknown=self.config.allow_unknown)
        logger.info("[VERIFIER] Prompt sent (%d chars)", len(prompt))
        resp   = self.vllm.call(prompt)
        label      = str(resp.get("label", "ERROR")).upper()
        confidence = float(resp.get("confidence", 1.0))
        logger.info("[VERIFIER] INFERENCE verdict: %s (confidence=%.2f)", label, confidence)
        print(f"  [VERIFIER] INFERENCE verdict: {label} (confidence={confidence:.2f})")

        if label == "TRUE":
            if self._pending_revision:
                self.compact_state.revise_claim(*self._pending_revision)
                logger.info("[VERIFIER] Pending revision applied: %s", self._pending_revision)
                self._pending_revision = None
            self.compact_state.add_claim(content.strip()[:100])
            print("  [VERIFIER] INFERENCE: PASS ✓")
            return True, None

        if label == "FALSE":
            # Confidence gate
            if confidence < 0.9:
                logger.info("[VERIFIER] FALSE ignored — confidence %.2f < 0.9", confidence)
                print(f"  [VERIFIER] INFERENCE: FALSE ignored (confidence={confidence:.2f} < 0.9)")
                return True, None

            wrong      = resp.get("wrong_claim")
            correction = resp.get("correction")

            # Flip-flop prevention
            topic_key = (wrong or "")[:50].lower().strip()
            if topic_key and topic_key in self._corrected_topics:
                logger.info("[VERIFIER] Flip-flop guard: already addressed '%s'", topic_key[:40])
                print("  [VERIFIER] INFERENCE: skipping repeat correction (flip-flop guard)")
                return True, None
            if topic_key:
                self._corrected_topics.add(topic_key)

            if wrong:
                self._pending_revision = (wrong, correction or "")

            logger.info("[VERIFIER] INFERENCE FAIL | confidence=%.2f | wrong: %s", confidence, wrong)
            print(f"  [VERIFIER] INFERENCE: FAIL ✗ (confidence={confidence:.2f})")
            print(f"    Claim : {wrong}")
            print(f"    Alt   : {correction}")

            # Fetch external evidence based on evidence_source
            evidence_context = self._fetch_evidence(wrong)

            # Directive only at very high confidence
            directive = (
                "Re-evaluate your option selection. Your final answer may need to change."
                if confidence >= 0.95 else None
            )
            fb = self._format_feedback(resp, evidence_context=evidence_context, directive=directive)
            logger.info("[VERIFIER] Feedback generated:\n%s", fb)
            return False, fb

        if label == "UNKNOWN" and not _retried:
            logger.info("[VERIFIER] INFERENCE UNKNOWN — fetching SNOMED for terms")
            print("  [VERIFIER] INFERENCE: UNKNOWN — fetching SNOMED...")
            if self.snomed and self.config.run_snomed:
                terms_resp = self.vllm.call(
                    MedicalReasoningPromptBuilder.build_snomed_term_extraction_prompt(
                        question="", options_text="", reasoning_chunk=content,
                    )
                )
                terms = terms_resp.get("terms", [])[:3]
                if isinstance(terms, list) and terms:
                    self._realtime_snomed(terms)
                    return self._verify_inference(content, prior_context, _retried=True)

            if self.config.unknown_defaults_to_pass:
                return True, None
            return False, self._format_feedback(resp, directive=None)

        if self.config.unknown_defaults_to_pass:
            return True, None
        return False, self._format_feedback(resp, directive=None)

    def _verify_conclusion(self, content, prior_context):
        logger.info("[VERIFIER] CONCLUSION verify | %d chars", len(content))
        print(f"  [VERIFIER] CONCLUSION: verifying ({len(content)} chars)")
        prompt = self._build_verify_prompt(prior_context, content, allow_unknown=False)
        resp   = self.vllm.call(prompt)
        label      = str(resp.get("label", "ERROR")).upper()
        confidence = float(resp.get("confidence", 1.0))
        logger.info("[VERIFIER] CONCLUSION verdict: %s (confidence=%.2f)", label, confidence)
        print(f"  [VERIFIER] CONCLUSION verdict: {label} (confidence={confidence:.2f})")

        if label == "TRUE":
            self.compact_state.working_conclusion = content.strip()[:100]
            print("  [VERIFIER] CONCLUSION: PASS ✓")
            return True, None

        if label == "FALSE":
            if confidence < 0.9:
                logger.info("[VERIFIER] CONCLUSION FALSE ignored — confidence %.2f < 0.9", confidence)
                print(f"  [VERIFIER] CONCLUSION: FALSE ignored (confidence={confidence:.2f} < 0.9)")
                return True, None
            wrong            = resp.get("wrong_claim")
            evidence_context = self._fetch_evidence(wrong)
            directive        = (
                "Re-evaluate your option selection. Your final answer may need to change."
                if confidence >= 0.95 else None
            )
            fb = self._format_feedback(resp, evidence_context=evidence_context, directive=directive)
            logger.info("[VERIFIER] CONCLUSION FAIL:\n%s", fb)
            print(f"  [VERIFIER] CONCLUSION: FAIL ✗ (confidence={confidence:.2f})")
            return False, fb

        return True, None

    # ── feedback formatting — directive, not punishing ────────────────────────

    @staticmethod
    def _format_feedback(resp, evidence_context=None, directive=None):
        """
        Directive style: guide thought, don't punish.
        'Consider reconsidering X' instead of 'Incorrect: X'.
        Provides evidence for the model to reason from rather than
        asserting a correction.
        """
        lines  = []
        wrong      = resp.get("wrong_claim")
        correction = resp.get("correction")
        evidence   = resp.get("evidence", [])

        if wrong:
            lines.append(f"Consider reconsidering: {wrong}")
        if correction:
            lines.append(f"Alternative perspective: {correction}")
        if evidence:
            lines.extend(f"  - {e}" for e in evidence)
        if evidence_context:
            lines.append(f"\nRelevant evidence:\n{evidence_context}")
        if directive:
            lines.append(f"\n{directive}")

        return "\n".join(l for l in lines if l).strip() \
               or "Review this reasoning step before continuing."

    # ── logging ───────────────────────────────────────────────────────────────

    def _log(self, para_type, label, content, feedback):
        self.decision_log.append({
            "type":              para_type,
            "label":             label,
            "paragraph_preview": content[:80],
            "feedback":          feedback,
        })

    # ── main entry point ──────────────────────────────────────────────────────

    def verify_trace(self, text, question="", options=None):
        """
        question and options accepted for interface compatibility
        but not sent to verifier LLM — reasoning consistency only.
        Uses config.verification_window paragraphs per call.
        """
        prior_context, new_content = self._split_latest(
            text, window=self.config.verification_window
        )

        if not new_content.strip():
            return True, None

        logger.info("[VERIFIER] ─────────────────────────────────────────────")
        logger.info("[VERIFIER] New content (%d chars, window=%d): %s...",
                    len(new_content), self.config.verification_window, new_content[:80])
        print(f"\n  [VERIFIER] >>> Verifying ({len(new_content)} chars, window={self.config.verification_window})")
        print(f"  [VERIFIER]     Preview: {new_content[:80]!r}")

        para_type = self._classify(new_content)

        if para_type == "OBSERVATION":
            passed, fb = self._verify_observation(new_content)
        elif para_type in ("INFERENCE", "OPTION_COMPARISON"):
            passed, fb = self._verify_inference(new_content, prior_context)
        elif para_type == "CONCLUSION":
            passed, fb = self._verify_conclusion(new_content, prior_context)
        else:
            logger.info("[VERIFIER] Type %s → SKIP", para_type)
            print(f"  [VERIFIER] Type {para_type}: skipped")
            return True, None

        result_str = "PASS ✓" if passed else "FAIL ✗"
        logger.info("[VERIFIER] <<< Result: %s", result_str)
        print(f"  [VERIFIER] <<< Result: {result_str}")

        self._log(para_type, "PASS" if passed else "FAIL", new_content, fb)
        return passed, fb
