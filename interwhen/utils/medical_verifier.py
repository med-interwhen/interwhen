"""
medical_verifier.py
=====================
Verifier core for medical reasoning traces.

Constructed by MedicalMonitor.__init__ (medical_monitor.py) from
verifier_port/verifier_model/run_snomed, and called from
MedicalMonitor._call_verifier via verify_trace().

Contract
--------
    verify_trace(text, question, options) -> (passed: bool, feedback: str | None)

feedback is plain text. MedicalMonitor wraps it in [FEEDBACK]...[/FEEDBACK]
itself, so feedback returned here must never include those markers.

All retry-counting, max_corrections, and the give-up sentinel live in
MedicalMonitor.verify()/fix(). This file only judges whether the newest
content in a trace is acceptable, and if not, why.

.env
----
  VLLM_BASE_URL=http://localhost:8000/v1
  VLLM_MODEL=medverifier
  BIOPORTAL_API_KEY=<your-key>
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import requests
from dotenv import load_dotenv

from .medical_reasoning_prompts import MedicalReasoningPromptBuilder


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL vLLM CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class LocalVLLMClient:
    """Sync wrapper around vLLM's OpenAI-compatible /v1/chat/completions.

    This is the verifier's own model — a separate vLLM server/port from
    whatever is serving the solver.
    """

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 90,
    ):
        load_dotenv()
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or os.getenv("VLLM_MODEL", "medverifier")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    # ── JSON cleaning pipeline (verifier models often think before answering) ──

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

    # ── Sync call — MedicalMonitor._call_verifier is sync, so this must be too ──

    def call(self, prompt: str, retries: int = 3, wait: int = 3) -> dict:
        """POST prompt to vLLM, return parsed JSON dict, or {"error": ...}."""
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
                    "model": self.model,
                    "messages": [{"role": "user", "content": send_prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
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


class SnomedClient:
    """BioPortal SNOMED CT lookup. Requires BIOPORTAL_API_KEY in .env or environment."""

    SEARCH_URL = "https://data.bioontology.org/search"

    def __init__(self, top_k: int = 5, timeout: int = 15):
        load_dotenv()
        self.top_k = top_k
        self.timeout = timeout
        self.api_key = os.getenv("BIOPORTAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "BIOPORTAL_API_KEY not found in environment. "
                "Add it to .env or set VerifierConfig.run_snomed=False."
            )

    def _search(self, term: str) -> list:
        headers = {"Authorization": f"apikey token={self.api_key}"}
        params = {"q": term, "ontologies": "SNOMEDCT", "pagesize": self.top_k}
        try:
            r = requests.get(self.SEARCH_URL, headers=headers, params=params, timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("collection", [])
        except Exception as e:
            print(f"  [SNOMED] search failed for '{term}': {e}")
            return []

    @staticmethod
    def _normalize(question: str, term: str) -> str:
        """
        Context-aware query normalization.
        Example extension point: if a flagged term is ambiguous on its own
        but unambiguous in the case context, expand it here. Add new cases
        as you hit poor-recall terms in practice.
        """
        CORPUS_CALLOSUM_PARTS = {"rostrum", "genu", "body", "splenium"}
        if term.strip().lower() in CORPUS_CALLOSUM_PARTS and "corpus callosum" in question.lower():
            return f"{term} of corpus callosum"
        return term

    def enrich(self, question: str, option_text: str) -> dict:
        """
        Look up a flagged term in SNOMED CT, trying a few query variants.
        Returns {"found", "query", "definition", "synonyms", "ontology_status"}.
        """
        normalized = self._normalize(question, option_text)
        variants = [normalized, normalized.lower(), normalized.replace(" of ", " ")]

        for term in variants:
            print(f"    [SNOMED] -> '{term}'")
            results = self._search(term)
            if results:
                top = results[0]
                defs = top.get("definition", [])
                defn = defs[0] if isinstance(defs, list) and defs else top.get("prefLabel", "")
                return {
                    "found": True,
                    "query": normalized,
                    "definition": defn,
                    "synonyms": top.get("synonym", [])[:5],
                    "ontology_status": "FOUND",
                }

        return {"found": False, "query": normalized, "definition": "", "synonyms": [], "ontology_status": "NOT_FOUND"}

    @staticmethod
    def build_feedback_block(enrichments: dict) -> str:
        """Render {term: enrich_dict} into the block embedded in verify_cycle_snomed."""
        if not enrichments:
            return "No SNOMED CT feedback available."

        lines = []
        for _term, data in enrichments.items():
            query = data.get("query", "")
            status = data.get("ontology_status", "UNKNOWN")
            defn = data.get("definition", "")
            syns = data.get("synonyms", [])
            if status == "FOUND":
                syn_txt = f" Synonyms: {', '.join(syns[:5])}." if syns else ""
                lines.append(f"- {query}: {defn}.{syn_txt}")
            else:
                lines.append(f"- {query}: SNOMED CT entry not found.")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# VERIFIER CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerifierConfig:
    """Tunables for the orchestration around the prompts — not the prompts themselves."""

    run_snomed: bool = True                 # query SNOMED when the judge says UNKNOWN
    unknown_defaults_to_pass: bool = True    # policy for UNKNOWN that SNOMED can't resolve either
    allow_unknown: bool = True               # passed through to build_reasoning_hypothesis_prompt
    max_prior_context_chars: int = 4000      # truncate very long accepted-context blocks
    snomed_rate_limit_sleep: float = 0.3


class MedicalReasoningVerifier:
    """Judges a medical reasoning trace.

    verify_trace() is the entry point MedicalMonitor calls: one method,
    synchronous, (text) -> (passed, feedback).
    """

    def __init__(
        self,
        vllm: LocalVLLMClient,
        snomed: Optional[SnomedClient] = None,
        config: Optional[VerifierConfig] = None,
    ):
        self.vllm = vllm
        self.snomed = snomed
        self.config = config or VerifierConfig()

    @staticmethod
    def _split_latest(text: str, think_open_tag: str = "<think>") -> Tuple[str, str]:
        """
        Returns (prior_context, new_content).

        prior_context = everything inside the <think> block BEFORE the last
                         [FEEDBACK]...[/FEEDBACK] marker (already verified).
        new_content   = everything after it (what we actually judge now).

        We deliberately do NOT regex-parse Observation/Inference/Evidence/
        Diagnosis/Plan into separate fields. The judge LLM reads structured
        text natively and was told the section conventions in the prompt;
        brittle per-section regex on free-form structured medical text is a
        worse failure mode than trusting a capable judge model to read the
        structure it (or a sibling model) was trained to produce.
        """
        idx = text.rfind(think_open_tag)
        think_text = text[idx + len(think_open_tag):] if idx != -1 else text

        parts = re.split(r"\[FEEDBACK\].*?\[/FEEDBACK\]", think_text, flags=re.DOTALL)
        new_content = parts[-1].strip()
        prior_context = "".join(parts[:-1]).strip()
        return prior_context, new_content

    def _truncate_prior(self, prior: str) -> str:
        limit = self.config.max_prior_context_chars
        if len(prior) <= limit:
            return prior
        return "[...earlier accepted context truncated...]\n" + prior[-limit:]

    @staticmethod
    def _options_text(options: dict) -> str:
        return "\n".join(f"{k}. {v}" for k, v in options.items()) if options else ""

    @staticmethod
    def _format_feedback(resp: dict, snomed_block: Optional[str] = None, prefix: str = "") -> str:
        """
        Plain text only — never include [FEEDBACK]/[/FEEDBACK] markers here.
        MedicalMonitor.verify() wraps whatever this returns in those markers
        itself; double-wrapping would result if we added them too.

        Reads build_reasoning_hypothesis_prompt's actual output schema:
        {"label", "evidence": [...], "option_probabilities"} — there is no
        separate "reasoning" string field, so evidence is the feedback body.
        """
        evidence = resp.get("evidence", [])
        lines = []
        if prefix:
            lines.append(prefix.strip())
        if evidence:
            lines.append("Evidence:")
            lines.extend(f"  - {e}" for e in evidence)
        if snomed_block:
            lines.append(f"\nRelevant SNOMED CT definitions:\n{snomed_block}")
        text = "\n".join(l for l in lines if l).strip()
        return text or "The verifier rejected this reasoning. Please reconsider."

    def verify_trace(
        self,
        text: str,
        question: str = "",
        options: Optional[dict] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Matches MedicalMonitor._call_verifier's contract exactly.

        Calls MedicalReasoningPromptBuilder.build_reasoning_hypothesis_prompt
        unchanged: new_content becomes `hypothesis`, and question + options +
        accepted-so-far context becomes `reasoning_trace`. On label UNKNOWN,
        the same method is called again with SNOMED CT definitions appended
        to reasoning_trace — there is no separate SNOMED prompt.

        This method never raises on a malformed verifier response; a
        parsing failure fails OPEN (returns True, None) rather than
        blocking generation on a problem with the judge call itself.
        """
        options = options or {}
        prior_context, new_content = self._split_latest(text)

        if not new_content.strip():
            return True, None  # nothing new since the last check yet

        context_block = question
        if options:
            context_block += "\n\nOptions:\n" + self._options_text(options)
        if prior_context.strip():
            context_block += "\n\nReasoning so far:\n" + self._truncate_prior(prior_context)

        prompt = MedicalReasoningPromptBuilder.build_reasoning_hypothesis_prompt(
            reasoning_trace=context_block,
            hypothesis=new_content,
            allow_unknown=self.config.allow_unknown,
        )
        resp = self.vllm.call(prompt)
        label = str(resp.get("label", "ERROR")).strip().upper()

        if label == "TRUE":
            return True, None

        if label == "FALSE":
            return False, self._format_feedback(resp)

        if label == "UNKNOWN":
            if self.snomed is not None and self.config.run_snomed:
                print(f"  [SNOMED] resolving: '{new_content[:60]}'")
                enrichment = self.snomed.enrich(question=question, option_text=new_content)
                time.sleep(self.config.snomed_rate_limit_sleep)
                snomed_block = SnomedClient.build_feedback_block({new_content: enrichment})

                prompt2 = MedicalReasoningPromptBuilder.build_reasoning_hypothesis_prompt(
                    reasoning_trace=context_block + f"\n\nSNOMED CT definitions:\n{snomed_block}",
                    hypothesis=new_content,
                    allow_unknown=self.config.allow_unknown,
                )
                resp2 = self.vllm.call(prompt2)
                label2 = str(resp2.get("label", "ERROR")).strip().upper()

                if label2 == "FALSE":
                    return False, self._format_feedback(resp2, snomed_block=snomed_block)
                if label2 == "TRUE":
                    return True, None
                # still ambiguous even after SNOMED -> fall through to policy below,
                # using whichever response we have (resp2 if it at least parsed).
                resp = resp2 if "error" not in resp2 else resp

            # No SNOMED client, or unresolved after SNOMED.
            if self.config.unknown_defaults_to_pass:
                return True, None
            return False, self._format_feedback(resp, prefix="Unresolved uncertainty: ")

        # Malformed/unexpected label from the judge model — fail open.
        return True, None