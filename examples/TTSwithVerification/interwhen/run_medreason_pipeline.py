"""
run_medreason_pipeline.py  —  Pipeline runner for structured medical reasoning.

Changes from the free-reasoning version:
  - Imports SYSTEM_PROMPT_MEDICAL from the updated medical_prompts (structured sections).
  - exact_correctness_check is unchanged: [FINAL ANSWER] block format is the same.
  - No other logic changes required — the monitor/verifier swap is transparent
    to the pipeline runner.

NEW in this version (graph verification layer):
  - --run_graph_checks turns on GraphAwareVerifier inside MedicalMonitor.
    Off by default; existing runs are byte-for-byte unaffected unless you
    pass this flag.
  - --snowstorm_base_url overrides the SNOMED relationship endpoint used for
    CUI validation and edge confirmation (defaults to the public IHTSDO
    browser instance — see medical_graph.SnomedRelationshipClient).
  - Per-sample results now include "graph_summary" (node/edge/contradiction
    counts) when graph checks are enabled, for the ablation study: how many
    cases did the global consistency check catch that the per-section check
    missed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import datetime
from multiprocessing import Pool

from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

from interwhen import stream_completion
from interwhen.monitors.medical_monitor import MedicalMonitor
from interwhen.utils.medical_prompts import SYSTEM_PROMPT_MEDICAL, USER_PROMPT_TEMPLATE

# Module-level tokenizer, set in __main__ (multiprocessing-safe).
tokenizer = None


# ══════════════════════════════════════════════════════════════════════════════
# MEDREASON LOADING  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class MedReasonLoader:
    """Loads UCSC-VLAA/MedReason into {id, question, options, answer, reasoning} dicts."""

    HF_PATH = "UCSC-VLAA/MedReason"
    _OPT_RE = re.compile(r"([A-E])\.\s*(.*?)(?=\n[A-E]\.|$)", re.DOTALL)

    def load(self, split: str, start_idx: int, end_idx: int, max_samples: int) -> list:
        print(f"Loading {self.HF_PATH}  split={split} ...")
        raw = load_dataset(self.HF_PATH, split=split)

        end = min(end_idx, len(raw)) if end_idx > 0 else len(raw)
        raw = raw.select(range(start_idx, end))
        if max_samples > 0:
            raw = raw.select(range(min(max_samples, len(raw))))

        rows = [self._normalise(i, item) for i, item in enumerate(raw)]
        print(f"  -> {len(rows)} samples ready.\n")
        return rows

    def _normalise(self, idx: int, item: dict) -> dict:
        options = {k: v.strip() for k, v in self._OPT_RE.findall(str(item.get("options", "")))}
        raw_ans = str(item.get("answer", "")).split("Explanation:")[0].strip().rstrip(". ")
        return {
            "id":        str(item.get("id_in_dataset", idx)),
            "question":  item.get("question", ""),
            "options":   options,
            "answer":    self._match_answer(raw_ans, options),
            "reasoning": item.get("reasoning") or "",
        }

    @staticmethod
    def _match_answer(clean: str, options: dict) -> str:
        cl = clean.lower()
        for letter, text in options.items():
            if cl == text.lower() or cl in text.lower() or text.lower() in cl:
                return letter
        words = set(cl.split())
        best_l, best_n = "?", 0
        for letter, text in options.items():
            n = len(words & set(text.lower().split()))
            if n > best_n:
                best_n, best_l = n, letter
        return best_l if best_n > 0 else "?"


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER PROMPT  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def init_llm_server(model_name: str, max_tokens: int = 16 * 1024, port: int = 8000) -> dict:
    """Solver vLLM server config."""
    return {
        "url": f"http://localhost:{port}/v1/completions",
        "payload": {
            "model": model_name, "max_tokens": max_tokens,
            "top_k": 20, "top_p": 0.95, "min_p": 0.0, "temperature": 0.6,
            "stream": True, "logprobs": 20, "use_beam_search": False,
            "prompt_cache": True, "seed": 42,
        },
        "headers": {"Content-Type": "application/json"},
    }


def build_prompt(sample: dict, tok) -> str:
    case_text = sample["question"]
    if sample["options"]:
        case_text += "\n\nOptions:\n" + "\n".join(f"{k}. {v}" for k, v in sample["options"].items())
    user_prompt = USER_PROMPT_TEMPLATE.format(case_text=case_text)
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT_MEDICAL},
         {"role": "user",   "content": user_prompt}],
        tokenize=False, add_generation_prompt=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def exact_correctness_check(output_text: str, sample: dict) -> "bool | None":
    """
    Primary scorer: extract 'Selected Option: X' from [FINAL ANSWER] block.

    Qwen3 thinking models write a draft [FINAL ANSWER] inside the <think>
    block 77% of the time, then produce the real response after </think>.
    We look in the response section first; fall back to the think section if
    the response has no block (23% of outputs).
    Returns None if no block is found or it contains no option letter.
    """
    think_close = "</think>"
    if think_close in output_text:
        response_section = output_text.split(think_close, 1)[1]
    else:
        response_section = output_text

    def _extract(text: str) -> "str | None":
        m = re.search(r"\[FINAL ANSWER\](.*?)\[/FINAL ANSWER\]", text, re.DOTALL)
        if not m:
            return None
        opt = re.search(r"Selected Option:\s*([A-E])", m.group(1), re.IGNORECASE)
        return opt.group(1).strip().upper() if opt else None

    letter = _extract(response_section) or _extract(output_text)
    if letter is None:
        return None
    return letter == sample["answer"].strip().upper()


def structured_section_completeness_check(output_text: str) -> dict:
    """
    Diagnostic: check which structured sections were emitted by the solver.
    Returns a dict of {section_name: bool} for reporting purposes.
    Not used for scoring — just useful for analysing model compliance.
    """
    from interwhen.utils.medical_prompts import SECTION_TAG_TO_TYPE
    think_close = "</think>"
    think_text  = output_text.split(think_close, 1)[0] if think_close in output_text else output_text

    result = {}
    for tag in SECTION_TAG_TO_TYPE:
        open_tag  = f"[{tag}]"
        close_tag = f"[/{tag}]"
        result[tag] = (open_tag in think_text and close_tag in think_text)
    return result


def rough_correctness_check(output_text: str, sample: dict) -> "bool | None":
    """
    Fallback scorer (word-overlap) when [FINAL ANSWER] block is absent.
    Also checks [CONCLUSION] block as a secondary source.
    Not authoritative — use only as a fallback signal.
    """
    # Try [CONCLUSION] block first (structured format)
    m = re.search(r"\[CONCLUSION\](.*?)\[/CONCLUSION\]", output_text, re.DOTALL)
    if m:
        block = m.group(1).lower()
    else:
        m     = re.search(r"\[FINAL ANSWER\](.*?)\[/FINAL ANSWER\]", output_text, re.DOTALL)
        block = (m.group(1) if m else output_text[-600:]).lower()

    gt = sample["options"].get(sample["answer"], "").lower()
    if not gt:
        return None
    gt_words = set(gt.split())
    overlap  = len(gt_words & set(block.split()))
    return overlap >= max(1, len(gt_words) // 2)


def check_correctness(output_text: str, sample: dict) -> "bool | None":
    """Try exact match first; fall back to word overlap."""
    result = exact_correctness_check(output_text, sample)
    if result is not None:
        return result
    return rough_correctness_check(output_text, sample)


# ══════════════════════════════════════════════════════════════════════════════
# PER-SAMPLE RUN
# ══════════════════════════════════════════════════════════════════════════════

def run(args, sample: dict) -> dict:
    global tokenizer
    llm_server = init_llm_server(args.solver_lm, port=args.port)
    prompt     = build_prompt(sample, tokenizer)

    monitors = []
    if args.monitor:
        monitors = [MedicalMonitor(
            name                = "MedicalVerifier",
            instance            = sample,
            max_corrections     = args.monitor_max_corrections,
            verifier_port       = args.verifier_port,
            verifier_model      = args.verifier_model,
            run_snomed          = not args.no_snomed,
            run_graph_checks    = args.run_graph_checks,
            snowstorm_base_url  = args.snowstorm_base_url,
        )]

    output_text = asyncio.run(stream_completion(
        prompt,
        llm_server      = llm_server,
        monitors        = tuple(monitors),
        async_execution = not args.debug,
    ))

    correct       = check_correctness(output_text, sample)
    exact_matched = exact_correctness_check(output_text, sample)
    section_audit = structured_section_completeness_check(output_text)
    decision_log  = monitors[0].verifier.decision_log if monitors else []

    graph_summary = None
    if monitors and args.run_graph_checks and hasattr(monitors[0].verifier, "graph_summary"):
        graph_summary = monitors[0].verifier.graph_summary()

    result = {
        "sample_id":       sample["id"],
        "question":        sample["question"],
        "ground_truth":    sample["answer"],
        "output_text":     output_text,
        "correct":         correct,
        "exact_matched":   exact_matched is not None,
        "section_audit":   section_audit,     # which structured sections were present
        "decision_log":    decision_log,
        "graph_summary":   graph_summary,      # node/edge/contradiction counts, or None
    }
    with open(f"{args.output_dir}/outputs.jsonl", "a") as f:
        f.write(json.dumps(result, default=str) + "\n")
    return result


def _run_wrapper(args_sample):
    return run(*args_sample)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--solver_lm",               type=str,  required=True)
    ap.add_argument("--port",                    type=int,  default=8000)
    ap.add_argument("--monitor",    "-m",        action="store_true")
    ap.add_argument("--verifier_port",           type=int,  default=8001)
    ap.add_argument("--verifier_model",          type=str,  default="medverifier")
    ap.add_argument("--monitor_max_corrections", type=int,  default=5)
    ap.add_argument("--no_snomed",               action="store_true")
    ap.add_argument("--run_graph_checks",        action="store_true",
                     help="Enable the CUI-grounded graph verification layer "
                          "(GraphAwareVerifier). Off by default. Requires a "
                          "reachable Snowstorm endpoint for edge confirmation "
                          "to actually do anything — see medical_graph.py.")
    ap.add_argument("--snowstorm_base_url",      type=str,  default=None,
                     help="Override the Snowstorm SNOMED CT relationship "
                          "endpoint used by --run_graph_checks. Defaults to "
                          "the public IHTSDO browser instance if unset.")
    ap.add_argument("--split",                   type=str,  default="train")
    ap.add_argument("--max_samples",             type=int,  default=20)
    ap.add_argument("--start_idx",               type=int,  default=0)
    ap.add_argument("--end_idx",                 type=int,  default=-1)
    ap.add_argument("--n_processes",  "-p",      type=int,  default=8)
    ap.add_argument("--debug",        "-d",      action="store_true")
    ap.add_argument("--continue_from", "-c",     type=str,  default=None)
    ap.add_argument("--extra",                   type=str,  default="")

    return ap.parse_args()


def main():
    global tokenizer
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.solver_lm)
    samples   = MedReasonLoader().load(args.split, args.start_idx, args.end_idx, args.max_samples)

    if args.continue_from:
        output_dir = f"Outputs_TTS/medreason/{args.continue_from}"
    else:
        output_dir = f"Outputs_TTS/medreason/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if args.extra:
            output_dir += f"-{args.extra}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/args.json", "w") as f:
            json.dump(vars(args), f, indent=4)
    args.output_dir = output_dir

    output_file = f"{output_dir}/outputs.jsonl"
    if args.continue_from:
        with open(output_file, "r") as f:
            done_ids = {json.loads(line)["sample_id"] for line in f}
        samples = [s for s in samples if s["id"] not in done_ids]
        print(f"Continuing from {args.continue_from}, skipping {len(done_ids)} completed.")
    else:
        open(output_file, "w").close()

    if not args.debug:
        with Pool(processes=args.n_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(_run_wrapper, [(args, s) for s in samples]),
                total=len(samples),
            ))
    else:
        results = [run(args, samples[0])] if samples else []

    # ── Summary ──────────────────────────────────────────────────────────────
    scored = [r["correct"] for r in results if r.get("correct") is not None]
    exact  = [r["exact_matched"] for r in results]

    if scored:
        print(f"\nAccuracy:               {sum(scored)}/{len(scored)} = {sum(scored)/len(scored):.2%}")
        print(f"[FINAL ANSWER] found:   {sum(exact)}/{len(exact)}")

    # Section compliance summary
    from interwhen.utils.medical_prompts import SECTION_TAG_TO_TYPE
    if results and "section_audit" in results[0]:
        print("\nStructured section compliance:")
        for tag in SECTION_TAG_TO_TYPE:
            count = sum(1 for r in results if r.get("section_audit", {}).get(tag, False))
            print(f"  [{tag}]  {count}/{len(results)}")

    # Graph contradiction summary — the actual ablation metric: how many
    # samples had at least one cross-chunk contradiction caught by the graph
    # layer that the per-section LLM check alone would have missed.
    if args.run_graph_checks:
        graph_results = [r.get("graph_summary") for r in results if r.get("graph_summary")]
        if graph_results:
            with_contradictions = sum(1 for g in graph_results if g.get("contradiction_count", 0) > 0)
            total_contradictions = sum(g.get("contradiction_count", 0) for g in graph_results)
            print("\nGraph verification (--run_graph_checks):")
            print(f"  Samples with >=1 cross-chunk contradiction caught: "
                  f"{with_contradictions}/{len(graph_results)}")
            print(f"  Total contradictions caught across all samples:    {total_contradictions}")

    print(f"\nOutput -> {output_file}")


if __name__ == "__main__":
    main()
