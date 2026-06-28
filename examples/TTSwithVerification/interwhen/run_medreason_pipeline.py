"""
run_medreason_pipeline.py — confidence-gated version

Changes vs original
-------------------
* --conf_threshold CLI arg (default 0.75) — passed to MedicalMonitor.
* --no_confidence_gate flag — disables the gate (reverts to original behaviour
  for ablation: verify every paragraph).
* Stream runner feeds vLLM logprob payloads into monitor.push_logprob_chunk()
  so the confidence scorer can use real token probabilities when available.
* Accuracy reporting now also prints gate stats from the decision_log:
    - total paragraphs triggered
    - paragraphs skipped by gate
    - paragraphs verified
    - verifier FAIL rate (among verified)
  This lets you tune conf_threshold without re-running everything.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
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

logger    = logging.getLogger(__name__)
tokenizer = None


# ══════════════════════════════════════════════════════════════════════════════
# MEDREASON LOADING  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class MedReasonLoader:
    HF_PATH = "UCSC-VLAA/MedReason"
    _OPT_RE = re.compile(r"([A-E])\.\s*(.*?)(?=\n[A-E]\.|$)", re.DOTALL)

    def load(self, split, start_idx, end_idx, max_samples):
        print(f"Loading {self.HF_PATH}  split={split} ...")
        raw = load_dataset(self.HF_PATH, split=split)
        end = min(end_idx, len(raw)) if end_idx > 0 else len(raw)
        raw = raw.select(range(start_idx, end))
        if max_samples > 0:
            raw = raw.select(range(min(max_samples, len(raw))))
        rows = [self._normalise(i, item) for i, item in enumerate(raw)]
        print(f"  -> {len(rows)} samples ready.\n")
        return rows

    def _normalise(self, idx, item):
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
    def _match_answer(clean, options):
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
# LLM SERVER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def init_llm_server(model_name, max_tokens=16 * 1024, port=8000):
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


def build_prompt(sample, tok):
    case_text = sample["question"]
    if sample["options"]:
        case_text += "\n\nOptions:\n" + "\n".join(
            f"{k}. {v}" for k, v in sample["options"].items()
        )
    user_prompt = USER_PROMPT_TEMPLATE.format(case_text=case_text)
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT_MEDICAL},
         {"role": "user",   "content": user_prompt}],
        tokenize=False, add_generation_prompt=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SCORING  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def exact_correctness_check(output_text, sample):
    m = re.search(r"\[FINAL ANSWER\](.*?)\[/FINAL ANSWER\]", output_text, re.DOTALL)
    if not m:
        return None
    block = m.group(1)
    opt   = re.search(r"Selected Option:\s*([A-E])", block, re.IGNORECASE)
    if not opt:
        return None
    return opt.group(1).strip().upper() == sample["answer"].strip().upper()


def rough_correctness_check(output_text, sample):
    m     = re.search(r"\[FINAL ANSWER\](.*?)\[/FINAL ANSWER\]", output_text, re.DOTALL)
    block = (m.group(1) if m else output_text[-600:]).lower()
    gt    = sample["options"].get(sample["answer"], "").lower()
    if not gt:
        return None
    gt_words = set(gt.split())
    overlap  = len(gt_words & set(block.split()))
    return overlap >= max(1, len(gt_words) // 2)


def check_correctness(output_text, sample):
    result = exact_correctness_check(output_text, sample)
    if result is not None:
        return result
    return rough_correctness_check(output_text, sample)


# ══════════════════════════════════════════════════════════════════════════════
# GATE STATS  (new)
# ══════════════════════════════════════════════════════════════════════════════

def extract_gate_stats(decision_log: list) -> dict:
    """
    Compute per-sample gate statistics from the decision_log.

    Returns a dict with:
        triggered  — how many paragraph boundaries fired
        skipped    — skipped by confidence gate (GATED_SKIP)
        verified   — actually sent to verifier
        failed     — verifier returned FAIL
        skip_rate  — skipped / triggered
        fail_rate  — failed / verified  (0 if verified == 0)
    """
    triggered = len(decision_log)
    skipped   = sum(1 for e in decision_log if e.get("label") == "SKIP")
    verified  = triggered - skipped
    failed    = sum(1 for e in decision_log if e.get("label") == "FAIL")
    return {
        "triggered": triggered,
        "skipped":   skipped,
        "verified":  verified,
        "failed":    failed,
        "skip_rate": round(skipped  / triggered, 4) if triggered else 0.0,
        "fail_rate": round(failed   / verified,  4) if verified  else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LOGPROB FEEDING HELPER  (new)
# ══════════════════════════════════════════════════════════════════════════════

def _make_logprob_callback(monitor: MedicalMonitor):
    """
    Returns a callback that can be registered with stream_completion (if your
    interwhen version supports an on_logprobs hook) to feed logprob data into
    the confidence scorer.

    If stream_completion does NOT support this hook yet, the callback is still
    returned but won't be called — the confidence scorer will fall back to the
    text-heuristic automatically.

    Expected call signature from stream_completion:
        callback(token_logprob_dicts: list[dict])
    where each dict has the vLLM logprob format:
        {"token": str, "logprob": float, "top_logprobs": {str: float}}
    """
    def _cb(token_logprob_dicts):
        monitor.push_logprob_chunk(token_logprob_dicts)
    return _cb


# ══════════════════════════════════════════════════════════════════════════════
# PER-SAMPLE RUN
# ══════════════════════════════════════════════════════════════════════════════

def run(args, sample):
    global tokenizer
    llm_server = init_llm_server(args.solver_lm, port=args.port)
    prompt     = build_prompt(sample, tokenizer)

    monitors = []
    if args.monitor:
        # If --no_confidence_gate is set, use threshold=1.0 so every trigger
        # fires the verifier (identical to original behaviour for ablation).
        effective_threshold = 1.0 if args.no_confidence_gate else args.conf_threshold

        monitor = MedicalMonitor(
            name             = "MedicalVerifier",
            instance         = sample,
            verifier_port    = args.verifier_port,
            verifier_model   = args.verifier_model,
            run_snomed       = not args.no_snomed,
            preprocess_case  = args.preprocess_case,
            prefetch_snomed  = args.prefetch_snomed,
            conf_threshold   = effective_threshold,
        )
        monitors = [monitor]

    output_text  = ""
    decision_log = []
    try:
        # Build optional kwargs for logprob hook — gracefully ignored if
        # stream_completion doesn't support the on_logprobs kwarg yet.
        extra_kwargs = {}
        if monitors:
            lp_callback = _make_logprob_callback(monitors[0])
            extra_kwargs["on_logprobs"] = lp_callback  # no-op if unsupported

        output_text = asyncio.run(stream_completion(
            prompt,
            llm_server      = llm_server,
            monitors        = tuple(monitors),
            async_execution = not args.debug,
            **extra_kwargs,
        ))
        if monitors:
            decision_log = monitors[0].verifier.decision_log
    except TypeError:
        # stream_completion doesn't accept on_logprobs yet — retry without it
        logger.info("[run] stream_completion doesn't support on_logprobs, retrying without")
        output_text = asyncio.run(stream_completion(
            prompt,
            llm_server      = llm_server,
            monitors        = tuple(monitors),
            async_execution = not args.debug,
        ))
        if monitors:
            decision_log = monitors[0].verifier.decision_log
    except Exception as e:
        logger.warning("stream_completion failed for sample %s: %s", sample["id"], e)
        output_text = output_text or ""

    correct       = check_correctness(output_text, sample)
    exact_matched = exact_correctness_check(output_text, sample)
    gate_stats    = extract_gate_stats(decision_log) if decision_log else {}

    result = {
        "sample_id":     sample["id"],
        "question":      sample["question"],
        "ground_truth":  sample["answer"],
        "output_text":   output_text,
        "correct":       correct,
        "exact_matched": exact_matched is not None,
        "decision_log":  decision_log,
        "gate_stats":    gate_stats,   # new field
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

    ap.add_argument("--solver_lm",       type=str,  required=True)
    ap.add_argument("--port",            type=int,  default=8000)

    ap.add_argument("--monitor",  "-m",  action="store_true")
    ap.add_argument("--verifier_port",   type=int,  default=8001)
    ap.add_argument("--verifier_model",  type=str,  default="medverifier")
    ap.add_argument("--no_snomed",       action="store_true")

    # Preprocessing
    ap.add_argument("--preprocess_case",  action="store_true")
    ap.add_argument("--prefetch_snomed",  action="store_true")

    # ── Confidence gate (new) ─────────────────────────────────────────────
    ap.add_argument(
        "--conf_threshold", type=float, default=0.65,
        help=(
            "Confidence gate threshold. Verifier is called only when "
            "paragraph confidence < this value. "
            "Text-heuristic mode (no logprobs): 0.65 skips confident paragraphs, "
            "verifies hedging/uncertain ones only. "
            "Logprob mode: raise to 0.80 for tighter gating. "
            "1.0 = verify everything (original behaviour). "
            "0.0 = never verify (disabled)."
        ),
    )
    ap.add_argument(
        "--no_confidence_gate", action="store_true",
        help="Disable confidence gate (verify every paragraph). Equivalent to --conf_threshold 1.0.",
    )
    # ─────────────────────────────────────────────────────────────────────

    ap.add_argument("--split",           type=str,  default="train")
    ap.add_argument("--max_samples",     type=int,  default=20)
    ap.add_argument("--start_idx",       type=int,  default=0)
    ap.add_argument("--end_idx",         type=int,  default=-1)
    ap.add_argument("--n_processes","-p",type=int,  default=8)
    ap.add_argument("--debug",     "-d", action="store_true")
    ap.add_argument("--continue_from","-c", type=str, default=None)
    ap.add_argument("--extra",           type=str,  default="")

    return ap.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE GATE STATS REPORTING  (new)
# ══════════════════════════════════════════════════════════════════════════════

def _print_gate_report(results: list, conf_threshold: float) -> None:
    total_triggered = sum(r.get("gate_stats", {}).get("triggered", 0) for r in results)
    total_skipped   = sum(r.get("gate_stats", {}).get("skipped",   0) for r in results)
    total_verified  = sum(r.get("gate_stats", {}).get("verified",  0) for r in results)
    total_failed    = sum(r.get("gate_stats", {}).get("failed",    0) for r in results)

    if total_triggered == 0:
        return

    print(f"\n{'═'*55}")
    print(f"  CONFIDENCE GATE REPORT  (τ = {conf_threshold})")
    print(f"{'═'*55}")
    print(f"  Paragraph boundaries triggered : {total_triggered}")
    print(f"  Skipped by gate                : {total_skipped}"
          f"  ({total_skipped/total_triggered:.1%})")
    print(f"  Sent to verifier               : {total_verified}"
          f"  ({total_verified/total_triggered:.1%})")
    if total_verified:
        print(f"  Verifier FAIL rate             : {total_failed/total_verified:.1%}"
              f"  ({total_failed} / {total_verified})")
    print(f"{'═'*55}\n")
    print("  Tuning guidance:")
    skip_rate = total_skipped / total_triggered
    if skip_rate < 0.20:
        print(f"  → Gate is skipping only {skip_rate:.0%} — consider raising τ to reduce")
        print("    verifier load (try τ = 0.80 or 0.85).")
    elif skip_rate > 0.60:
        print(f"  → Gate is skipping {skip_rate:.0%} — if accuracy dropped, try")
        print("    lowering τ to 0.65 or 0.70 to let more paragraphs through.")
    else:
        print(f"  → Skip rate {skip_rate:.0%} looks healthy. Fine-tune based on accuracy.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global tokenizer
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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

    if args.monitor and not args.no_confidence_gate:
        print(f"  Confidence gate ENABLED  (τ = {args.conf_threshold})")
        print("  Verifier will be called only for low-confidence paragraphs.\n")
    elif args.monitor and args.no_confidence_gate:
        print("  Confidence gate DISABLED  (ablation mode — verifying every paragraph)\n")

    if not args.debug:
        with Pool(processes=args.n_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(_run_wrapper, [(args, s) for s in samples]),
                total=len(samples),
            ))
    else:
        results = [run(args, samples[0])] if samples else []

    scored = [r["correct"] for r in results if r.get("correct") is not None]
    exact  = [r["exact_matched"] for r in results]

    if scored:
        print(f"\nAccuracy:       {sum(scored)}/{len(scored)} = {sum(scored)/len(scored):.2%}")
        print(f"[FINAL ANSWER] block found in {sum(exact)}/{len(exact)} outputs")

    if args.monitor:
        _print_gate_report(results, args.conf_threshold)

    print(f"Output -> {output_file}")


if __name__ == "__main__":
    main()