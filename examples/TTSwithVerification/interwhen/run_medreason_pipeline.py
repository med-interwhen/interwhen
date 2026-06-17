"""
run_medreason_pipeline.py
============================
Runs MedReason cases through interwhen's stream_completion with
MedicalMonitor performing live verification during generation.

Per sample: builds the solver prompt from SYSTEM_PROMPT_MEDICAL and
USER_PROMPT_TEMPLATE, constructs a MedicalMonitor configured with the
verifier's connection details, and streams the solver's output through
it. The monitor inspects the trace as it generates, injects feedback on
failure, and retries or gives up internally. This script persists the
final generated text per sample to JSONL and prints an approximate
correctness signal.

Two separate vLLM servers
--------------------------
  --solver_lm / --port                the model generating the trace
  --verifier_port / --verifier_model  the model judging it

They can point at the same server+model by passing matching ports.

Usage
-----
  python run_medreason_pipeline.py \\
      --solver_lm Qwen/Qwen3-8B --port 8000 \\
      --verifier_port 8001 --verifier_model medverifier \\
      --monitor
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
# MEDREASON LOADING
# ══════════════════════════════════════════════════════════════════════════════

class MedReasonLoader:
    """Loads UCSC-VLAA/MedReason into {id, question, options, answer, reasoning} dicts."""

    HF_PATH = "UCSC-VLAA/MedReason"
    _OPT_RE = re.compile(r"([A-D])\.\s*(.*?)(?=\n[A-D]\.|$)", re.DOTALL)

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
            "id": str(item.get("id_in_dataset", idx)),
            "question": item.get("question", ""),
            "options": options,
            "answer": self._match_answer(raw_ans, options),
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
# SOLVER PROMPT
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
         {"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True,
    )


def rough_correctness_check(output_text: str, sample: dict) -> "bool | None":
    """
    Approximate, word-overlap heuristic ONLY — not authoritative. The
    structured prompt produces a free-text Diagnosis, not a lettered MCQ
    choice, so there is no exact-match scorer here. Use this as a rough
    signal, not a benchmark number.
    """
    m = re.search(r"\[FINAL ANSWER\](.*?)\[/FINAL ANSWER\]", output_text, re.DOTALL)
    block = (m.group(1) if m else output_text[-600:]).lower()
    gt_text = sample["options"].get(sample["answer"], "").lower()
    if not gt_text:
        return None
    gt_words = set(gt_text.split())
    overlap = len(gt_words & set(block.split()))
    return overlap >= max(1, len(gt_words) // 2)


# ══════════════════════════════════════════════════════════════════════════════
# PER-SAMPLE RUN
# ══════════════════════════════════════════════════════════════════════════════

def run(args, sample: dict) -> dict:
    global tokenizer
    llm_server = init_llm_server(args.solver_lm, port=args.port)
    prompt = build_prompt(sample, tokenizer)

    if args.monitor:
        monitors = [MedicalMonitor(
            name="MedicalVerifier",
            instance=sample,
            line_interval=args.line_interval,
            max_corrections=args.monitor_max_corrections,
            verifier_port=args.verifier_port,
            verifier_model=args.verifier_model,
            run_snomed=not args.no_snomed,
        )]
    else:
        monitors = []

    output_text = asyncio.run(stream_completion(
        prompt, llm_server=llm_server,
        monitors=tuple(monitors) if monitors else [],
        async_execution=not args.debug,
    ))

    result = {
        "sample_id": sample["id"],
        "question": sample["question"],
        "ground_truth_key": sample["answer"],
        "ground_truth_text": sample["options"].get(sample["answer"], ""),
        "output_text": output_text,
        "approx_correct": rough_correctness_check(output_text, sample),
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
    ap = argparse.ArgumentParser(description="MedReason solver+verifier run via interwhen stream_completion")

    ap.add_argument("--solver_lm", type=str, required=True, help="Solver model name")
    ap.add_argument("--port", type=int, default=8000, help="Solver vLLM server port")

    ap.add_argument("--monitor", "-m", action="store_true", help="Enable live verification")
    ap.add_argument("--verifier_port", type=int, default=8001, help="Verifier vLLM server port")
    ap.add_argument("--verifier_model", type=str, default="medverifier", help="Verifier --served-model-name")
    ap.add_argument("--monitor_max_corrections", type=int, default=5)
    ap.add_argument("--line_interval", type=int, default=5, help="Non-empty think-block lines between verifier calls")
    ap.add_argument("--no_snomed", action="store_true", help="Disable SNOMED fallback on UNKNOWN verdicts")

    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max_samples", type=int, default=20, help="-1 = all")
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=-1)

    ap.add_argument("--n_processes", "-p", type=int, default=8)
    ap.add_argument("--debug", "-d", action="store_true", help="Single-process, synchronous monitor execution")
    ap.add_argument("--continue_from", "-c", type=str, default=None)
    ap.add_argument("--extra", type=str, default="")

    return ap.parse_args()


def main():
    global tokenizer
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.solver_lm)
    samples = MedReasonLoader().load(args.split, args.start_idx, args.end_idx, args.max_samples)

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

    scored = [r["approx_correct"] for r in results if r.get("approx_correct") is not None]
    if scored:
        print(f"\nApprox correctness (word-overlap heuristic, NOT authoritative): "
              f"{sum(scored)}/{len(scored)} = {sum(scored)/len(scored):.2%}")
    print(f"Output -> {output_file}")


if __name__ == "__main__":
    main()