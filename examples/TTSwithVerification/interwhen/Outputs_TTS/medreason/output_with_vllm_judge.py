"""
evaluate_with_vllm_judge.py

Reads outputs.jsonl, extracts the predicted answer (Selected Option + Selected Text)
from each record's [FINAL ANSWER] block, then asks a vLLM-hosted Qwen 30B model
to judge whether the prediction matches the ground truth.

Usage:
    python evaluate_with_vllm_judge.py --input outputs.jsonl --output judge_results.jsonl --base-url http://localhost:8000/v1 --model Qwen/Qwen3-30B-A3B-Thinking-2507 --limit 100

vLLM server (example):
    vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 \
        --tensor-parallel-size 4 \
        --max-model-len 4096
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_final_answer(output_text: str) -> tuple[str | None, str | None]:
    """
    Pull Selected Option (letter) and Selected Text from the real [FINAL ANSWER]
    block – i.e. the one that appears *after* the </think> scratchpad.
    Returns (option_letter, selected_text) or (None, None) if not found.
    """
    # Remove the <think>…</think> scratchpad so we only see the structured output.
    clean = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()

    # Grab the *last* [FINAL ANSWER] block (model occasionally drafts extras inside think).
    fa_blocks = list(
        re.finditer(r"\[FINAL ANSWER\](.*?)(?:\[/FINAL ANSWER\]|\Z)", clean, re.DOTALL)
    )
    if not fa_blocks:
        return None, None

    fa_text = fa_blocks[-1].group(1).strip()

    opt_match = re.search(r"Selected Option:\s*([A-Ea-e])", fa_text)
    txt_match = re.search(r"Selected Text:\s*(.+)", fa_text)

    option = opt_match.group(1).upper() if opt_match else None
    selected_text = txt_match.group(1).strip() if txt_match else None

    # Clean trailing junk from selected_text (e.g. "Sulphuric acid, Reasoning: ...")
    if selected_text:
        selected_text = re.split(r",\s*Reasoning:", selected_text)[0].strip()

    return option, selected_text


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a strict medical MCQ answer judge.
You will be given:
  - A medical question
  - The correct ground-truth answer (option letter + text)
  - The model's predicted answer (option letter + text)

Your job is to determine whether the predicted answer is CORRECT or INCORRECT.

Rules:
1. If the predicted option letter matches the ground-truth letter → CORRECT.
2. If the predicted option letter is missing/unclear, fall back to semantic
   comparison of the text: if the predicted text and ground-truth text refer
   to the same medical concept → CORRECT, otherwise → INCORRECT.
3. Minor spelling/capitalisation differences are acceptable if they clearly
   refer to the same answer.
4. Respond ONLY in the following JSON format (no extra text):
{
  "judgment": "CORRECT" | "INCORRECT",
  "reason": "<one sentence explanation>"
}"""


def build_user_message(question: str, ground_truth_option: str,
                       pred_option: str | None, pred_text: str | None) -> str:
    pred_str = ""
    if pred_option:
        pred_str += f"Option letter: {pred_option}"
    if pred_text:
        pred_str += f"\nOption text: {pred_text}"
    if not pred_str:
        pred_str = "No answer extracted."

    return (
        f"Question: {question}\n\n"
        f"Ground-truth answer option letter: {ground_truth_option}\n\n"
        f"Model's predicted answer:\n{pred_str}"
    )


# ---------------------------------------------------------------------------
# vLLM call
# ---------------------------------------------------------------------------

def call_vllm(
    base_url: str,
    model: str,
    question: str,
    ground_truth: str,
    pred_option: str | None,
    pred_text: str | None,
    timeout: int = 60,
    max_retries: int = 3,
) -> dict:
    """Call the vLLM OpenAI-compatible /v1/chat/completions endpoint."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(
                question, ground_truth, pred_option, pred_text
            )},
        ],
        "max_tokens": 200,
        "temperature": 0.0,
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            # Strip markdown fences if present
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

            parsed = json.loads(content)
            return {
                "judgment": parsed.get("judgment", "PARSE_ERROR"),
                "reason": parsed.get("reason", ""),
                "raw": content,
                "error": None,
            }
        except requests.exceptions.Timeout:
            if attempt == max_retries:
                return {"judgment": "ERROR", "reason": "timeout", "raw": "", "error": "timeout"}
            time.sleep(2 ** attempt)
        except json.JSONDecodeError as e:
            return {"judgment": "PARSE_ERROR", "reason": str(e), "raw": content, "error": str(e)}
        except Exception as e:
            if attempt == max_retries:
                return {"judgment": "ERROR", "reason": str(e), "raw": "", "error": str(e)}
            time.sleep(2 ** attempt)

    return {"judgment": "ERROR", "reason": "max retries exceeded", "raw": "", "error": "max retries"}


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

@dataclass
class Result:
    sample_id: str
    question: str
    ground_truth: str
    pred_option: str | None
    pred_text: str | None
    existing_correct: bool          # label already in the file
    existing_exact_matched: bool
    judge_judgment: str             # CORRECT / INCORRECT / ERROR / PARSE_ERROR
    judge_reason: str
    judge_agrees_with_existing: bool | None
    error: str | None


def process_record(record: dict, base_url: str, model: str) -> Result:
    sample_id = record["sample_id"]
    question = record["question"]
    ground_truth = record["ground_truth"]
    existing_correct = record.get("correct", False)
    existing_exact = record.get("exact_matched", False)

    pred_option, pred_text = extract_final_answer(record["output_text"])

    judge = call_vllm(base_url, model, question, ground_truth, pred_option, pred_text)

    judgment = judge["judgment"]
    judge_correct = judgment == "CORRECT" if judgment in ("CORRECT", "INCORRECT") else None
    agrees = (judge_correct == existing_correct) if judge_correct is not None else None

    return Result(
        sample_id=sample_id,
        question=question,
        ground_truth=ground_truth,
        pred_option=pred_option,
        pred_text=pred_text,
        existing_correct=existing_correct,
        existing_exact_matched=existing_exact,
        judge_judgment=judgment,
        judge_reason=judge["reason"],
        judge_agrees_with_existing=agrees,
        error=judge["error"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Judge predicted answers with vLLM Qwen.")
    parser.add_argument("--input",       default="outputs.jsonl",       help="Input JSONL file")
    parser.add_argument("--output",      default="judge_results.jsonl", help="Output JSONL file")
    parser.add_argument("--base-url",    default="http://localhost:8000/v1",
                        help="vLLM OpenAI-compatible base URL")
    parser.add_argument("--model",       default="Qwen/Qwen3-30B-A3B-Thinking-2507",
                        help="Model name as registered in vLLM")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Parallel threads (tune to your vLLM throughput)")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Only process first N records (for testing)")
    parser.add_argument("--timeout",     type=int, default=60,
                        help="HTTP timeout per request (seconds)")
    args = parser.parse_args()

    # Load records
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[: args.limit]

    total = len(records)
    print(f"Loaded {total} records from {args.input}")
    print(f"Model : {args.model}  |  vLLM : {args.base_url}")
    print(f"Workers: {args.max_workers}  |  Timeout: {args.timeout}s\n")

    results: list[Result] = [None] * total
    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_to_idx = {
            pool.submit(process_record, rec, args.base_url, args.model): i
            for i, rec in enumerate(records)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                rec = records[idx]
                results[idx] = Result(
                    sample_id=rec["sample_id"],
                    question=rec["question"],
                    ground_truth=rec["ground_truth"],
                    pred_option=None, pred_text=None,
                    existing_correct=rec.get("correct", False),
                    existing_exact_matched=rec.get("exact_matched", False),
                    judge_judgment="ERROR",
                    judge_reason=str(e),
                    judge_agrees_with_existing=None,
                    error=str(e),
                )

            done += 1
            if results[idx].error:
                errors += 1

            # Progress
            if done % 50 == 0 or done == total:
                print(f"  [{done}/{total}]  errors so far: {errors}", flush=True)

    # Write output
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

    # Summary stats
    correct_count   = sum(1 for r in results if r.judge_judgment == "CORRECT")
    incorrect_count = sum(1 for r in results if r.judge_judgment == "INCORRECT")
    error_count     = sum(1 for r in results if r.judge_judgment in ("ERROR", "PARSE_ERROR"))
    agree_count     = sum(1 for r in results if r.judge_agrees_with_existing is True)
    judged          = correct_count + incorrect_count

    print(f"\n{'='*50}")
    print(f"Total records  : {total}")
    print(f"CORRECT        : {correct_count}  ({correct_count/total*100:.1f}%)")
    print(f"INCORRECT      : {incorrect_count}  ({incorrect_count/total*100:.1f}%)")
    print(f"ERROR/PARSE    : {error_count}")
    if judged:
        print(f"Judge accuracy vs existing 'correct' field: "
              f"{agree_count}/{judged} ({agree_count/judged*100:.1f}%)")
    print(f"\nResults written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()