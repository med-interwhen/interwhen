# Running interwhen on Agent-SafetyBench
---

**Note:** The code provided builds on top of the existing code found in https://github.com/thu-coai/Agent-SafetyBench. In each file, we have mentioned the changes we made, and the code we used verbatim, compared to the same file in the original Agent-SafetyBench repo.

## 1. Clone the upstream repo

```bash
git clone https://github.com/thu-coai/Agent-SafetyBench.git
cd Agent-SafetyBench
```

## 2. Overlay the modified files

The modified files live alongside this README in the repo where
you got it. Their target paths in the upstream clone are shown below ; copy
each to its destination:

| Source (flat, next to this README) | Destination in upstream clone | Purpose |
|---|---|---|
| `__init__.py` | `evaluation/model_api/__init__.py` | Adds `VllmAPI` to the public exports |
| `VllmAPI.py` | `evaluation/model_api/VllmAPI.py` | Thin OpenAI-compatible client targeting a vLLM server |
| `eval.py` | `evaluation/eval.py` | Adds `--safety_rules`, `--env_rules`, `--rules_mode`, `--prompt_check`, `--vllm_host/port`, `--outdir`, and the `vllm-<served_name>[:<hf_id>]` `--model_name` dispatch. Built on top of the same file in the original repo |
| `safety_rules.py` | `evaluation/safety_rules.py` | `SafetyRuleEngine` — generates declarative pre/post rules per task and checks tool calls |
| `eval_with_shield.py` | `score/eval_with_shield.py` | Basic shield judge (single-judge). Mostly follows the original repo |
| `eval_with_shield_full.py` | `score/eval_with_shield_full.py` | Dual-judge eval: HF safety judge + OpenAI-compatible helpfulness judge |

Copy them in:

```bash
SRC=/interwhen/examples/AgenticBenchmarks/Agent-SafetyBench        
DST=/path/to/upstream/clone/Agent-SafetyBench

declare -A MAP=(
  [__init__.py]=evaluation/model_api/__init__.py
  [VllmAPI.py]=evaluation/model_api/VllmAPI.py
  [eval.py]=evaluation/eval.py
  [safety_rules.py]=evaluation/safety_rules.py
  [eval_with_shield.py]=score/eval_with_shield.py
  [eval_with_shield_full.py]=score/eval_with_shield_full.py
  [eval_with_shield_full2.py]=score/eval_with_shield_full2.py
)

for src in "${!MAP[@]}"; do
  dst="${MAP[$src]}"
  mkdir -p "$DST/$(dirname "$dst")"
  cp "$SRC/$src" "$DST/$dst"
done
```



## 3. Python environment

Follow the upstream README for base dependencies. In addition you need:

```bash
pip install openai transformers accelerate tqdm tabulate vllm tomli scikit-learn
```

`secrets.toml` at the repo root (only needed when a judge or model uses the real
OpenAI API):

```toml
openai_api_key = ""
```

## 4. Setting up model servers

### 4a. Agent model — vLLM (OpenAI-compatible)

Serve the agent under test as a vLLM endpoint. Example with
Qwen3-30B-A3B-Thinking-2507:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --served-model-name qwen3-30b-a3b-thinking-2507 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768
    --tool-call-parser hermes 
    --enable-auto-tool-choice 
```

Notes:
- `--served-model-name` is what `eval.py` sends as the `model=` field. It must
  match the string you put after `vllm-` in `--model_name`.

### 4b. Safety judge — HF (loaded in-process)

The safety judge uses local HF weights (no separate server). Pass
`--model_path` to the score scripts; transformers will load it directly.

### 4c. Helpfulness judge — OpenAI

Set `openai_api_key` in `secrets.toml`. Default model is `gpt-4o`. Override
with `--openai_model`. You can also use your local vllm server here.

## 5. Run an evaluation

```bash
cd Agent-SafetyBench/evaluation

CUDA_VISIBLE_DEVICES=0 python -u eval.py \
    --model_name vllm-qwen3-30b-a3b-thinking-2507:Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --vllm_host localhost --vllm_port 8000 \
    --greedy 1 \
    --regen_exceed 1 \
    --safety_rules \ # interwhen (policy+prompt)
    --env_rules \ # interwhen (policy)
    --rules_mode block \
    --extra_info ""
```

Results are written under
`./evaluation_results/<model_name>[_<output_suffix>]<extra_info>/`
unless `--outdir` is set explicitly.

### Flag reference (`eval.py`)

| Flag | Meaning |
|---|---|
| `--model_name` | Selects the agent backend. For vLLM use `vllm-<served_name>[:<hf_id>]`. The `<hf_id>` is used for the tokenizer (`AutoTokenizer.from_pretrained`) and defaults to `<served_name>` when omitted. |
| `--vllm_host`, `--vllm_port` | Where the vLLM OpenAI server is reachable. The client URL becomes `http://<host>:<port>/v1`. |
| `--greedy` | `1` → temperature 0 (greedy). `0` → sampling. |
| `--regen_exceed` | `1` → re-generate when the model produces over-long / malformed output, up to a small retry budget. |
| `--allow_empty` | `1` → accept empty assistant outputs without retry. |
| `--start`, `--end` | Slice the dataset by integer index range. |
| `--num_workers` | Concurrent in-flight examples (per-process). |
| `--safety_rules` | Enable `SafetyRuleEngine` — per-task declarative pre/post conditions checked against every tool call. This is the prompt+policy version of interwhen, where verifiers are created based on the policy (here, avoiding the failure modes) and the prompt. |
| `--env_rules` | Use environment-level (task-agnostic) rules instead of per-sample ones. Combine with `--safety_rules`. Equivalent to the pure policy variant of interwhen|
| `--rules_mode` | `monitor` = record violations only; `block` = reject unsafe calls and surface the rejection back to the agent. |
| `--prompt_check` | Only meaningful with `--env_rules`. Runs an LLM safety classifier on the user prompt and injects a warning the agent sees if the prompt is judged unsafe. |
| `--output_suffix` | Appended to the output dir name. |
| `--extra_info` | Appended to the output dir name (use `""` for none). |
| `--outdir` | Override the auto-named output dir entirely. |

### Output layout

```
evaluation_results/
└── vllm-qwen3-30b-a3b-thinking-2507/
    ├── <env>.json                # per-environment generation results
    ├── raw_env_rules/<env>.txt   # rules emitted by SafetyRuleEngine (if --safety_rules)
    └── ...
```

## 6. Score with the dual-judge shield

```bash
cd Agent-SafetyBench/score

python -u eval_with_shield_full.py \
    --model_path "$JUDGE_HF" \
    --model_base "$JUDGE_BASE" \
    --filepath ../evaluation/evaluation_results/vllm-qwen3-30b-a3b-thinking-2507 \
    --filename gen_res.json \
    --label_type "" \
    --target_model_name vllm-qwen3-30b-a3b-thinking-2507 \
    --judges safety,helpfulness \
    --openai_model qwen3-30b-a3b-thinking-2507 \
    --openai_base_url http://localhost:8000/v1 \
    --openai_api_key dummy \
    --shield_name qwen3-judge \
    --concurrency 32 \
    --batch_size 4
    --secrets_path ../secrets.toml \
```

### Flag reference (`eval_with_shield_full.py`)

| Flag | Meaning |
|---|---|
| `--model_path` | HF directory for the **safety** judge. Required if `safety` is in `--judges`. |
| `--model_base` | Chat-template family for the HF judge: `qwen` / `internlm` / `baichuan` / `chatglm`. |
| `--batch_size` | HF judge batch size. |
| `--filepath` | Directory containing the gen-results JSON to score. |
| `--filename` | The gen-results JSON filename (with `.json`). |
| `--label_type` | Free-form suffix used in output filenames; usually `""`. |
| `--target_model_name` | Identifier of the model being judged. Used in output paths and tables. |
| `--openai_model` | Model name for the **helpfulness** judge. Default `gpt-4o`. |
| `--openai_base_url` | Point at an OpenAI-compatible endpoint other than OpenAI (e.g. a vLLM server). |
| `--openai_api_key` | Override the API key (used with `--openai_base_url`). |
| `--secrets_path` | TOML file holding `openai_api_key`. Default `secrets.toml`. |
| `--concurrency` | Concurrent requests to the helpfulness judge. |
| `--judges` | Comma-separated subset of `{safety, helpfulness}`. |
| `--shield_name` | Override the directory name under `shield_results/`. Defaults to the HF judge basename, or the OpenAI model if no HF judge. |

Output:

```
score/shield_results/<shield_name>/
├── <target>_<filename>_<label_type>safety_results.json       # per-example safety judgement + stripped trace
├── <target>_<filename>_<label_type>helpfulness_results.json  # per-example helpfulness judgement
├── <target>_<filename>_<label_type>outputs_summary.json      # aggregate metrics (P/R/F1, fulfillable breakdown)
└── <target>_<filename>_<label_type>outputs_log.txt           # tabulated report
```

Each `*_results.json` example contains the original `output` trace and a
`scored_trajectory` field — the latter is the trace the judge actually saw, with
shield-injected scaffolding removed (see `_strip_blocked_calls` in
`eval_with_shield_full.py`).