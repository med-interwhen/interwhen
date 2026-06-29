# Running the OTA verifier on VitaBench

---

**Note:** The code provided in this folder is built on top of the original code for VitaBench, found at [https://github.com/meituan-longcat/vitabench](https://github.com/meituan-longcat/vitabench). In each file, we have mentioned the changes we have made, and the code we have used verbatim, relative to the same file in the original VitaBench repo.

## 1. Clone the upstream VitaBench repo

```bash
git clone https://github.com/meituan-longcat/vitabench.git
cd vitabench
```

This README and the files alongside it are an overlay on top of the upstream
`main` branch. The overlay keeps the upstream directory layout, so every file
lives at the same path it would occupy inside a VitaBench checkout.

## 2. Apply the overlay

Because the overlay mirrors the upstream layout, copy its `src/` tree straight
over your clone — files at matching paths are replaced, new files are added:

```bash
SRC=/path/to/this/overlay   # the directory containing this README
DST=/path/to/vitabench      # your upstream clone

cp -r "$SRC/src" "$DST/"    # merge the overlay sources into the clone
```

### Modified files

| Path | What changed |
|---|---|
| `src/vita/cli.py` | Adds the `--soundness-mode`, `--completeness-mode`, `--solo-user-mode` and `--solo-user-file` run flags. |
| `src/vita/data_model/simulation.py` | Adds the matching `RunConfig` fields (+ validation) and a `soundness_log` field on `SimulationRun`. |
| `src/vita/run.py` | Threads the new flags through the run pipeline, builds the OTA verifier, and resolves solo user messages. |
| `src/vita/orchestrator/orchestrator.py` | Runs the verifier inline: blocking soundness check before each tool call, and a completeness check on stop. |
| `src/vita/agent/llm_agent.py` | Solo agent honours `language`; relaxes the tool-call-only guard so the orchestrator can nudge instead of crashing. |
| `src/vita/user/user_simulator.py` | `DummyUser` can replay a pregenerated opening message instead of calling the LLM each run. |
| `src/vita/domains/ota/tools.py` | Adds an optional `override` flag to every OTA WRITE tool so the agent can bypass a soundness block when confident. |
| `src/vita/domains/ota/tools_schema.py` | Documents the new `override` argument (Chinese + English). |
| `src/vita/evaluator/evaluator_traj.py` | Flattens nested-list LLM rubric output before scoring. |
| `src/vita/utils/utils.py` | Hardens `evaluator_extracter` JSON extraction (think-block stripping, fenced/balanced-block fallback). |
| `src/vita/prompts/agent_system_prompt.yaml` | Adds an "always respond in English" instruction. |
| `src/vita/prompts/solo_agent_system_prompt.yaml` | Adds an "always respond in English" instruction. |

### New files

| Path | Purpose |
|---|---|
| `src/vita/domains/ota/verifier/` | `OTAVerifier` + `create_verifier()` factory that wires the soundness and completeness checks together. |
| `src/vita/domains/ota/soundness_judge_llm/` | LLM-judge soundness checker (`--soundness-mode llm`). |
| `src/vita/domains/ota/soundness_judge_harness/` | NL-constraint "harness" soundness checker with running memory (`--soundness-mode harness`). |
| `src/vita/domains/ota/completeness/` | Completeness checker that compares the final orders against extracted constraints at stop. |
| `src/vita/prompts/*.yaml` | New prompt templates: soundness/harness judges, constraint & completeness extraction, memory writer, and date resolution. |
| `src/vita/scripts/` | Offline preprocessing scripts and their guide — see [`src/vita/scripts/README.md`](src/vita/scripts/README.md). |

**Note on prompt language:** The prompt templates for our verifiers in this overlay come with **English prompts only**. For `--language chinese`, please add your own translations to:

 - `src/vita/prompts/completeness_extraction_template.yaml`
 - `src/vita/prompts/date_resolution_template.yaml`
 - `src/vita/prompts/harness_constraint_extraction_template.yaml`
 - `src/vita/prompts/harness_memory_writer_template.yaml`
 - `src/vita/prompts/harness_soundness_judge_template.yaml`
 - `src/vita/prompts/soundness_judge_template.yaml`

## 3. Python environment

Follow the upstream VitaBench README.

## 4. Offline preprocessing (optional)

Some verifier modes consume artifacts produced by the scripts in
`src/vita/scripts/` (resolved dates, extracted constraints, pregenerated solo
user messages). The dependency order and exact commands are documented in
[`src/vita/scripts/README.md`](src/vita/scripts/README.md). You only need these
if you run `--soundness-mode harness`, `--completeness-mode on`, or
`--solo-user-mode file`.

## 5. Environment variables

```bash
# Max times the agent is sent back after a failed completeness check (default 1)
export VITA_MAX_COMPLETENESS_RETRIES=1
```

## 6. Run

Reference command (OTA domain, solo agent, dummy user, harness soundness +
completeness checks on):

```bash
vita run \
  --domain ota \
  --agent llm_solo_agent \
  --user dummy_user \
  --agent-llm <model name> \
  --evaluator-llm <model name> \
  --language english \
  --soundness-mode harness \
  --completeness-mode on \
  --num-tasks 100
```

Flags (the four overlay flags are added by this overlay; the rest are upstream):

| Flag | Meaning |
|---|---|
| `--domain ota` | Run the OTA domain. The verifier only activates for `ota`. |
| `--agent llm_solo_agent` | Solo-mode agent: no user-simulator turn; it works the ticket autonomously via tool calls. |
| `--user dummy_user` | No-op user that only issues the opening message. |
| `--agent-llm <model name>` | Model (from `models.yaml`) the agent runs on. |
| `--evaluator-llm <model name>` | Model used by the rubric evaluator. |
| `--language english` | Prompt/task language (`english` or `chinese`). |
| `--num-tasks 100` | Number of tasks to run. |
| `--soundness-mode {llm,harness,off}` | Soundness checker before each write tool call. `llm` = LLM judge, `harness` = NL-constraint judge with memory, `off` = disabled. Default `off`. |
| `--completeness-mode {on,off}` | When `on`, run a completeness check at stop and send the agent back (up to `VITA_MAX_COMPLETENESS_RETRIES`) if requirements are unmet. Default `off`. |
| `--solo-user-mode {live,file}` | Solo opening message: `live` generates it via LLM each run (introduces variance); `file` loads a deterministic pregenerated message. Default `live`. |
| `--solo-user-file <path>` | JSON mapping `task_id -> message`, required when `--solo-user-mode=file`. Produced by `src/vita/scripts/pregenerate_solo_messages.py`. |

Results are written to `data/simulations/`. See the upstream README for the full
list of base flags (`--num-trials`, `--max-steps`, `--task-ids`, `--csv-output`,
…).
