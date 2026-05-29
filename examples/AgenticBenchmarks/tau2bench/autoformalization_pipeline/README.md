# Autoformalization pipeline

Generates the Lean policy checker (`policychecker_<domain>` binary) and the
Python glue (`<domain>_glue_spec.py`) directly from a domain's natural-
language policy, tool signatures, workflow doc, and DB schema. 

---

**Note:** While this pipeline works, some amount of human involvement is often needed to get the optimal set of verifiers.

## 1. Pipeline flow

```text
inputs (policy.md, tools.py, workflow.md, db_schema.py, [user_tools.py])
   │
   ▼
generate_spec.py        ── Copilot agent writes PolicyChecker.lean +
                                manifest.json
   │
   ▼
generate_runner.py           ── renders LeanMain.lean
   │
   ▼
lean_project.py + lake       ── builds the policychecker binary
   │
   ▼
generate_glue.py             ── renders <domain>_glue_spec_auto.py from the
                                manifest (PRE/POST specs + needs_hyp flags)
   │
   ▼
glue_runtime.py              ── serves check_all / check_all_results to the
                                verifier; performs runtime normalization,
                                identity bridging, and SLM hypothesis handling
   │
   ▼
PolicyVerifier (runtime)     ── calls glue, ships JSON to the binary, applies
                                optional Python fallback rules after Lean
```

## 2. Inputs

`spec_pipeline/cli.py` takes:

| Flag | Meaning |
|---|---|
| `--policy` | Markdown policy (e.g. `data/tau2/domains/telecom/main_policy_solo.md`) |
| `--tools` | Python file declaring the tool signatures (e.g. `src/tau2/domains/telecom/tools.py`) |
| `--workflow` | Workflow doc (e.g. `tech_support_workflow_solo.md`) |
| `--db-schema` | Domain DB Pydantic models (e.g. `src/tau2/domains/telecom/data_model.py`) |
| `--user-tools` | Optional user-side tools file. Added to auto inputs so generated rules can reason about which tools have actually been called. |
| `--out-dir` | Lean project output directory (e.g. `/tmp/policy_telecom`) |
| `--glue-out` | Generated glue file path (e.g. `src/tau2/verifier/telecom_glue_spec_auto.py`) |
| `-vv` | Verbose |

Reference Invocation:

```bash
python cli.py \
    --policy data/tau2/domains/telecom/main_policy_solo.md \
    --tools src/tau2/domains/telecom/tools.py \
    --workflow data/tau2/domains/telecom/tech_support_workflow_solo.md \
    --db-schema src/tau2/domains/telecom/data_model.py \
    --out-dir /tmp/policy_telecom \
    --glue-out src/tau2/verifier/telecom_glue_spec.py \
    -vv \
    --user-tools src/tau2/domains/telecom/user_tools.py
```

## 3. Pipeline stages and files

### 3.1 `cli.py` — orchestrator

End-to-end pipeline. Loads inputs, calls `generate_spec_auto`, then `generate_runner`, builds the Lean project via `lean_project`, then calls `generate_glue`.

### 3.2 `generate_spec.py` — Copilot Auto verifier generation

Single-shot LLM call that writes `PolicyChecker.lean` and a `manifest.json`
describing every rule it produced. The prompt template lives in
`prompts/auto.md`.


### 3.4 `generate_runner.py` — `LeanMain.lean`

Renders the Lean entry point that accepts JSON requests on stdin and writes
verdicts on stdout. Template lives in `templates/LeanMain.lean.j2`.

### 3.5 `lean_project.py` — Lake build

Creates a Lake project under `<out-dir>/`, drops `lakefile.toml`,
`lean-toolchain`, `LeanMain.lean`, and the generated `PolicyChecker.lean`, then
runs `lake build`. Produces `<out-dir>/policychecker`.

### 3.6 `generate_glue.py` — Python glue generation

Reads `manifest.json` and renders `<domain>_glue_spec.py` via the
`templates/glue_spec.py.j2` Jinja template. The generated file has:

- `cfg.pre_rules` — list of PRE rule specs (rule name, tool name, arg builder)
- `cfg.post_rules` — list of POST rule specs (rule name, tool name, hypothesis
  schema, `needs_hyp` flag)
- `check_all(tool_call, db_snapshot, ...)` — runtime entrypoint for PRE checks
- `check_all_results(tool_call, result_content, db_snapshot, ...)` — runtime
  entrypoint for POST checks

### 3.7 `glue_runtime.py` — runtime glue helpers

### 3.8 `manifest.py`, `copilot_client.py`

Plumbing: Pydantic model for the manifest, JSON parser for the Copilot
response, OpenAI-compatible client.

### 3.9 `templates/`

| Template | Renders |
|---|---|
| `LeanMain.lean.j2` | Lean stdin loop + dispatch over rule names |
| `glue_spec.py.j2` | Python glue module (`cfg`, `check_all`, `check_all_results`) |

## 4. Generated artifacts

A successful run produces:

```text
<out-dir>/
├── inputs/                       # snapshot of every input file
│   ├── policy.md
│   ├── tools.py
│   ├── workflow.md
│   ├── db_schema.py
│   └── user_tools.py             # if --user-tools was passed
├── PolicyChecker.lean            # Copilot AUTO output
├── LeanMain.lean                 # rendered entrypoint
├── lakefile.toml, lean-toolchain, lake-manifest.json
├── manifest.json                 # {domain, namespace, rules, stuck}
└── policychecker                 # built binary
```

Plus, at `--glue-out`:

```text
src/tau2/verifier/<domain>_glue_spec.py
```

## 5. Using the generated artifacts at runtime

```bash
export TAU2_LEAN_BINARY=<path_to_your_lean_verifier_binary>
export TAU2_USE_AUTO_GLUE=1                
export TAU2_POLICY_TODAY=2025-02-25 # for telecom
```

Also, copy the glue_runtime.py, {domain}_glue_spec.py, and {domain}_python_rules.py to the verifier folder in your tau2bench clone.