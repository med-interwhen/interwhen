# `vita/scripts`

Offline helper scripts. Run them all from the **repo root**. Each command below
shows only the required args; every flag is documented underneath and is optional
unless marked **required**.

---

### `preresolve_dates.py`

Rewrites relative date phrases ("next Saturday", "the 1st of next month") in OTA
task instructions into absolute dates with an LLM, writing
`data/vita/domains/ota/resolved_instructions_<model>.json`.

```bash
python src/vita/scripts/preresolve_dates.py --model <model>
```

- `--tasks-file` — input tasks JSON. Default `data/vita/domains/ota/tasks_en.json`.
- `--output` / `-o` — output path. Default derived from model: `resolved_instructions_<model>.json`.
- `--model` — **required**; LLM used for resolution.
- `--language` — `english` or `chinese`. Default `english`.
- `--task-ids` — only resolve these task IDs (space-separated). Default: all tasks.
- `--num-tasks` — only resolve the first N tasks. Default: all tasks.
- `--max-concurrency` — parallel workers. Default `16`.
- `--resume` — flag; skip tasks already present in the output file.

---

### `preextract_completeness.py`

Extracts per-task completeness constraints (booking counts, cities, routes, dates)
that the completeness checker loads at runtime, writing
`data/vita/domains/ota/completeness_<model>.json`.

```bash
python src/vita/scripts/preextract_completeness.py --model <model>
```

- `--tasks-file` — input tasks JSON. Default `data/vita/domains/ota/tasks_en.json`.
- `--output` / `-o` — output path. Default `data/vita/domains/ota/completeness_<model>.json`.
- `--model` — **required**; LLM used for extraction.
- `--language` — `english` or `chinese`. Default `english`.
- `--task-ids` — only extract these task IDs (space-separated). Default: all tasks.
- `--num-tasks` — only extract the first N tasks. Default: all tasks.
- `--max-concurrency` — parallel workers. Default `1`.
- `--resume` — flag; skip tasks already present in the output file.

---

### `preextract_constraints_harness.py`

Extracts the natural-language constraints used by the soundness judge harness from
the resolved user-sim messages, writing
`data/vita/domains/ota/harness_constraints_<model>.json`.

```bash
python src/vita/scripts/preextract_constraints_harness.py --model <model>
```

- `--tasks-file` — input tasks JSON. Default `data/vita/domains/ota/tasks_en.json`.
- `--user-sim-file` — user-sim messages keyed by task_id (use a date-resolved one). Default `data/user_sim_claude-opus-4.6_ota_english_resolved.json`.
- `--output` / `-o` — output path. Default `data/vita/domains/ota/harness_constraints_<model>.json`.
- `--model` — **required**; LLM used for extraction.
- `--language` — `english` or `chinese`. Default `english`.
- `--task-ids` — only extract these task IDs (space-separated). Default: all tasks.
- `--num-tasks` — only extract the first N tasks. Default: all tasks.
- `--max-concurrency` — parallel workers. Default `1`.
- `--resume` — flag; skip tasks already present in the output file.

---

### `pregenerate_solo_messages.py`

Pre-generates the opening user message for solo-agent runs, writing
`data/user_sim_<model>_<domain>_<language>.json` (pass it to a run via
`--solo-user-mode=file --solo-user-file=<path>`).

Depends on `preresolve_dates.py`: pass that script's output via
`--resolved-instructions` so the generated messages are built from absolute-date
instructions instead of the original relative ("next Saturday") phrasing. Run
`preresolve_dates.py` first, then point this script at its JSON.

```bash
python -m vita.scripts.pregenerate_solo_messages --domain ota --llm <model>
```

- `--domain` — **required**; domain name (`ota`, `delivery`, `instore`).
- `--llm` — **required**; LLM model used for generation.
- `--language` — `english` or `chinese`. Default `english`.
- `--resolved-instructions` — path to `preresolve_dates.py` output; uses absolute-date instructions. Default: none (uses original instructions).
- `--task-ids` — only generate for these task IDs (space-separated). Default: all tasks.
- `--max-concurrency` — parallel workers. Default `1`.
- `--force` — flag; regenerate entries already present in the output file.
