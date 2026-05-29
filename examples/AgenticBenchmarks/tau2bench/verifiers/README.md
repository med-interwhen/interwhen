# Runtime verifier

This folder ships **two backend variants**. Pick one and copy it into the
upstream clone as `src/tau2/verifier/`:

| Variant | Folder | Backend | Needs |
|---|---|---|---|
| Lean + Python + SLM | `verifier_lean/` | Lean policy-checker binary + Python rules + SLM extractor | Prebuilt `policychecker_telecom` binary on `TAU2_LEAN_BINARY`, and an SLM endpoint on `SLM_API_BASE`. |
| Python-only | `verifier_python/` | Python rules + SLM extractor | Only the SLM endpoint on `SLM_API_BASE`. |

Copy whichever you want:

```bash
DST=/path/to/upstream/tau2-bench

# Lean variant (recommended)
rm -rf "$DST/src/tau2/verifier"
cp -r verifiers/verifier_lean "$DST/src/tau2/verifier"

# OR Python-only variant
rm -rf "$DST/src/tau2/verifier"
cp -r verifiers/verifier_python "$DST/src/tau2/verifier"
```

If you copied the Lean variant, build the Lean binary and then drop it in:

```bash
cd verifiers/verifier_lean
lake build
mkdir -p "$DST/bin"
cp policychecker_telecom "$DST/bin/policychecker_telecom"
chmod +x "$DST/bin/policychecker_telecom"
export TAU2_LEAN_BINARY="$DST/bin/policychecker_telecom"
```

(See the top-level README for the full overlay procedure and run command.)

---

The verifier sits between the agent and the environment. Every time the agent
emits a tool call, the orchestrator hands the call (plus a snapshot of the DB
state) to `PolicyVerifier`. The verifier decides whether to **allow**, **deny**
(returning a structured rejection back to the agent), or **annotate** the result
(POST-tool notice).

The decision is the OR of up to three sources running in parallel (which run
depends on which variant you installed):

1. **Lean policy checker** — `policychecker_telecom` binary, fed JSON over stdin
   (Lean variant only).
2. **Python rules** — `telecom_python_rules.py`, called from Python directly.
3. **SLM helper** — `slm_helper.slm_extract` calls a small LLM to pull
   structured fields out of free-form tool arguments when a rule needs them.

If `TAU2_VERIFIER=0`, the whole verifier is bypassed and the orchestrator
behaves like upstream tau2. With the Lean variant, leaving `TAU2_LEAN_BINARY`
unset also disables the Lean path and runs Python rules only — equivalent to
the python-only variant at runtime.

---

## File-by-file

### `verifier.py` — `PolicyVerifier`

The runtime entry point. Imported by the orchestrator. Public API:

- `__init__(db, domain)` — picks the domain-specific glue (`telecom_glue_spec`,
  `airline_glue_spec`, etc.) and policy spec.
- `check_pre(tool_name, tool_args)` — returns a list of verdicts for any rule
  that fires **before** the call executes.
- `check_post(tool_name, tool_args, tool_result)` — verdicts for rules that
  inspect the result.
- Stats counters are written to `TAU2_VERIFIER_STATS_DIR` on shutdown so you
  can profile per-rule fire rates afterwards.

Behavior is gated on:

- `TAU2_VERIFIER` (default `"1"`)
- `TAU2_LEAN_BINARY` (path to the Lean checker; if unset Lean is skipped and
  only Python rules run)
- `TAU2_USE_AUTO_GLUE` (when set, swaps in `telecom_glue_spec.py` ; the Lean verifiers will now be used)
- `TAU2_POLICY_TODAY` — the date the Lean policy treats as "today" (some
  bill-overdue / contract-end-date rules are date-sensitive)

### `telecom_glue_spec.py` — Lean glue

Hand-curated mapping `tool_call + DB snapshot → Lean check_all request`. Each
function in this file builds the JSON payload sent to the Lean binary:

```json
{"id":"...", "rule":"check_billOverdue", "tool_args":{...}, "db_snapshot":{...}}
```

The Lean checker returns `{"id":..., "ok":bool, "verdict":"..."}` which the
verifier converts into either a denial back to the agent or a POST notice.

### `telecom_policy_spec.py` — Curated Python rules. Used when user wants only python verifiers

### `telecom_python_rules.py` — Python pre/post rules, run after lean rules

Pure-Python checks that are easier or cheaper to express imperatively than in
Lean.

Several rules call `slm_helper.slm_extract(...)` to pull structured fields
(customer id, phone number, ticket type) out of free-form arguments.

### `slm_helper.py`

Function `slm_extract(prompt, schema, **kwargs)` that:

1. Reads `SLM_API_BASE` for the OpenAI-compatible endpoint.
2. Calls the SLM, asks for JSON conforming to `schema`.
3. Returns the parsed dict, or raises on malformed output.

### `policychecker_telecom` (binary)

Lean executable. Reads one JSON request per line on
stdin, writes one verdict per line on stdout. 
---

## Request / response shape

PRE request (built by `telecom_glue_spec.py`, sent to the Lean binary):

```json
{
  "id": "send_payment_request#42",
  "rule": "check_billOverdue",
  "tool_args": {"bill_id": "B1002"},
  "db_snapshot": {"bills": [...], "customers": [...]}
}
```

Lean response:

```json
{"id": "send_payment_request#42", "ok": false, "verdict": "Bill 'B1002' is not in OVERDUE status"}
```

The verifier wraps `ok=false` verdicts into a `[VERIFIER]` user message:

```
[VERIFIER] This tool call is NOT SAFE and must not be executed.
You MUST NOT retry this call.
Attempted call: send_payment_request({"bill_id": "B1002"})
Reasons:
  - Bill 'B1002' is not in OVERDUE status: ...
```

The agent reads this on its next turn and adjusts.
---
