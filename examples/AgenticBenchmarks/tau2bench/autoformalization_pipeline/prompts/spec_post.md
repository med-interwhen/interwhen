## Hard constraints (read these first)

The Lean side must be self-contained. Python's only jobs are:
  (a) shipping raw, unprocessed inputs over JSON, and
  (b) optional SLM calls for free-text inference (these become opaque
      hypothesis Bools fed to Lean).

Specifically FORBIDDEN in this PR:
  - `-- Python preprocessor: …` comments. If Lean needs `exceeded`,
    Lean must compute `used >= limit + refueled` itself from a
    structured input.
  - `[normalised]` sentinel strings injected by Python glue.
  - Bool inputs that are really "Python already decided". The only
    legal Bool inputs are SLM-derived hypotheses (e.g. `travelling`).
  - Wrapping a single `containsSubstr` test as a "rule". A POST rule
    must either parse structured data, combine ≥2 conditions, or read
    `AgentState`.

## What Lean receives per POST call

Each `checkResult` invocation gets:
  - `state : AgentState` — the full slice you already ship for PRE,
    extended with `lastToolResults : RBMap String String` (raw output
    of every read tool called this session, keyed by tool name).
  - `tool : String` — the tool whose result we're checking.
  - `result : String` — the tool output, with at most global
    representation-only scalar canonicalisation applied symmetrically
    (for example, formatting-insensitive phone/date representations).
  - `hypotheses : Hyp` — opaque Bools from SLM (`travelling`,
    `userConfirmedRefuelPrice`, etc.). These are the ONLY pre-cooked
    inputs.

Python ships these without per-rule decisions. Python may apply global
or manifest/schema-declared scalar canonicalisation symmetrically before
shipping (for example, normalising representation-only formatting
differences), but it must not perform per-rule parsing, arithmetic, or
Bool derivation. Lean decides the rule from the shipped state, result,
and allowed hypotheses.

## Required Lean capabilities (extend `PolicyChecker.lean` as needed)

Add a small parsing layer in §C (utilities):
  - `parseJsonField (r : String) (key : String) : Option String`
  - `parseJsonNum (r : String) (key : String) : Option Float`
    (or rationals if you want exactness — Lean 4 has `Rat`)
  - `parseJsonBool (r : String) (key : String) : Option Bool`
  - `splitLines (r : String) : List String`
  - `lineMatching (r : String) (prefix : String) : Option String`

These are pure Lean and should have one-line `iff` lemmas
(`parseJsonNum_some_iff`, etc.) so downstream theorems compose.

Add `lastToolResult (s : AgentState) (tool : String) : Option String`
and a Decidable instance for "tool was called this session".

## POST triplet shape (no escape hatches)

Each rule:
  1. `def spec_result_X (s : AgentState) (r : String) (h : Hyp) : Prop`
     — written in terms of parsed fields, not substrings, where the
     output is structured. For free-text outputs, use the literal
     substring from `user_tools.py`.
  2. `def check_result_X (s : AgentState) (r : String) (h : Hyp) : Bool`
     — implements (1) using the parsing utilities. NO opaque Bool
     stand-ins for anything Python could compute.
  3. `theorem check_result_X_iff` — proves (2) iff (1). Must compose
     from the parsing-layer iff lemmas, not `decide` over a Bool input.
  4. `def feedback_result_X` — string from MD verbatim.

## Examples of what "non-trivial" means here

These are abstract shapes, NOT a recipe to transcribe. Read `policy.md`
and `tools.py` to discover which fields and rules actually exist in
*your* domain; the table below only teaches the structural pattern.

| Shape | Trivial form (FORBIDDEN) | Non-trivial form (REQUIRED) |
|---|---|---|
| **Numeric threshold from result** | `(exceeded : Bool) → ! exceeded` | parse two or more numeric JSON fields from `r` (e.g. a measured value, a limit, an optional adjustment); compute the inequality in Lean; iff theorem proves equivalence to spec written in `Float`/`Int`/`Rat` arithmetic |
| **Result field matches state** | `(matches : Bool) → matches` | parse a JSON field from `r`; look up the corresponding value from `state.<X>`; do equality in Lean; iff theorem proves it matches `state.<X>` |
| **Boolean flag in result conflicts with SLM hypothesis** | `(flagOff cond : Bool) → ! (flagOff && cond)` | parse a Bool JSON field from `r`; read the SLM-derived hypothesis from `Hyp`; spec is `¬(field = false ∧ hyp.cond)` |
| **Set/list membership in result** | `(missing : Bool) → ! missing` | parse the delimited list from `r` (`String.splitOn ","` etc.); check `"<required_item>" ∈ list`; iff theorem composes from list-membership decidability |
| **Multi-source coach** (substring trigger + structured cross-check) | substring `"<error phrase>"` alone | combine: substring trigger on `r` + look up an earlier tool result via `state.lastToolResults["<other_tool>"]` and check absence of a required token + compare a numeric field from yet another `lastToolResults` entry against a plan/limit on `state` — produces a structured `MissingHints` ADT, feedback is `formatHints` |

## What Python is still allowed to do

  - Ship `state.userPhone`, `state.customerPlan.dataLimitGb`,
    `state.lastToolResults`, `state.calledTools` over the wire.
  - Make ≤1 SLM call per task to populate `Hyp` (e.g. `travelling`).
  - Receive Lean's warning string and append it to the result.

No per-rule JSON parsing, arithmetic, string normalisation, or Bool
preprocessors in Python. Global representation-only canonicalisation is
allowed only when it is independent of the rule verdict and applied to
state/result fields by schema or field kind.

## SLM-derived hypotheses (the only legal "pre-cooked" inputs)

Add an opaque-prop layer (`Hyp` record) for facts that cannot be
extracted by parsing JSON or reading `AgentState` — typically things
like the user's stated intent, consent, or context that only appears
in free-text turns. Each `Hyp` field comes from one classify-time SLM
call. These are the ONLY Bools Lean receives that aren't computed
from structured fields.

Discover which hypotheses are needed by reading the policy: any
condition phrased as "the user must have <stated/agreed/confirmed/
acknowledged> X" is a candidate. Do NOT pre-populate the `Hyp` record
with hypotheses the policy doesn't actually require.

**Default-value rule for `Hyp` fields (MANDATORY).** Any `Hyp` field
referenced by a PRE rule MUST default to `true` in the `Hyp` structure
(innocent until contradicted); the SLM flips it to `false` only on
explicit contradicting evidence in the dialog. This prevents the
deadlock pattern where a PRE rule fires on every call because the SLM
is conservative. `Hyp` fields used only by POST rules may default to
`false` (since POST rules see the tool result and a false default
just means "no extra warning issued").

## Source of truth

`tools.py` and optional `user_tools.py` define callable tools, helper
tools, user/device-facing diagnostics, result formats, and side effects.
Read every provided tool source end-to-end and transcribe known output
contracts into the parsing-layer comments. When policy/workflow prose and
tool code disagree about output shape, the tool code wins.

## Deliverables

  1. Parsing utilities in §C with iff lemmas.
    2. `AgentState` extended with the session/result fields needed by the
      supported POST rules.
    3. Every POST triplet directly supported by policy/workflow plus a known
      result contract, each non-trivial per the table above.
  4. `checkResult` dispatcher in §G.
  5. Python glue: a single ~30-line function that serialises
     `AgentState` + `Hyp` + raw `result` and POSTs to Lean. No
     per-rule preprocessing.
