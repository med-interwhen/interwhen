# Policy → Lean 4 Spec + Executable Checker + Soundness Proofs

You are an expert in Lean 4 and formal verification. You will be given 2 files:  `policy.md`, and `tech_support_workflow.md` files describing the rules, data models, workflows, and constraints for an AI agent in the telecom domain .  The main policy file is policy.md. Focus first on this. You will also be given a `tools.py` listing the runtime tools the agent can actually call. **Read `tools.py` to learn the tool surface — do not `import` it from Lean (Lean cannot import Python).**

Your task: produce a **single, self-contained Lean 4 file** `PolicyChecker.lean` that — for every rule, constraint, workflow step, and invariant in the policies — contains a **triplet**.

1. A **Prop-level specification** (`spec_<name> : ... → Prop`) capturing the rule declaratively.
2. An **executable `Bool` checker** (`check_<name> : ... → Bool`) that decides the rule.
3. A **soundness/equivalence proof** (`theorem check_<name>_iff : check_<name> ... = true ↔ spec_<name> ...`) that the checker faithfully implements the spec, plus a `Decidable` instance derived from it.

Plus a **feedback string** for every rule (see §F).

Follow exactly the pattern in this sample (the canonical shape every triplet must follow):

```lean
import Mathlib

inductive RefundMethod
  | originalPayment | giftCard | differentPayment
  deriving DecidableEq, Repr

structure RefundRequest where
  amount : ℕ
  method : RefundMethod

-- Policy §"Refund Methods": "Refunds may be issued to the original payment method or as a gift card."
-- [source: args] [phase: pre] [tool: process_refund]
def validMethod (m : RefundMethod) : Prop :=
  m = RefundMethod.originalPayment ∨ m = RefundMethod.giftCard

def validMethodBool (m : RefundMethod) : Bool :=
  match m with
  | .originalPayment  => true
  | .giftCard         => true
  | .differentPayment => false

theorem validMethodBool_iff (m : RefundMethod) :
    validMethodBool m = true ↔ validMethod m := by
  cases m <;> simp [validMethodBool, validMethod]

instance (m : RefundMethod) : Decidable (validMethod m) :=
  decidable_of_iff _ (validMethodBool_iff m)

def feedback_validMethod (m : RefundMethod) : String :=
  s!"Refund method '{repr m}' is not allowed; must be original payment or gift card."
```

## Requirements

### A. Coverage & traceability
- **Encode every rule.** Every sentence in the policy describing a constraint, condition, workflow step, or data field must appear as a triplet (or as a data model used by one). If unsure whether something is a rule, encode it.
- **Encode implicit rules.** E.g. "lookup by phone, ID, or name+DOB" implies lookup-by-name-alone is forbidden — encode that.
- **No invented rules.** Only encode what the policy states or clearly implies.
- **Traceability comments.** Above every definition put a comment in this exact form:
  ```
  -- Policy §"Section Name": "quoted relevant text"
  -- [source: <db|args|history|context|args+history|...>] [phase: <pre|post>] [tool: <tool_name or N/A>]
  ```

### B. Data models
- Define data models (`structure`, `inductive`, etc.) for every entity the policy mentions (customer, bill, line, order, etc.).
- Define an `AgentState` structure aggregating the database snapshot, identified customer, and a `history : List ToolCall` of prior tool calls.
- Derive `DecidableEq, Repr` where reasonable.
- **Do NOT invent boolean fields on data structures to model facts that come from free-form text** (e.g. "the user gave permission to make payments", "the user is traveling"). These are not stored anywhere in the runtime. Instead, model them as **opaque hypothesis inputs** to the spec, e.g.:
  ```lean
  def spec_canMakePayment (s : AgentState) (b : BillId) (hPerm : UserGrantedPaymentPermission) : Prop := ...
  ```
  The Python translator will discharge `hPerm` either with an explicit kwarg or an SLM extraction call. The Lean side must remain agnostic about *how* it is discharged.

### C. Atomic decomposition (CRITICAL for Python translatability)
Each atomic policy clause must become its **own** triplet. Do **NOT** bundle independent failure modes into one checker.

- If a rule combines N independent facts (e.g. "bill must be overdue AND status was checked AND no other bill is awaiting"), write the N atomic triplets first, then a composite triplet that calls them. Never just the composite.
- A "failure mode" is independent if a Python rule could meaningfully fire on it alone with its own feedback string.
- Composite checkers must collect **all** failed sub-rules into a list of feedback strings — do not short-circuit on the first failure.

### D. Information-source and phase tagging (CRITICAL)
Every checker's traceability comment must include:

- `[source: ...]` — which inputs the checker actually inspects:
  - `db` — only the current database/state snapshot (e.g. bill status, line owner)
  - `args` — only the tool call arguments (e.g. `gb_amount ≤ 2`)
  - `history` — needs the list of prior tool calls (e.g. "must have called `check_bill_status` before `send_payment_request`")
  - `context` — needs free-text from the conversation/ticket (e.g. "is this a valid suspension reason?", "is the user traveling?")
  - Combine with `+` if multiple, e.g. `[source: db+history]`
- `[phase: pre|post]`:
  - `pre` — must hold *before* the tool executes; failure denies the call
  - `post` — must hold *after* a tool executes; failure injects a warning back to the agent (e.g. "after `resume_line`, the user must reboot their device")

### E. Tool surface and vacuous rules
You will be given a Python file (`tools.py`) listing available tools. **Every checker must declare which tool it gates** (or that it gates none).

- If the rule applies to a tool that exists in `tools.py`: tag with `[tool: <tool_name>]`.
- If the rule logically requires a tool that **does not exist** in `tools.py`: still write the full triplet, but tag it `[tool: N/A — would gate <missing_tool>]` and add a `-- TODO: vacuous in current runtime` comment. Do not silently encode rules that have no runtime hook.
- Each checker must take **only the inputs it directly inspects** as arguments. Do not pass the whole `AgentState` if the rule only needs one bill.

### F. Feedback strings
Alongside each `check_X`, define:
```lean
def feedback_X : <same inputs as check_X> → String
```
returning the human-readable failure reason (the message the agent will see when the check fails). Composite checkers' feedback returns `List String` of all failed sub-rule feedbacks.

### G. (omitted)

Do **not** define an `Action` inductive, `checkAction`, `applyAction`, `specCompliant`, or any aggregate soundness theorem. The runner dispatches per-tool using `manifest.json → rules`; aggregate Lean machinery is unused at runtime and only adds proof burden. Stop after the per-rule triplets and feedback strings.

### H. Style
- Single file, starts with `import Mathlib`.
- Pure and deterministic: no `IO`, no `unsafe`, no `native_decide`.
- All checks computable via `DecidableEq` / `BEq`.
- Prefer `simp`, `decide`, `omega`, `cases`, `match`, `rcases` in proofs.
- If a proof is genuinely intractable, mark with `sorry` and a `-- TODO:` — but do **not** stub out the `Bool` function, the spec, or the feedback string.
- Naming convention: `check_X` / `spec_X` / `feedback_X` for `:pre` rules; `check_result_X` / `spec_result_X` / `feedback_result_X` for `:post` rules. The Python translator will grep on these prefixes.

## Incremental build workflow (MANDATORY)

Proceed **sequentially, one triplet at a time**. Do not write the whole file in one shot. The workflow per rule:

1. Add the next `(spec_X, check_X, check_X_iff, decidability instance, feedback_X)` triplet (plus any data model it needs) to `PolicyChecker.lean`.
2. Run `lake build` in `{folder}`.
3. If it compiles cleanly → move to the next rule.
4. If it fails → debug and fix (adjust the proof, the Bool, or the data model — never weaken the spec to dodge the rule) and re-run `lake build`. Repeat until green before moving on.
5. When every rule is covered and the file builds clean, you are done. Do **not** add aggregate machinery (`Action`, `checkAction`, `specCompliant`, `checkAction_sound`).

Never advance to the next rule while the file is broken.

## Do NOT
- Do not split into multiple files — everything goes in `PolicyChecker.lean`.
- Do not `import` `tools.py` from Lean (Lean cannot import Python). You should still **read** `tools.py` to know which tools exist.
- Do not skip the Bool checker, the bridge proof, or the feedback string for any rule.
- Do not invent rules absent from the policy.
- Do not invent boolean data-model fields to model free-text facts (use opaque hypothesis inputs).
- Do not bundle independent atomic rules into one composite checker without first writing the atomic triplets.
- Do not use `native_decide` or unjustified `Decidable.decide` on nontrivial Props.
- Do not remove or shorten traceability comments — including the `[source:]`, `[phase:]`, `[tool:]` tags.
