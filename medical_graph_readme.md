# MedReason graph verification layer — what's here and how to run it

## What changed, in one paragraph

Your original 5 files are unchanged in behavior unless you opt in, with one
exception: `step_extractor` in `medical_monitor.py` now returns a bare
`bool` instead of a `(bool, str|None)` tuple, because that's a real bug
against your actual `VerifyMonitor` ABC (see point 3 below) — confirmed by
instantiating `MedicalMonitor` against the real ABC, not guessed at. A new
module, `medical_graph.py`, implements the three-component architecture
from the design discussion (LLM entity-to-CUI mapper → chunk graph builder →
global graph accumulator) plus a `StructuredDiff` feedback format. A new
`GraphAwareVerifier` in `medical_verifier_graph.py` subclasses your existing
`MedicalReasoningVerifierSnomedFirst` and layers the graph checks on top —
it runs every existing check first, unchanged, and can only *downgrade* an
otherwise-passing section, never rescue a failing one. `MedicalMonitor` gets
one new constructor flag, `run_graph_checks` (default `False`), that swaps
in `GraphAwareVerifier`. Nothing else changes.

## File-by-file map

```
interwhen/
  utils/
    medical_prompts.py            unchanged, +1 new prompt builder call site
                                   (build_entity_cui_mapping_prompt added to
                                   medical_reasoning_prompts.py, not this file)
    medical_reasoning_prompts.py  unchanged + build_entity_cui_mapping_prompt()
    medical_verifier.py           UNCHANGED — your original file, verbatim
    medical_verifier_snomed.py    UNCHANGED — your original file, verbatim
    medical_graph.py              NEW — the whole graph pipeline
    medical_verifier_graph.py     NEW — GraphAwareVerifier (wires graph into verifier)
  monitors/
    base.py                       STUB — you did not provide this file; see below
    medical_monitor.py            +2 constructor params (default off) + step_extractor
                                   bugfix (tuple -> bare bool, see below)
run_medreason_pipeline.py         +2 CLI flags, both optional, default off
tests/
  test_medical_graph.py           20 assertions, all passing, no network needed
  test_medical_verifier_graph.py  6 assertions, all passing, no network needed
```

## What I could and couldn't verify in this environment

I have no network egress here, so I could not hit a real Snowstorm or
BioPortal endpoint. Two consequences:

1. **`SnomedRelationshipClient`'s endpoint paths are unverified against a
   live instance.** They follow Snowstorm's documented REST shape
   (`/concepts/{id}`, `/concepts/{id}/relationships`), but I flagged this as
   the highest-risk dependency before writing any code, and that risk is
   still open. **Before relying on this for anything real, hit those two
   endpoints by hand against your actual Snowstorm instance** (or whichever
   SNOMED relationship API you have licensed access to) and confirm the
   response shape matches what `validate_cui()` / `get_inferred_relationships()`
   expect. If it doesn't, those two methods are the only places that need
   to change — everything downstream (chunk graph, global graph, structured
   diff, verifier wiring) is written against their return contract, not
   their implementation.

2. **I could not run the real pipeline end-to-end** (`run_medreason_pipeline.py`
   needs `interwhen.stream_completion` and `monitors/base.py`, neither of
   which you pasted, plus live vLLM solver/verifier servers). What I *did*
   verify, with fakes standing in for the LLM and SNOMED clients:
   - confidence-floor gating, CUI validation gating, and that a fabricated
     CUI is rejected even at high mapper confidence
   - SNOMED-unreachable degrades to "unverifiable," never to a false
     confirmation or an exception
   - edge confirmation and disconfirmation against a known relationship set
   - **the actual headline capability**: two individually-plausible but
     mutually contradictory `[INFERENCE]` claims (e.g. "DrugX treats
     ConditionY" then later "DrugX is contraindicated with ConditionY") —
     the first passes, the second is caught and downgraded to FAIL through
     `GraphAwareVerifier`'s real public interface, with a structured
     feedback message
   - a base-verifier FAIL is never overridden to PASS by the graph layer
   - `run_graph_checks=False` fully disables all of the above, confirming
     the rest of your pipeline is undisturbed

   Run them yourself:
   ```bash
   cd medreason
   python3 tests/test_medical_graph.py
   python3 tests/test_medical_verifier_graph.py
   ```
   Both should print `RESULT: N passed, 0 failed`.

3. **`monitors/base.py` is a stub I wrote, now verified against the real
   file from your fork.** You pasted the actual `interwhen/monitors/base.py`
   from `med-interwhen/interwhen` (`structured_reasoning` branch), and I
   tested compatibility for real — not by eyeballing signatures, but by
   swapping in the real `VerifyMonitor` ABC and confirming `MedicalMonitor`
   actually instantiates against it (`inspect.isabstract()` returns `False`,
   `__abstractmethods__` is empty).

   **One real bug was found and fixed in the process, present since the
   original code you pasted at the start of this conversation (not
   introduced by anything I added):** `step_extractor` returned a
   `(bool, str|None)` 2-tuple, but the real ABC's docstring specifies a
   bare `bool` return. This matters because a non-empty tuple is always
   truthy in Python regardless of its first element — `bool((False, None))`
   is `True`. If the real `stream_completion` harness does anything like
   `if monitor.step_extractor(...):`, the original code would have
   triggered verification on *every* chunk instead of only on section close
   tags. `step_extractor` now returns a bare `bool`, matching the real
   contract exactly. `verify()` and `fix()` already matched the real ABC's
   signatures with no changes needed.

   The stub `base.py` in this package now mirrors the real ABC's
   `__init__`/`verify`/`fix`/`step_extractor` signatures exactly (the real
   file also defines an unrelated `Monitor` ABC and imports
   `pydantic.BaseModel`, neither of which `VerifyMonitor` or
   `MedicalMonitor` actually use, so both are omitted from the stub to
   avoid an unnecessary dependency). **Replace this file with your real
   `base.py` before running** — at that point the only difference will be
   the harmless unused `Monitor`/`pydantic` content, which doesn't affect
   `MedicalMonitor` at all.

## How to run it for real

### 1. Confirm Snowstorm access first (do this before anything else)

```bash
export SNOWSTORM_BASE_URL="https://your-snowstorm-instance/snomed-ct"  # or leave unset for the public IHTSDO browser
curl "$SNOWSTORM_BASE_URL/MAIN/concepts/372567009"
curl "$SNOWSTORM_BASE_URL/MAIN/concepts/372567009/relationships?active=true&characteristicType=INFERRED_RELATIONSHIP"
```
If these don't return what `validate_cui()` / `get_inferred_relationships()`
in `medical_graph.py` expect (see docstrings), fix those two methods before
trusting any graph output. Everything else is written to degrade safely
even if this step fails — but "degrades safely" means "checks nothing,"
not "checks correctly."

### 2. Run without the graph layer (identical to your original behavior)

```bash
python3 run_medreason_pipeline.py \
  --solver_lm <your-solver-model> \
  --monitor \
  --max_samples 20
```

### 3. Run with the graph layer enabled

```bash
python3 run_medreason_pipeline.py \
  --solver_lm <your-solver-model> \
  --monitor \
  --run_graph_checks \
  --snowstorm_base_url "$SNOWSTORM_BASE_URL" \
  --max_samples 20
```

Each result in `outputs.jsonl` gains a `graph_summary` field:
```json
{
  "node_count": 7,
  "edge_count": 5,
  "contradiction_count": 1,
  "contradictions": [{"source": "...", "target": "...", "description": "..."}]
}
```

The run summary at the end prints the ablation metric you actually want:
```
Graph verification (--run_graph_checks):
  Samples with >=1 cross-chunk contradiction caught: 3/20
  Total contradictions caught across all samples:    4
```
That ratio — cases the graph layer caught that the per-section verifier let
through — is the empirical result the original design discussion was
aiming at.

### 4. Unit/integration tests (no servers needed)

```bash
python3 tests/test_medical_graph.py
python3 tests/test_medical_verifier_graph.py
```

## Things I deliberately did NOT build

- **A separate "local coherence" check** inside the chunk graph. A 2-3 node
  graph from one section has nothing to be locally incoherent with except
  SNOMED itself, so that step and the SNOMED edge check are the same step
  in `ChunkGraphBuilder.build()`.
- **Trusting a low-confidence or unvalidated CUI as a merge key**, ever.
  `MappedEntity.grounded` is only `True` after both the confidence floor
  and SNOMED validation pass; ungrounded entities use a
  `unverified::<span>`-namespaced node id that can never collide with a
  *different* unverified entity, by construction.
- **Graph checks enforced on `[CONCLUSION]`** by default
  (`GraphVerifierConfig.graph_enforced_sections` is `("INFERENCE",
  "OPTION_COMPARISON")`). Downgrading the final answer based on the
  accumulated graph is a stronger and riskier claim than catching a
  contradiction earlier in the trace; it's wired and available
  (`_verify_conclusion` calls the same check) but off until you decide you
  want it on.
- **A NER-to-CUI middleware that auto-promotes guesses.** If the mapper
  isn't confident and SNOMED can't confirm it, the entity stays ungrounded
  rather than being silently included with a best-effort CUI — this was the
  single point of failure flagged before any code was written, and it's the
  one constraint that shaped every other design choice in `medical_graph.py`.
