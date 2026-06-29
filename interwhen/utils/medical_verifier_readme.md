# Medical Interwhen - Documentation

Inference-time verification for medical reasoning on the MedReason dataset.
Built on top of the interwhen framework (Microsoft).

---

## What it does

A solver LLM (e.g. Qwen3-30B) generates a chain-of-thought answer to a medical MCQ. A separate verifier LLM monitors the reasoning in real time, triggering every 15 non-empty lines. When the verifier finds a problem, it injects a `[FEEDBACK]` block into the stream. The solver reads the feedback and continues, potentially revising its answer. External evidence (PubMed abstracts or SNOMED CT definitions) is fetched to ground the feedback.

```
Solver LLM (streaming)
       │
       │ tokens
       ▼
   Monitor
   count non-empty lines
       │
       │ every 15 lines  (or on UNKNOWN)
       ▼
   Verifier LLM
   classify paragraph → OBSERVATION / INFERENCE / CONCLUSION / OTHER
       │
       ├── TRUE  → continue
       │
       ├── UNKNOWN → fetch SNOMED → re-verify → TRUE/FALSE
       │
       └── FALSE (confidence ≥ threshold)
               │
               ├── flip-flop guard (already corrected this topic? → skip)
               │
               ├── fetch evidence (PubMed / SNOMED / both / none)
               │
               └── inject [FEEDBACK] → solver continues
                   (max 10 feedback blocks per sample)
```

---

## File structure

```
interwhen/
├── interwhen/
│   ├── monitors/
│   │   └── medical_monitor.py           - VerifyMonitor subclass; trigger + inject logic
│   └── utils/
│       ├── medical_prompts.py           - Solver system prompt + user prompt template
│       ├── medical_reasoning_prompts.py - All verifier LLM prompt builders
│       ├── medical_verifier.py          - Core verifier: LocalVLLMClient, SnomedClient,
│       │                                  PubMedClient, VerifierConfig, CompactState,
│       │                                  MedicalPreprocessor, MedicalReasoningVerifier
│       └── medical_verifier_snomed.py   - MedicalReasoningVerifierSnomedFirst subclass
│                                          (real-time SNOMED enrichment per paragraph)
└── examples/
    └── TTSwithVerification/
        └── interwhen/
            └── medical_example.py - Dataset loader, scorer, multiprocessing runner
```

---

## Setup

### Requirements

```bash
pip install transformers datasets requests python-dotenv tqdm
```

### Environment variables

```bash
# Required only when evidence_source includes "snomed"
BIOPORTAL_API_KEY=your_key_here

# Optional - increases PubMed rate limit from 3 to 10 req/s
NCBI_API_KEY=your_key_here
```

### vLLM servers

Two separate vLLM servers are expected by default:

```bash
# Solver (port 8000)
vllm serve Qwen/Qwen3-30B --port 8000

# Verifier (port 8001) - any medical or general LLM
vllm serve Qwen/Qwen3-30B --port 8001
```

The verifier and solver can be the same model on the same port. Pass `--verifier_port 8000` and `--verifier_model Qwen/Qwen3-30B` to use the solver as its own verifier.

---

## Running the pipeline

### Basic (with monitor, PubMed evidence)

```bash
python medical_example.py \
    --solver_lm Qwen/Qwen3-30B \
    --port 8000 \
    --monitor \
    --verifier_port 8001 \
    --verifier_model medverifier \
    --evidence_source pubmed \
    --line_interval 15 \
    --max_corrections 10 \
    --verification_window 3 \
    --max_samples 1000 \
    --n_processes 8
```

### Baseline (no monitor)

```bash
python medical_example.py \
    --solver_lm Qwen/Qwen3-30B \
    --port 8000 \
    --max_samples 1000 \
    --n_processes 8
```

### With SNOMED + PubMed combined

```bash
python medical_example.py \
    --solver_lm Qwen/Qwen3-30B \
    --port 8000 \
    --monitor \
    --verifier_port 8001 \
    --verifier_model medverifier \
    --evidence_source both \
    --line_interval 15 \
    --max_corrections 10 \
    --max_samples 1000 \
    --n_processes 8
```

### With preprocessing (case extraction + SNOMED prefetch)

```bash
python medical_example.py \
    --solver_lm Qwen/Qwen3-30B \
    --port 8000 \
    --monitor \
    --verifier_port 8001 \
    --verifier_model medverifier \
    --evidence_source pubmed \
    --preprocess_case \
    --prefetch_snomed \
    --max_samples 1000 \
    --n_processes 8
```

### Continue an interrupted run

```bash
python medical_example.py \
    --solver_lm Qwen/Qwen3-30B \
    --port 8000 \
    --monitor \
    --verifier_port 8001 \
    --continue_from 20240101_120000
```

---

## CLI reference

| Argument | Default | Description |
|---|---|---|
| `--solver_lm` | required | HuggingFace model name for the solver |
| `--port` | 8000 | Solver vLLM server port |
| `--monitor` / `-m` | off | Enable verification monitor |
| `--verifier_port` | 8001 | Verifier vLLM server port |
| `--verifier_model` | `medverifier` | Verifier model name |
| `--evidence_source` | `pubmed` | `pubmed` / `snomed` / `both` / `none` |
| `--line_interval` | 15 | Trigger verification every N lines |
| `--max_corrections` | 10 | Max feedback blocks per sample before stopping |
| `--verification_window` | 3 | Paragraphs sent per verification call |
| `--confidence_threshold` | 0.9 | Min verifier confidence to act on FALSE. Lower for stronger/same-family verifiers (e.g. 0.7). Keep high for weak/misaligned verifiers (e.g. Meditron3 8B). |
| `--no_snomed` | off | Disable SNOMED even if evidence_source includes it |
| `--preprocess_case` | off | Extract compact case facts before generation (1 extra LLM call/sample) |
| `--prefetch_snomed` | off | Batch-fetch SNOMED for option terms before generation |
| `--split` | `train` | Dataset split |
| `--max_samples` | 20 | Number of samples to run |
| `--start_idx` | 0 | Start index in dataset |
| `--end_idx` | -1 | End index (-1 = all) |
| `--n_processes` / `-p` | 8 | Parallel workers |
| `--debug` / `-d` | off | Run first sample only, synchronous |
| `--continue_from` / `-c` | None | Resume from existing output directory |
| `--extra` | - | Suffix appended to output directory name |

---

## Configuration in code

`VerifierConfig` (in `medical_verifier.py`) controls all verifier behaviour:

```python
@dataclass
class VerifierConfig:
    run_snomed:               bool  = True
    unknown_defaults_to_pass: bool  = True   # UNKNOWN → PASS (no disruption)
    allow_unknown:            bool  = True   # allow verifier to return UNKNOWN
    max_prior_context_chars:  int   = 4000   # truncate prior reasoning if too long
    snomed_rate_limit_sleep:  float = 0.3    # seconds between SNOMED API calls
    evidence_source:          str   = "pubmed"
    verification_window:      int   = 3      # paragraphs per call
    max_feedback_per_sample:  int   = 10
    confidence_threshold:     float = 0.8    # minimum confidence to act on FALSE
                                             # directive fires at threshold + 0.05
```

---

## Verification logic

### Trigger

The monitor fires when either:
- `UNKNOWN` appears in the streaming chunk - solver flagged its own uncertainty
- Newline count reaches `line_interval` (default 15) since the last trigger

Trigger only fires inside the `<think>` block. Once `</think>` appears, no more verification.

Note: vLLM strips the opening `<think>` tag from the SSE stream. The monitor detects "inside think block" by checking whether `</think>` has appeared yet.

### Paragraph classification

On each trigger, the verifier classifies the latest content:

| Class | What it is | Verification |
|---|---|---|
| `OBSERVATION` | Facts stated from the case (vitals, symptoms, labs) | Grounding check against compact case - skipped for knowledge MCQs |
| `INFERENCE` | Clinical conclusion, causal claim, likelihood judgment | Hypothesis check against prior reasoning |
| `OPTION_COMPARISON` | Explicit evaluation of answer options | Same as INFERENCE |
| `CONCLUSION` | Final answer selection or diagnosis | Hypothesis check, no UNKNOWN allowed |
| `OTHER` | Transitions, restatements, meta-text | Skipped |

### Knowledge MCQ detection

If `compact_case` has no patient, vitals, chief complaint, labs, or imaging - it is a factual MCQ with no clinical vignette. Observation grounding is skipped entirely. Only INFERENCE and CONCLUSION checks run, using prior reasoning as context.

### Confidence gate

The verifier returns a confidence score (0–1) alongside TRUE/FALSE/UNKNOWN.

- `confidence < confidence_threshold` → treat as PASS regardless of label
- `confidence_threshold ≤ confidence < threshold + 0.1` → FAIL, feedback injected without the "your final answer may need to change" directive
- `confidence ≥ threshold + 0.1` → FAIL, full directive feedback injected

Default threshold is 0.8. Recommended values:

| Verifier | Recommended threshold |
|---|---|
| Meditron3 8B | 0.9 - small model, strong parametric opinions, high false positive rate |
| Same model as solver | 0.7 - better calibrated on its own reasoning style |
| Larger/stronger model | 0.7 - more trustworthy FALSEs |

### Flip-flop prevention

Each verifier instance tracks corrected topics in `_corrected_topics`. If the same wrong claim (first 50 chars, lowercased) has already been corrected once in this sample, a second correction on the same topic is skipped and treated as PASS. Prevents the verifier contradicting itself across consecutive calls.

### Evidence fetching on FALSE

When a paragraph fails with confidence ≥ 0.8:

**`evidence_source="pubmed"`** - queries NCBI E-utilities:
1. `esearch`: finds PMIDs matching the wrong claim + quality filter (`meta-analysis[pt] OR systematic review[pt] OR practice guideline[pt]`). Falls back to plain search if no results.
2. `efetch`: retrieves abstracts (capped at 2000 chars).
3. Abstract text is injected into the feedback block.

**`evidence_source="snomed"`** - queries BioPortal SNOMED CT:
1. Verifier LLM extracts terms from the wrong claim.
2. Each term is looked up via BioPortal API.
3. Definitions are injected into the feedback block.

**`evidence_source="both"`** - both of the above.

**`evidence_source="none"`** - no external lookup. Verifier uses parametric knowledge only.

### Feedback format

Directive style - guides thought, does not punish:

```
[FEEDBACK]
Consider reconsidering: {wrong_claim}
Alternative perspective: {correction}
  - {verifier evidence}

Relevant evidence:
{PubMed abstracts or SNOMED definitions}

Re-evaluate your option selection. Your final answer may need to change.
[/FEEDBACK]
```

The final directive line only appears when `confidence ≥ 0.9`.

---

## Evidence sources

### PubMedClient

Free NCBI E-utilities API. No API key required for basic use (rate limited to 3 req/s). Set `NCBI_API_KEY` for 10 req/s.

```python
pubmed = PubMedClient(
    max_results = 3,      # number of papers to fetch abstracts from
    max_chars   = 2000,   # total chars cap across all abstracts
    timeout     = 15,     # seconds per HTTP request
)
text = pubmed.get_evidence("urea breath test H pylori sensitivity")
```

Best for: sensitivity rankings, most common causes, investigation of choice, drug mechanisms, clinical guidelines.

### SnomedClient

BioPortal SNOMED CT API. Requires `BIOPORTAL_API_KEY`.

```python
snomed = SnomedClient(top_k=5, timeout=15)
result = snomed.enrich(question="", option_text="troponin I")
# result["definition"] = "A cardiac biomarker..."
```

Best for: terminology definitions, anatomical relationships, disease classifications.

### MedicalReasoningVerifierSnomedFirst

Subclass of `MedicalReasoningVerifier` that additionally pre-fetches SNOMED for each paragraph before verification. This enriches the SNOMED cache so the verifier prompt includes relevant definitions even before a FALSE is returned.

Only runs when `evidence_source` includes `"snomed"`. No-op when `evidence_source="pubmed"` or `"none"`.

---

## Pre-processing (optional)

Both steps run before generation starts. Both default to off (cost saving).

### Case extraction (`--preprocess_case`)

One verifier LLM call per sample. Converts the raw question text into a compact JSON:

```json
{
    "patient": "58yo male",
    "chief_complaint": "crushing chest pain 45min, left arm radiation",
    "vitals": "BP 88/60, HR 110, SpO2 96%",
    "ecg": "ST-elevation II, III, aVF",
    "labs": ["troponin pending"],
    "history": null
}
```

Used by `_verify_observation` to ground-check facts against actual case data. Without this, compact_case is empty and observation grounding is skipped for all samples.

### SNOMED prefetch (`--prefetch_snomed`)

One batch of SNOMED lookups per sample before generation. Fetches definitions for the option terms (A/B/C/D) so they are in the cache for the first verification call. Only useful when `evidence_source` includes `"snomed"`.

---

## Output format

Results written to `Outputs_TTS/medreason/{timestamp}/outputs.jsonl`. Each line:

```json
{
    "sample_id":    "1234",
    "question":     "Most sensitive test for H. pylori is...",
    "ground_truth": "B",
    "output_text":  "<think>...\n</think>\n\n[FINAL ANSWER]...[/FINAL ANSWER]",
    "correct":      true,
    "exact_matched": true,
    "decision_log": [
        {
            "type":              "INFERENCE",
            "label":             "FAIL",
            "paragraph_preview": "The rapid urease test has the highest...",
            "feedback":          "Consider reconsidering: ..."
        }
    ]
}
```

`correct` - primary scorer, reads `[FINAL ANSWER] Selected Option:` block. Falls back to word overlap if block is absent.

`exact_matched` - whether a `[FINAL ANSWER]...[/FINAL ANSWER]` block was found.

`decision_log` - one entry per verifier call. Empty if monitor is off.

---

## Scoring

```python
# Primary: reads [FINAL ANSWER] block
def exact_correctness_check(output_text, sample):
    m = re.search(r"\[FINAL ANSWER\](.*?)\[/FINAL ANSWER\]", output_text, re.DOTALL)
    ...

# Fallback: word overlap on last 600 chars
def rough_correctness_check(output_text, sample):
    ...
```

---

## Key design decisions

**Verifier never sees question or options.** The verifier judges reasoning consistency only - it receives prior reasoning as context and the new paragraph as hypothesis. Question and options are not forwarded. This prevents the verifier from using its own answer preference to override the solver's reasoning chain.

**No routing LLM call.** Evidence source is purely a config parameter. SNOMED or PubMed (or both) is called on every FALSE at confidence ≥ 0.8, regardless of claim type.

**Pending revision pattern.** When a paragraph fails, `_pending_revision` stores `(wrong_claim, correction)`. The compact state is not updated yet. Only when the next verification call returns TRUE (model incorporated the feedback) is the revision written to `CompactState.claims`. This prevents the state from showing a revision that the model never made.

**Fail-open on verifier errors.** Any exception in `_call_verifier` (connection drop, timeout, JSON parse failure) returns `(True, None)` - the sample continues without feedback. A crashed worker does not take down the multiprocessing pool.

---
