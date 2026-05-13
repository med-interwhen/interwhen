<h1 align="center">interwhen</h1>

<p align="center">
   A Generalizable Framework for Steering Reasoning Models with  Test-time Verification
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.11202"><img src="https://img.shields.io/badge/arXiv-2602.11202-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="Paper"></a>
  <!-- <a href="https://pypi.org/project/interwhen/"><img src="https://img.shields.io/pypi/v/interwhen?logo=python&logoColor=white&color=3776ab" alt="PyPI"></a> -->
  <a href="https://github.com/microsoft/interwhen"><img src="https://img.shields.io/github/stars/microsoft/interwhen?style=flat&logo=github&color=181717" alt="GitHub stars"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat&logo=python&logoColor=white" alt="Python 3.10+">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.11202"><b>Paper</b></a> &nbsp;|&nbsp;
  <a href="#installation"><b>Quick Start</b></a> &nbsp;|&nbsp;
  <a href="#examples"><b>Examples</b></a> &nbsp;|&nbsp;
  <a href="#available-monitors-and-verifiers"><b>Monitors</b></a> &nbsp;|&nbsp;
  <a href="#creating-custom-verifiers-and-monitors"><b>Create your own Monitors</b></a>
</p>

interwhen is a test-time verification framework for language models that enforces correctness with respect to a set of verifiers. It is designed to improve *instance*-level reliability in reasoning systems, particularly in high-stakes domains where occasional errors are unacceptable. 

This is especially important for agentic workflows, where models make sequences of decisions interleaved with tool calls, database writes, or external API actions. In these settings, verifying only the final answer can miss early policy violations or irreversible mistakes. interwhen instead enables LLM-Process-Modulo execution: the model is steered during the reasoning or action process so that its trace remains compliant with task-specific policies

The framework has two parts. Offline, interwhen can synthesize code-based verifiers from natural-language policy documents, including provably correct verifiers in Lean or Z3. Online, interwhen periodically polls the
reasoning trace and forks inference of the reasoning model to recover intermediate states. Verifiers are run
asynchronously alongside generation, adding negligible overhead on correct executions and intervening only
when violations occur. 

interwhen addresses the problem by providing a plug-and-play mechanism to improve instance-level reliability of any language model, which we call *verifier-guided reasoning*. Instead of verifying only the final output, the framework enables verification of intermediate reasoning traces during generation. When a violation is detected, the system can steer, revise, or halt generation. If no output is produced, the system abstains; if an output is produced, it satisfies the specified verifiers.

From a research perspective, interwhen makes the following contributions:

- **A New Axis for Test-Time Scaling** — Introduces verifier compute as an additional dimension of scaling at inference time. Rather than scaling model size or sampling alone, performance can be improved by allocating compute to structured verification.

- **Automatic Verifier Synthesis** — Provides a method for generating verifiers automatically from a given natural-language policy. We also propose a Lean-based variant that produces formal specifications, corresponding verifiers, and machine checked proofs of soundness and completeness of the verifiers. 

- **A Testbed for Verifier Development** — Enables systematic evaluation of verifier designs at inference time before incorporating them into training objectives (e.g., as reward models or critics).

A detailed discussion of interwhen, including how it was developed and tested, can be found in our [paper](https://arxiv.org/abs/2602.11202).

<table>
<tr>
<td align="center" colspan="2">

<img src="https://github.com/user-attachments/assets/41496b82-bdfd-4e07-8871-3fd9dfc46105" alt="Agentic policy compliance demo" height="400" /></a>
<br>
A demo on the Telecom domain in Tau2-Bench, operating in solo mode. The verifiers are first generated from the rules defined in the policy. As the agent's execution progresses, each tool call is checked against the applicable policy verifiers, with feedback returned when a violation is detected. The demo shows how interwhen steers the same trajectory toward policy-compliant execution without restarting the agent.

</td>
</tr>
<tr>
<td align="center">

<img src="https://github.com/user-attachments/assets/b456273c-efa6-4231-8946-34c234e7607d" alt="Maze" height="300" />
<br>
A demo on the Maze dataset. The task is to find the number of right and left turns on the path from the starting to the ending position. The colour green indicates steps that pass verification, while red marks those that failed. The text stream on the right is the verifier output. A higher res mp4 can be found <a href="https://github.com/user-attachments/assets/ce6133c5-b4f4-4578-b9d8-65d4e4475054">here</a>.

</td>
<td align="center">

<img src="https://github.com/user-attachments/assets/67d3a461-1954-4c09-9c37-cf14db47b9bb" alt="ZebraLogic" height="300" />
<br>
A demo on the ZebraLogic dataset. The task is to find the correct assignments given the constraints. The colour green indicates steps that pass verification, while red marks those that failed. The text stream on the right is the verifier output. A higher res mp4 can be found <a href="https://github.com/user-attachments/assets/5b5fc6d8-2239-443b-b26b-6324a9e3556b">here</a>.

</td>
</tr>
</table>



## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Verifiable Reasoning in Three Lines](#verifiable-reasoning-in-three-lines)
- [Examples](#examples)
- [Available Monitors](#available-monitors-and-verifiers)
- [Creating Custom Verifiers and Monitors](#creating-custom-verifiers-and-monitors)
- [How It Works](#how-it-works)
- [Evaluation](#evaluation)
- [Intended Uses](#intended-uses)
- [Limitations](#limitations)
- [License](#license)
- [Contact](#contact)

## Key Features
interwhen changes the inference pipeline of a language model by creating an auxiliary Monitor that runs alongside the main model and interacts with the model’s output to improve its quality. The Monitor agent reads the output of a language model in real time and calls necessary verifiers to check its validity. 

1. **Policy Compliant Agentic Reasoning** 
interwhen verifies intermediate reasoning states, tool-use decisions, and tool-responses before the model reaches a final answer, with the aim of ensuring that the actions taken by the agent are compliant with the policy provided. This is useful for agentic workflows where early mistakes can propagate into irreversible tool calls or invalid task outcomes, and hence process verification is essential.


2. **Verification During Generation**. interwhen verifies reasoning traces as they are produced, without requiring external step extraction or structured decomposition. This allows the model to retain flexible reasoning strategies while remaining subject to correctness constraints.

3. **Asynchronous and Efficient Execution**. Verifiers are executed asynchronously and intervene only when violations are detected, minimizing inference overhead while preserving responsiveness.

4. **Unified Model–Verifier Interface**. The framework provides a general API for interaction between language models and different kind of verifiers. Based on the objectivity of a domain, verifiers can be symbolic, neuro-symbolic or even fully neural verifiers. They can operate on partial outputs,  final answers, or both. 

----------------

At a conceptual level, interwhen reframes reliability in language models:

> Instead of asking whether a model is accurate on average, we ask whether a particular output complies with explicit, verifiable constraints derived from a natural language policy.

By integrating verification directly into generation, interwhen provides a general mechanism for improving the soundness of reasoning systems without restricting model expressivity or requiring retraining.

## Installation

**Clone repo**
```bash
git clone https://github.com/microsoft/interwhen.git
cd interwhen
```
It is recommended to setup a fresh environment before installing the library.

**Install dependencies**

```bash
pip install -e .
```

**Test installation**
Run the following script to reproduce the text replacement example from above. 

```bash
python ./examples/text_replacement_example.py
```

## Verifiable Reasoning in Three Lines 
Running verifier-guided inference requires only a few lines of code: just specify the list of monitors to be used with a target LLM. Each monitor requires specifying the kind of verifier, when it should be invoked (e.g., each step or after a reflection token like 'Wait'), and the text pattern to intervene with. 

**Set up target LLM server**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --max-model-len 65536 \
  --port 8000 \
  --tensor-parallel-size 8
```

**Generate answer enabled with given monitors**
```python
llm_server = init_llm_server("Qwen/Qwen3-30B-A3B-Thinking-2507", max_tokens=32768, port=8000)
stream_completion(
    prompt,
    llm_server=llm_server,
    monitors=(monitor = StepVerifierMazeMonitor.from_prompt(
                  prompt_text=user_prompt,
                  max_corrections=5,
                  name="maze_step_verifier"),
    ),
    async_execution=True
)
```
The above example is for the Maze dataset, where a maze is given and questions are asked with respect to the maze, such as how many right turns are present in the path from given starting and ending positions. The above code implements a simple monitor that watches the model's output stream and verifies if the step proposed by the model is valid or not. You can run the full example using the following command:
```bash
python ./examples/TTSwithVerification/maze_stepverifier.py.
```

## Examples

### Improving accuracy through intermediate verification
We provide examples using 5 datasets: Maze, Game of 24, SpatialMap, Verina, and ZebraLogic.

```bash
python ./examples/TTSwithVerification/[your_dataset]_stepverifier.py -n 1 # dataset=maze,game24,spatialmap,verina_code, verina_spec, zebralogic
```

### Early stopping
Early stopping monitors detect when a model has converged on an answer and terminate generation to save compute. For example, if the model produces the same answer *k* times in a row within its reasoning stream, further generation is unlikely to change the final outcome.

```bash
python ./examples/EarlyStopping/[your_dataset]_example.py -n 1
```

## Available Monitors and Verifiers

interwhen provides two types of components: **monitors** that orchestrate when and how to intervene during generation, and **verifiers** that check the correctness of reasoning steps.

### Monitors

Monitors watch the model's output stream and decide when to invoke verifiers and how to act on their results.

- **Single Trajectory Monitor** : Watches a single generation stream, invokes verifiers on each intermediate state (obtained by forking at trigger points), and injects corrective feedback when a violation is found. This is the default monitor used for step-level verification.

- **K-Stable Monitor** : Tracks whether the model has converged on a stable answer by checking if the same answer appears *k* consecutive times. Used for early stopping to save compute once the model is unlikely to change its output.

### Verifiers

Verifiers provide the domain-specific correctness checks that monitors invoke. Different verifiers return different types of feedback to steer the model.

| Type of Verifier | Type of Feedback | Datasets |
|------------------|------------------|----------|
| Z3-based verifier | Feedback on satisfiability | ZebraLogic, SpatialMap |
| Compiler-based verifier | Compilation error message | Verina |
| Python code | Custom feedback (eg. "right turn not allowed", <br> "numbers do not add up to 24", etc.) | Maze, Game of 24 |

📖 Full documentation and parameter reference: [Step Verification](./examples/TTSwithVerification/README.md) · [Early Stopping](./examples/EarlyStopping/README.md)


## Creating custom verifiers and monitors
You can create your own custom monitors by subclassing `VerifyMonitor` in `interwhen/monitors/base.py`. A custom monitor requires implementing three methods:

- **`step_extractor(chunk, generated_text)`** — Determines *when* to intervene by detecting meaningful reasoning steps in the model's streaming output. Returns a boolean indicating whether a new step has been identified and should be verified.
- **`verify(chunk, token_index, event, event_info)`** — Checks the correctness of the extracted step using domain-specific logic (symbolic solvers, rule checks, etc.) and signals whether a correction is needed.
- **`fix(generated_text, event_info)`** — Constructs the corrective feedback that is injected into the model's generation stream to steer it back on track.

## How It Works

![diagram](https://github.com/user-attachments/assets/8c9e2992-2fa4-49de-bab5-2c36d47fb05c)

interwhen implements **LLM-Process-Modulo**: instead of verifying only the final answer, it monitors a single reasoning or agentic trajectory as it unfolds and checks whether intermediate states satisfy a task policy. The framework is built around two operations: extracting verifiable states from a partial trace, and running policy verifiers on those states.

Given a new task domain, interwhen operates in two phases.

1. **Offline policy formalization.** A natural-language policy document is treated as a set of rules. For agentic domains, this policy may be an operational rulebook, such as a telecom or retail agent policy, optionally paired with a description of the available tools. interwhen uses a frontier LLM to generate code-based verifiers for the policy rules and a mapping from state patterns to the verifiers that should be invoked. For domains requiring stronger guarantees, interwhen can generate Lean specifications, verifier implementations, and machine-checked proofs that the verifier code is sound and complete with respect to the formalized rule.

2. **Streaming generation.** At inference time, the target model generates a single reasoning trace. This trace may contain chain-of-thought tokens, tool calls, tool outputs, intermediate answers, and a final response. interwhen does not require the model to follow a rigid step-by-step template. Instead, it uses lightweight boundaries, such as paragraph breaks, reflection tokens, or tool-call events, to decide when the current partial trace should be checked.

3. **State extraction.** At each boundary, interwhen extracts the variables needed by the relevant verifiers from the partial trace. These variables may include tool names, tool arguments, database fields, proposed actions, intermediate formulas, next game moves, or partial answers. In structured agentic settings, some states can be parsed directly from tool calls. In less structured reasoning traces, interwhen forks the model execution and prompts the model itself to extract the required state variables into a dictionary.

4. **Asynchronous verification.** Once state variables are available, interwhen invokes the applicable verifiers. A verifier may return `True`, `False`, or `Unknown`: `True` means the state satisfies the policy rule, `False` means a violation was detected, and `Unknown` means the verifier does not yet have enough information to decide. Verification runs asynchronously alongside generation, so correct executions are not forced to wait for every check to complete.

5. **Intervention.** If a verifier returns `False`, it also returns text feedback explaining the violation. interwhen interrupts the main generation, rolls the trace back to the checked point, appends the verifier feedback, and resumes generation from there. In agentic settings, feedback can be provided as a tool response or as part of the model’s reasoning context. For write-like tool calls, verification can be made blocking so invalid irreversible actions are stopped before execution.

6. **Termination or abstention.** This extract, verify, and intervene loop continues until the model produces a final answer, reaches a token limit, or exceeds the allowed number of correction attempts. If the retry limit is exceeded, interwhen abstains rather than returning an answer that violates the specified verifiers.

This design lets interwhen steer reasoning and tool-using agents without finetuning, branch search, or repeated full retries. The main model follows one trajectory, while verifiers run in parallel and intervene only when the trace becomes non-compliant.

## Intended Uses
- interwhen was developed to improve the quality of a reasoning model’s outputs without requiring finetuning.
- interwhen is best suited for tasks where verification is feasible, such as math, code reasoning, or structured document generation—not highly subjective tasks like creative writing or open-ended opinion pieces where correctness cannot be formally defined. 
- interwhen is being shared with the research community to facilitate reproduction of our results and foster further research in this area.
- interwhen is intended to be used by domain experts who are independently capable of evaluating the quality of outputs before acting on them.

## Out-of-scope Uses
- interwhen is not well suited for subjective tasks where answer verification is harder (or more complex).
We do not recommend using interwhen in commercial or real-world applications without further testing and development. It is being released for research purposes.
- interwhen was not designed or evaluated for all possible downstream purposes. Developers should consider its inherent limitations as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness concerns specific to each intended downstream use.
- Without further testing and development, interwhen should not be used in sensitive domains where inaccurate outputs could suggest actions that lead to injury or negatively impact an individual's legal, financial, or life opportunities.
- We do not recommend using interwhen in the context of high-risk decision making (e.g. law enforcement, legal, finance, or healthcare).

## Evaluation
interwhen was evaluated on its ability to improve the reasoning quality of existing language models on benchmarks spanning planning, math, logic, and deep research.
A detailed discussion of our evaluation methods and results can be found in our [paper](https://www.microsoft.com/en-us/research/publication/interwhen-a-generalizable-framework-for-verifiable-reasoning-with-test-time-monitors/).

### Evaluation Methods
We used accuracy and efficiency metrics to measure interwhen’s performance.
We compared the performance of interwhen against baseline methods such as tree-of-thought and tool calling using benchmarks such as Maze, SpatialEval, Game of 24, Verina and others.
The target model (the model whose reasoning performance was improved) used in our experiments varied by task and included models from Qwen2, Qwen3 and Phi-4 series.
In our experiments, we used models from Qwen2 and Qwen3 series as the auxiliary monitor model. Results may vary if interwhen is used with a different monitor model, based on its unique design, configuration and training.

### Evaluation Results
At a high level, we found that interwhen allows a plug-and-play solution for improving the accuracy (and/or) efficiency of language models. The accuracy improvement on various benchmarks is shown below. Depending on the goal, interwhen can improve the accuracy of a language model given a compute budget or improve the efficiency of the model at a given accuracy.

## Limitations
- interwhen was developed for research and experimental purposes. Further testing and validation are needed before considering its application in commercial or real-world scenarios.
- interwhen supports a human-feedback based monitor, however, such a monitor may not be feasible in situations where latency of the model’s output is a key consideration.
- interwhen was designed and tested using the English language. Performance in other languages may vary and should be assessed by someone who is both an expert in the expected outputs and a native speaker of that language.
- Outputs generated by AI may include factual errors, fabrication, or speculation. Users are responsible for assessing the accuracy of generated content. All decisions leveraging outputs of the system should be made with human oversight and not be based solely on system outputs.
- interwhen inherits any biases, errors, or omissions produced by the auxiliary monitor model, as chosen by the developer. Developers are advised to choose appropriate target and auxiliary LLMs carefully, depending on the intended use case.
- interwhen is a framework which can run with any language model preferred by a user. Users can specify the language model whose reasoning they want to improve (“target” model) and an auxiliary model that monitors the target model’s reasoning trace.
- interwhen inherits any biases, errors, or omissions characteristic of the training data of the language models used, which may be amplified by any AI-generated interpretations.
- There has not been a systematic effort to ensure that systems using interwhen are protected from security vulnerabilities such as indirect prompt injection attacks. Any systems using it should take proactive measures to harden their systems as appropriate.

## Best Practices
Better performance can be achieved by assessing the utility of included verifiers for your task and activating only the necessary ones.
We strongly encourage users to use LLMs/MLLMs that support robust Responsible AI mitigations, such as Azure Open AI (AOAI) services. Such services continually update their safety and RAI mitigations with the latest industry standards for responsible use. For more on AOAI’s best practices when employing foundations models for scripts and applications:

- [What is Azure AI Content Safety?](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview)
- [Overview of Responsible AI practices for Azure OpenAI models](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/overview)
- [Azure OpenAI Transparency Note](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/transparency-note)
- [OpenAI’s Usage policies](https://openai.com/policies/usage-policies)
- [Azure OpenAI’s Code of Conduct](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/code-of-conduct)

## License
MIT License

Nothing disclosed here, including the Out of Scope Uses section, should be interpreted as or deemed a restriction or modification to the license the code is released under.

## Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

## Contact
This research was conducted by members of Microsoft Research.  We welcome feedback and collaboration from our audience. If you have suggestions, questions, or observe unexpected/offensive behavior in our technology, please contact us by posting an issue on Github or at interwhen@microsoft.com.

If the team receives reports of undesired behavior or identifies issues independently, we will update this repository with appropriate mitigations.

## Citation
If you are using interwhen, please cite the corresponding paper below.

```bibtext
@article{bhat2026interwhen,
  title={interwhen: A Generalizable Framework for Verifiable Reasoning with Test-time Monitors},
  author={Bhat, Vishak K and Chanda, Prateek and Ekbote, Vijval and Khandelwal, Ashmit and Swaroop, Maitreyi and Balasubramanian, Vineeth N and Kambhampati, Subbarao and Natarajan, Nagarajan and Sharma, Amit},
  journal={arXiv preprint arXiv:2602.11202},
  year={2026}
}
```
