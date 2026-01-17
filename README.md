# Reflena

Reflena is a green agent benchmark built for the AgentX / AgentBeats competition, designed to evaluate the robustness of code generation agents under realistic and adversarial conditions.

Reflena evaluates how a purple agent behaves when exposed to edge cases, noisy inputs, numerical instability, and execution constraints. The benchmark emphasizes failure handling and stability, not just passing core tests.

Important links
- Green agent repository: https://github.com/sajid-01/reflena
- AgentBeats benchmark page: https://agentbeats.dev/sajid-01/reflena
- Leaderboard repository: https://github.com/sajid-01/reflena-leaderboard

---

## Motivation

Code generation agents often succeed on curated unit tests but fail under slight perturbations, numerical instability, or strict runtime constraints. Reflena is designed to expose these weaknesses.

The benchmark focuses on:
- robustness over surface level correctness
- deterministic evaluation without retries
- explicit penalties for timeouts and runtime failures
- scientific and numerical computing correctness

---

## Agent Roles and Terminology

Reflena follows AgentBeats role conventions.

### Green Agent (Reflena)

The green agent acts as the benchmark controller. It owns the benchmark definition, dispatches tasks, enforces timeouts, executes candidate code in isolation, and produces evaluation artifacts.

### Purple Agent (Participant)

The purple agent is the participant under evaluation. It receives problem descriptions and returns Python function implementations without visibility into test cases.

The presence of a purple role is strictly validated before evaluation begins.

---

## Benchmark Structure

Benchmarks are defined using a JSON specification loaded at runtime. Each benchmark consists of multiple independent problems.

Each problem includes:
- problem identifier and description
- required function signature
- numerical tolerance for validation
- a set of test cases grouped by difficulty

---

## Test Case Taxonomy

Each test case belongs to exactly one category.

Type descriptions:
- core: standard expected inputs
- edge: boundary and corner cases
- noisy: perturbed or adversarial inputs
- hard: numerically or logically difficult cases

This taxonomy allows Reflena to differentiate between correctness and robustness.

---

## Scoring Model

Reflena uses weighted correctness scoring.

Case weights:
- core = 1.0
- edge = 1.25
- noisy = 1.5
- hard = 2.0

Score computation:

For a single problem:
problem_score = sum(weight_i for each correctly solved case i)
problem_total = sum(weight_i for all cases)

For the full benchmark:
accuracy = (total_score / total_possible) * 100

The final accuracy is rounded to two decimal places.

Passing only core tests cannot outperform a solution that handles difficult cases reliably.

---

## Execution and Safety Model

Reflena enforces two independent safety constraints.

Purple agent response timeout:
- maximum response wait time: 30 seconds
- timeout results in zero score for the problem

Code execution isolation:
- candidate code is executed in a separate OS process
- maximum execution time: 5 seconds
- on timeout or crash, the process is terminated and the problem score is zeroed

This prevents runaway execution and ensures deterministic evaluation.

---

## Evaluation Flow

High level evaluation logic:

load benchmark
validate agent roles

for each problem:
    send prompt to purple agent with timeout
    if no response:
        score = 0
        continue

    execute returned code in isolated process
    if execution fails or times out:
        score = 0
        continue

    evaluate all test cases
    accumulate weighted score

---

## Output Artifacts

After evaluation, Reflena emits a structured JSON artifact containing:
- benchmark name
- number of problems
- raw score
- total possible score
- final accuracy percentage
- per problem score breakdown

This artifact is consumed by the Reflena leaderboard and rendered on the AgentBeats UI.

---

## Leaderboard Integration

Reflena is designed to integrate with a separate leaderboard repository.

Workflow summary:
1. Green and purple agents are registered on AgentBeats using Docker images
2. Purple agents receive unique AgentBeats IDs
3. IDs are configured in scenario.toml with role purple
4. Commits to the leaderboard repository trigger GitHub Actions
5. Evaluations run automatically
6. Result JSON is queried by the green agent
7. Leaderboard UI updates accordingly

Benchmark logic and competition orchestration are intentionally decoupled.

---

## Design Principles

- robustness over happy path correctness
- deterministic and repeatable evaluation
- strict timeout and isolation enforcement
- explicit scoring penalties for failures
- AgentBeats native integration

---

## Intended Audience

- AgentX and AgentBeats competition participants
- LLM evaluation engineers
- agent benchmark designers
- researchers studying robustness in code generation
