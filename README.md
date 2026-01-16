# Reflena Green Agent

Reflena is a Green Agent built for the AgentBeats ecosystem. This repository contains the complete implementation of a Green Agent that evaluates Purple Agents in a controlled and secure execution environment. The focus of this project is correctness, reproducibility, and transparent evaluation rather than raw performance.

The agent is designed to be simple to understand, easy to extend, and safe to run in automated evaluation pipelines.

**AgentBeats Reflena Green Agent:** https://agentbeats.dev/sajid-01/reflena

---

## What this Agent Does

Reflena acts as an evaluator. It receives trajectories and outputs produced by Purple Agents and assigns scores based on predefined evaluation rules. These rules measure factors such as task correctness, logical consistency, and adherence to constraints defined by the environment.

All Purple Agent code is executed in a restricted sandbox to ensure safety and fairness. The Green Agent never executes untrusted code directly on the host system.

---

## Design Goals

The main goals of this project are:

Provide a clear and deterministic evaluation framework  
Ensure safe execution of untrusted agent code  
Support multiple task environments and difficulty levels  
Make scoring explainable and auditable  
Keep the codebase minimal and readable  

---

## Project Structure

src/
  server.py        Entry point that starts the Green Agent server and exposes metadata
  executor.py      Handles execution requests and orchestrates evaluations
  agent.py         Core evaluation logic and scoring rules
  messenger.py     Utilities for communication and message formatting

data/
  reflena_benchmark.json   Benchmark definition containing tasks, test cases, and expected outputs

tests/
  test_agent.py    Basic tests to validate evaluation behavior

Dockerfile         Container configuration for deployment
pyproject.toml     Dependency and tooling configuration
.github/workflows  Continuous integration workflows

---

## Benchmark Data

The data folder contains `reflena_benchmark.json`, which defines the evaluation benchmark used by the Green Agent.

This file includes:

Tasks that describe the problem the Purple Agent must solve  
Test cases associated with each task  
Expected outputs used only by the Green Agent for evaluation  

The expected outputs are never exposed to Purple Agents. Purple Agents only receive the task description and input data, while Reflena uses the hidden expected outputs to compute correctness and assign scores.

This separation ensures fair evaluation and prevents leakage of ground truth information.

---

## Benchmark Format Example

Below is a simplified example of a single benchmark entry from `reflena_benchmark.json`. This is provided for clarity and does not expose the full benchmark used during evaluation.

```
{
  "problem": "spectral_radius_power_iteration",
  "function_name": "spectral_radius",
  "signature": "A, num_iter",
  "description": "Estimate the spectral radius (largest absolute eigenvalue) of a square matrix using the power iteration method.",
  "cases": [
    {
      "type": "core",
      "input": {
        "A": [[4, 1], [2, 3]],
        "num_iter": 50
      },
      "output": 5.0
    },
    {
      "type": "noisy",
      "input": {
        "A": [[4.001, 1], [2, 2.999]],
        "num_iter": 60
      },
      "output": 5.0
    }
  ],
  "tolerance": 1e-2
}
```

Each benchmark entry defines the expected function behavior, multiple test cases with varying difficulty or noise, and a tolerance value used during numerical comparison.

---

## Running the Agent Locally

Clone the repository and move into the project directory.

git clone https://github.com/sajid-01/reflena.git
cd reflena

Install dependencies using uv.

uv sync

Start the agent.

uv run src/server.py

By default the agent will be available on port 9009.

---

## Running with Docker

Build the container image.

docker build -t reflena .

Run the container.

docker run -p 9009:9009 reflena

The Green Agent will now be accessible on port 9009.

---

## Testing

The repository includes a small test suite to ensure the agent behaves as expected.

Install test dependencies.

uv sync --extra test

Run tests against a running agent.

pytest --agent-url http://localhost:9009

---

## Evaluation Logic

Reflena evaluates Purple Agents by analyzing their trajectories rather than only their final answers. This allows the agent to penalize incorrect reasoning, unsafe actions, or violations of environment constraints even if the final output appears correct.

Scores can be weighted based on task difficulty, environment type, or noise level. This makes it possible to reward correct solutions on harder tasks more than trivial ones.

---

## Customization

To modify the evaluation behavior, edit src/agent.py.  
To change agent metadata such as name, description, or supported skills, edit src/server.py.  
Additional benchmark tasks or test cases can be added by extending reflena_benchmark.json in the data folder.

The code is intentionally straightforward so that new evaluation strategies can be added without refactoring the entire system.

---

## Validation

Purple Agent results are validated by comparing expected outcomes with observed trajectories and outputs. The Green Agent itself is validated through unit tests and controlled example runs where the correct score is known in advance.

This two layer validation ensures both agent execution and agent evaluation are behaving correctly.

---
