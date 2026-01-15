from typing import Any
import json
import numpy as np
from pathlib import Path

from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]
    config: dict[str, Any]


class Agent:
    required_roles = ["purple"]
    required_config_keys = []

    def __init__(self):
        self.messenger = Messenger()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        return True, "ok"

    def load_first_scicode_steps(self):
        path = Path("data/problems_all.jsonl")
        steps = []

        with path.open() as f:
            for line in f:
                obj = json.loads(line)
                sub_steps = obj.get("sub_steps", [])
                if not sub_steps:
                    continue

                step = sub_steps[0]
                if "function_header" not in step or "test_cases" not in step:
                    continue

                steps.append({
                    "problem": obj["problem_name"],
                    "sub_step": step["step_number"],
                    "function_header": step["function_header"],
                    "test_cases": step["test_cases"],
                })

        return steps

    def execute_candidate(self, code: str, tests: list[str]):
        env = {
            "__builtins__": {},
            "np": np,
            "numpy": np,
        }

        exec(code, env, env)

        callables = [v for v in env.values() if callable(v)]
        if not callables:
            raise RuntimeError("No callable function defined")

        results = []
        for test in tests:
            try:
                exec(test, env, env)
                results.append(True)
            except Exception:
                results.append(False)

        return results

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            request = EvalRequest.model_validate_json(input_text)
        except ValidationError:
            await updater.reject(
                new_agent_text_message(
                    "Invalid request format. Expected EvalRequest JSON."
                )
            )
            return

        ok, msg = self.validate_request(request)
        if not ok:
            await updater.reject(new_agent_text_message(msg))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                "Running SciCode benchmark (first sub-problem per problem)..."
            ),
        )

        steps = self.load_first_scicode_steps()

        total_tests = 0
        total_passed = 0
        details = []

        purple_url = str(request.participants["purple"])

        for idx, step in enumerate(steps, start=1):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"[{idx}/{len(steps)}] "
                    f"Evaluating {step['problem']} / {step['sub_step']}"
                ),
            )

            prompt = f"""
Write Python code that defines the following function exactly:

{step['function_header']}

The function must satisfy the provided test cases.
Do NOT print anything.
Return ONLY valid Python code.
"""

            try:
                code = await self.messenger.talk_to_agent(
                    message=prompt,
                    url=purple_url,
                    new_conversation=True,
                )

                results = self.execute_candidate(code, step["test_cases"])
                passed = sum(results)
                total = len(results)
            except Exception:
                passed = 0
                total = len(step["test_cases"])

            total_passed += passed
            total_tests += total

            details.append({
                "problem": step["problem"],
                "sub_step": step["sub_step"],
                "passed": passed,
                "total": total,
            })

        accuracy = total_passed / total_tests if total_tests > 0 else 0.0

        await updater.add_artifact(
            name="Result",
            parts=[
                Part(
                    root=DataPart(
                        data={
                            "num_problems": len(details),
                            "passed": total_passed,
                            "total": total_tests,
                            "accuracy": accuracy,
                            "details": details,
                        }
                    )
                )
            ],
        )

        await updater.complete()
