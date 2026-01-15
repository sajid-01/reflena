from typing import Any
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

import json
import numpy as np
from pathlib import Path


class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl]
    config: dict[str, Any]


class Agent:
    required_roles = ["purple"]
    required_config_keys = []

    def __init__(self):
        self.messenger = Messenger()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants)
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        return True, "ok"

    def load_scicode_steps(self, limit: int = 3):
        path = Path("data/problems_all.jsonl")
        steps = []

        with path.open() as f:
            for line in f:
                obj = json.loads(line)
                for step in obj.get("sub_steps", []):
                    if "function_header" in step and "test_cases" in step:
                        steps.append({
                            "problem": obj["problem_name"],
                            "sub_step": step["step_number"],
                            "function_header": step["function_header"],
                            "test_cases": step["test_cases"],
                        })
                        if len(steps) >= limit:
                            return steps
        return steps

    def extract_function_name(self, function_header: str) -> str:
        header = function_header.strip()
        name = header.split("def ", 1)[1].split("(", 1)[0].strip()
        if not name.isidentifier():
            raise RuntimeError(f"Invalid function name: {name}")
        return name

    def execute_candidate(self, code: str, function_name: str, tests: list[str]):
        safe_globals = {
            "__builtins__": {},
            "np": np,
            "numpy": np,
        }
        safe_locals = {}

        exec(code, safe_globals, safe_locals)

        if function_name not in safe_locals:
            raise RuntimeError(f"{function_name} not defined")

        fn = safe_locals[function_name]

        results = []
        for test in tests:
            try:
                env = {
                    function_name: fn,
                    "np": np,
                }
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
                new_agent_text_message("Invalid request format")
            )
            return

        ok, msg = self.validate_request(request)
        if not ok:
            await updater.reject(new_agent_text_message(msg))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Running SciCode multi-task benchmark...")
        )

        steps = self.load_scicode_steps(limit=3)

        total_tests = 0
        total_passed = 0
        details = []

        purple_url = str(request.participants["purple"])

        for step in steps:
            function_name = self.extract_function_name(step["function_header"])

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Evaluating {step['problem']} / {step['sub_step']} ({function_name})"
                )
            )

            prompt = f"""
Write Python code that defines the following function exactly:

{step['function_header']}

The function must satisfy the test cases.
Do NOT print anything.
Return ONLY valid Python code.
"""

            try:
                code = await self.messenger.talk_to_agent(
                    message=prompt,
                    url=purple_url,
                    new_conversation=True,
                )

                results = self.execute_candidate(
                    code,
                    function_name=function_name,
                    tests=step["test_cases"],
                )
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
                "function": function_name,
                "passed": passed,
                "total": total,
            })

        accuracy = total_passed / total_tests if total_tests else 0.0

        await updater.add_artifact(
            name="Result",
            parts=[
                Part(
                    root=DataPart(
                        data={
                            "num_tasks": len(details),
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
