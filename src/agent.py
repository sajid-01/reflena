from typing import Any
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
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

    def extract_target(self, test: str):
        """
        Extracts `target` from lines like:
        ref1 = ...
        """
        for line in test.splitlines():
            if line.strip().startswith("ref"):
                _, rhs = line.split("=", 1)
                return eval(rhs.strip(), {"np": np})
        return None

    def execute_candidate(
        self,
        code: str,
        function_name: str,
        tests: list[str],
        shared_env: dict,
        debug: bool = False,
    ):
        exec(code, shared_env, shared_env)

        if function_name not in shared_env:
            raise RuntimeError(f"{function_name} not defined")

        results = []
        debug_info = {
            "function_found": True,
            "tests": [],
        }

        for test in tests:
            record = {"test": test, "passed": False}

            try:
                target_value = self.extract_target(test)
                shared_env["target"] = target_value

                exec(test, shared_env, shared_env)

                record["passed"] = True
                record["target"] = target_value
                results.append(True)

            except Exception as e:
                record["error"] = str(e)
                results.append(False)

            debug_info["tests"].append(record)

        return results, debug_info

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            request = EvalRequest.model_validate_json(input_text)
        except ValidationError:
            await updater.reject(
                new_agent_text_message("Invalid request format. Expected EvalRequest JSON.")
            )
            return

        ok, msg = self.validate_request(request)
        if not ok:
            await updater.reject(new_agent_text_message(msg))
            return

        debug = bool(request.config.get("debug", False))
        num_tasks = int(request.config.get("num_tasks", 3))

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Running SciCode benchmark (num_tasks={num_tasks}, debug={debug})"
            )
        )

        steps = self.load_scicode_steps(limit=num_tasks)

        shared_env = {
            "__builtins__": {},
            "np": np,
            "numpy": np,
        }

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
You are solving a scientific programming task from the SciCode benchmark.

Write Python code that defines the following function EXACTLY:

{step['function_header']}

Rules:
- Do NOT print anything.
- Do NOT include extra comments or examples.
- Assume `numpy` is available as `np`.
- Return ONLY valid Python code.
"""

            try:
                code = await self.messenger.talk_to_agent(
                    message=prompt,
                    url=purple_url,
                    new_conversation=True,
                )

                results, debug_info = self.execute_candidate(
                    code=code,
                    function_name=function_name,
                    tests=step["test_cases"],
                    shared_env=shared_env,
                    debug=debug,
                )

                passed = sum(results)
                total = len(results)

                if debug:
                    await updater.add_artifact(
                        name=f"Debug-{step['problem']}-{step['sub_step']}",
                        parts=[
                            Part(
                                root=DataPart(
                                    data={
                                        "function": function_name,
                                        "code": code,
                                        "execution": debug_info,
                                    }
                                )
                            )
                        ],
                    )

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
