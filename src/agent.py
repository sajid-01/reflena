from typing import Any
from pathlib import Path
import json
import numpy as np

from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl]
    config: dict[str, Any]


class Agent:
    required_roles = ["purple"]
    required_config_keys = []

    def __init__(self):
        self.messenger = Messenger()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing = set(self.required_roles) - set(request.participants)
        if missing:
            return False, f"Missing roles: {missing}"
        return True, "ok"

    def load_scicode_problems(self, limit: int):
        path = Path("data/problems_all.jsonl")
        problems = []

        with path.open() as f:
            for line in f:
                obj = json.loads(line)
                steps = [
                    step for step in obj.get("sub_steps", [])
                    if "function_header" in step and "test_cases" in step
                ]
                if not steps:
                    continue

                problems.append({
                    "problem": obj["problem_name"],
                    "steps": steps,
                })

                if len(problems) >= limit:
                    break

        return problems

    def build_shared_env(self):
        def safe_import(name, *args, **kwargs):
            if name.startswith("scicode"):
                return __import__(name, *args, **kwargs)
            raise ImportError(name)

        return {
            "__builtins__": {
                "__import__": safe_import,
            },
            "np": np,
            "numpy": np,
        }

    def execute_problem(self, code: str, steps: list[dict], debug: bool):
        env = self.build_shared_env()

        # Execute candidate code
        exec(code, env, env)

        total = 0
        passed = 0
        details = []
        debug_tests = []

        for step in steps:
            step_env = dict(env)
            step_passed = 0
            step_total = len(step["test_cases"])

            func_name = step["function_header"].split("def ", 1)[1].split("(")[0]

            if func_name not in env:
                total += step_total

                details.append({
                    "sub_step": step["step_number"],
                    "function": func_name,
                    "passed": 0,
                    "total": step_total,
                })

                if debug:
                    for test in step["test_cases"]:
                        debug_tests.append({
                            "sub_step": step["step_number"],
                            "test": test,
                            "passed": False,
                            "error": f"Function '{func_name}' not defined",
                            "target": None,
                        })

                continue 

            for test in step["test_cases"]:
                try:
                    if "assert" in test:
                        setup, assertion = test.split("assert", 1)
                        exec(setup, step_env, step_env)

                        step_env["target"] = None
                        for ref in ("ref1", "ref2", "ref3", "ref4"):
                            if ref in step_env:
                                step_env["target"] = step_env[ref]

                        exec("assert" + assertion, step_env, step_env)
                    else:
                        exec(test, step_env, step_env)

                    step_passed += 1
                    passed += 1
                    ok = True
                    err = None

                except Exception as e:
                    ok = False
                    err = str(e)

                if debug:
                    debug_tests.append({
                        "sub_step": step["step_number"],
                        "test": test,
                        "passed": ok,
                        "error": err,
                        "target": step_env.get("target"),
                    })

            total += step_total

            details.append({
                "sub_step": step["step_number"],
                "function": func_name,
                "passed": step_passed,
                "total": step_total,
            })

        return passed, total, details, debug_tests

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            request = EvalRequest.model_validate_json(input_text)
        except ValidationError:
            await updater.reject(new_agent_text_message("Invalid request format."))
            return

        ok, msg = self.validate_request(request)
        if not ok:
            await updater.reject(new_agent_text_message(msg))
            return

        config = request.config or {}
        num_tasks = int(config.get("num_tasks", 1))
        debug = bool(config.get("debug", False))

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Running SciCode benchmark (num_tasks={num_tasks}, debug={debug})"
            )
        )

        problems = self.load_scicode_problems(limit=num_tasks)
        purple_url = str(request.participants["purple"])

        grand_passed = 0
        grand_total = 0
        all_details = []

        for prob in problems:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Evaluating problem: {prob['problem']}")
            )

            headers = "\n".join(step["function_header"] for step in prob["steps"])

            prompt = f"""
You are solving a SciCode scientific programming benchmark.

Implement ALL of the following Python functions.
They may call each other.

Rules:
- Define EXACT function signatures
- No printing
- No markdown
- No extra helpers
- Deterministic & numerically stable
- Assume numpy is available as np

Functions:
{headers}

Return ONLY valid Python code containing ALL function definitions.
"""

            code = await self.messenger.talk_to_agent(
                message=prompt,
                url=purple_url,
                new_conversation=True,
            )

            passed, total, details, debug_tests = self.execute_problem(
                code=code,
                steps=prob["steps"],
                debug=debug,
            )

            grand_passed += passed
            grand_total += total

            all_details.append({
                "problem": prob["problem"],
                "passed": passed,
                "total": total,
                "details": details,
            })

            if debug:
                await updater.add_artifact(
                    name=f"Debug-{prob['problem']}",
                    parts=[
                        Part(
                            root=DataPart(
                                data={
                                    "code": code,
                                    "tests": debug_tests,
                                }
                            )
                        )
                    ],
                )

        accuracy = grand_passed / grand_total if grand_total else 0.0

        await updater.add_artifact(
            name="Result",
            parts=[
                Part(
                    root=DataPart(
                        data={
                            "num_tasks": len(all_details),
                            "passed": grand_passed,
                            "total": grand_total,
                            "accuracy": accuracy,
                            "details": all_details,
                        }
                    )
                )
            ],
        )

        await updater.complete()
