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
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


class Agent:
    # REQUIRED by AgentBeats
    required_roles = ["purple"]
    required_config_keys = []

    def __init__(self):
        self.messenger = Messenger()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        return True, "ok"
    
    def load_integrate_dos_problem(self):
        path = Path("data/problems_all.jsonl")

        with path.open() as f:
            for line in f:
                obj = json.loads(line)
                if obj["problem_name"] == "linear_tetrahedron_method":
                    for step in obj["sub_steps"]:
                        if step["step_number"] == "17.2":
                            return step

        raise RuntimeError("integrate_DOS sub-step not found")
    
    def execute_candidate(self, code: str, tests: list[str]):
        safe_globals = {
            "__builtins__": {},
            "np": np,
            "numpy": np,
        }

        safe_locals = {}

        # Compile + execute candidate code
        exec(code, safe_globals, safe_locals)

        if "integrate_DOS" not in safe_locals:
            raise RuntimeError("integrate_DOS not defined")

        integrate_DOS = safe_locals["integrate_DOS"]

        results = []
        for test in tests:
            try:
                # `target` is provided by the dataset during evaluation
                local_env = {
                    "integrate_DOS": integrate_DOS,
                    "np": np,
                    "target": None,
                }
                exec(test, {}, local_env)
                results.append(True)
            except Exception:
                results.append(False)

        return results


    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        # Handle non-evaluation messages (e.g. "Hello" from tests)
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

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Running SciCode integrate_DOS benchmark...")
        )

        # Load benchmark
        step = self.load_integrate_dos_problem()
        tests = step["test_cases"]

        # Build prompt
        prompt = f"""
        Write Python code that defines the following function exactly:

        {step["function_header"]}

        The function must satisfy the following test cases.
        Do NOT print anything.
        Return ONLY valid Python code.
        """

        # Send to purple agent
        purple_url = str(request.participants["purple"])
        code = await self.messenger.talk_to_agent(
            message=prompt,
            url=purple_url,
            new_conversation=True,
        )

        # Execute + evaluate
        try:
            results = self.execute_candidate(code, tests)
            passed = sum(results)
            total = len(results)
            accuracy = passed / total
        except Exception as e:
            passed = 0
            total = len(tests)
            accuracy = 0.0

        # Emit artifact
        await updater.add_artifact(
            name="Result",
            parts=[
                Part(root=TextPart(text="SciCode integrate_DOS evaluation completed")),
                Part(
                    root=DataPart(
                        data={
                            "problem": "linear_tetrahedron_method",
                            "sub_step": "17.2",
                            "passed": passed,
                            "total": total,
                            "accuracy": accuracy,
                        }
                    )
                ),
            ],
        )

        await updater.complete()