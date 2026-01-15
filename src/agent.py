from typing import Any
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


class Agent:
    # REQUIRED by AgentBeats
    required_roles = ["purple"]
    required_config_keys = ["task", "input"]

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
            new_agent_text_message("Running A2A evaluation...")
        )

        purple_url = str(request.participants["purple"])
        task = request.config["task"]
        user_input = request.config["input"]

        # Minimal deterministic benchmark
        prompt = f"Reverse this string: {user_input}"
        response = await self.messenger.talk_to_agent(
            message=prompt,
            url=purple_url,
            new_conversation=True,
        )

        expected = user_input[::-1]
        actual = response.strip()

        score = 1 if actual == expected else 0
        accuracy = float(score)

        await updater.add_artifact(
            name="Result",
            parts=[
                Part(root=TextPart(text="A2A evaluation completed.")),
                Part(
                    root=DataPart(
                        data={
                            "task": task,
                            "input": user_input,
                            "expected": expected,
                            "actual": actual,
                            "score": score,
                            "accuracy": accuracy,
                        }
                    )
                ),
            ],
        )

        await updater.complete()
