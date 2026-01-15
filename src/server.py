import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    # Fill in your agent card
    # See: https://a2a-protocol.org/latest/tutorials/python/3-agent-skills-and-card/
    
    skill = AgentSkill(
        id="scicode-a2a-eval",
        name="SciCode Agent-to-Agent Evaluation",
        description=(
            "Evaluates a solver agent on real scientific coding tasks from the SciCode benchmark "
            "(arXiv:2407.13168) using Agent-to-Agent interaction, restricted code execution, "
            "and test-based scoring."
        ),
        tags=["a2a", "benchmark", "scicode", "scientific-coding", "evaluation"],
        examples=[
            "Implement integrate_DOS for the linear tetrahedron method",
        ],
    )

    agent_card = AgentCard(
        name="Reflena Green Agent",
        description=(
            "An Agent-to-Agent evaluation agent that benchmarks solver agents on real scientific "
            "programming tasks from the SciCode dataset. The agent orchestrates task prompts, "
            "executes returned code in a restricted environment, runs official test cases, "
            "aggregates results, and reports scores to an AgentBeats leaderboard."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
