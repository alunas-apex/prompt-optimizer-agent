"""Prompt Optimizer Agent — main entry point.

Orchestrates the MCP servers to provide an end-to-end prompt optimization
workflow accessible from the command line.

Usage examples
--------------
    # Analyze a prompt
    python -m agent_core.main analyze "Write me a story"

    # Optimize a prompt (3 rounds by default)
    python -m agent_core.main optimize "Write me a story"

    # Optimize and test across all LLMs
    python -m agent_core.main optimize --test "Write me a story"

    # Test a prompt across all LLMs
    python -m agent_core.main test "Explain quantum computing"

    # Get recommended parameters for a task type
    python -m agent_core.main params code_generation

    # Get a system prompt suggestion
    python -m agent_core.main system-prompt creative_writing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure project root is on sys.path so agent_core is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent_core.config import load_config, setup_logging, validate_api_keys
from agent_core.optimizer import PromptOptimizer

load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pretty(obj: Any) -> str:
    """Return a nicely-formatted JSON string."""
    return json.dumps(obj, indent=2, default=str)


def _print_api_key_status() -> None:
    """Print which API keys are configured."""
    status = validate_api_keys()
    print("\nAPI key status:")
    for provider, available in status.items():
        marker = "OK" if available else "MISSING"
        print(f"  {provider:>12}: {marker}")
    print()


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

async def cmd_analyze(optimizer: PromptOptimizer, args: argparse.Namespace) -> None:
    """Analyze prompt quality without changing anything."""
    prompt = " ".join(args.prompt)
    print(f"\nAnalyzing prompt: {prompt!r}\n")
    result = await optimizer.analyze(prompt)
    print(_pretty(result))


async def cmd_optimize(optimizer: PromptOptimizer, args: argparse.Namespace) -> None:
    """Run the optimization pipeline."""
    prompt = " ".join(args.prompt)
    print(f"\nOptimizing prompt: {prompt!r}")
    print(f"Rounds: {args.rounds}, Task type: {args.task_type or 'auto-detect'}")
    if args.test:
        print("LLM testing: enabled")
    print()

    result = await optimizer.optimize(
        prompt=prompt,
        task_type=args.task_type,
        rounds=args.rounds,
        test_with_llms=args.test,
    )

    print("=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"\nOriginal prompt:\n  {result['original_prompt']}")
    print(f"\nFinal prompt:\n  {result['final_prompt']}")
    print(f"\nRounds completed: {result['rounds_completed']}")

    final = result.get("final_analysis", {})
    if final:
        print(f"Final quality score: {final.get('overall_score', 'N/A')}")

    if args.verbose:
        print(f"\nFull results:\n{_pretty(result)}")


async def cmd_test(optimizer: PromptOptimizer, args: argparse.Namespace) -> None:
    """Test a prompt across all LLM providers."""
    prompt = " ".join(args.prompt)
    print(f"\nTesting prompt across all LLMs: {prompt!r}\n")

    result = await optimizer.test_prompt(
        prompt=prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    evaluation = result.get("evaluation", {})
    ranking = evaluation.get("ranking", [])

    if ranking:
        print("Ranking:")
        for entry in ranking:
            print(f"  #{entry['rank']} {entry['provider']} ({entry['model']}) — score {entry['score']}")
    else:
        print("Not enough responses to rank.")

    if args.verbose:
        print(f"\nFull results:\n{_pretty(result)}")


async def cmd_params(optimizer: PromptOptimizer, args: argparse.Namespace) -> None:
    """Show recommended parameters for a task type."""
    result = await optimizer.suggest_parameters(args.task_type)
    print(f"\nRecommended parameters for '{args.task_type}':\n")
    print(_pretty(result))


async def cmd_system_prompt(optimizer: PromptOptimizer, args: argparse.Namespace) -> None:
    """Show a recommended system prompt for a task type."""
    result = await optimizer.suggest_system_prompt(args.task_type)
    print(f"\nSystem prompt for '{args.task_type}':\n")
    print(_pretty(result))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt-optimizer",
        description="Prompt Optimizer Agent — improve your LLM prompts",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to agent_config.yaml",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print full JSON results",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze prompt quality")
    p_analyze.add_argument("prompt", nargs="+", help="The prompt to analyze")

    # optimize
    p_opt = sub.add_parser("optimize", help="Optimize a prompt")
    p_opt.add_argument("prompt", nargs="+", help="The prompt to optimize")
    p_opt.add_argument("--rounds", type=int, default=None, help="Number of optimization rounds")
    p_opt.add_argument("--task-type", type=str, default="", help="Task type hint")
    p_opt.add_argument("--test", action="store_true", help="Also test with all LLMs")

    # test
    p_test = sub.add_parser("test", help="Test a prompt across all LLMs")
    p_test.add_argument("prompt", nargs="+", help="The prompt to test")
    p_test.add_argument("--temperature", type=float, default=None)
    p_test.add_argument("--max-tokens", type=int, default=None)

    # params
    p_params = sub.add_parser("params", help="Recommend model parameters")
    p_params.add_argument("task_type", help="Task type (e.g. code_generation)")

    # system-prompt
    p_sys = sub.add_parser("system-prompt", help="Suggest a system prompt")
    p_sys.add_argument("task_type", help="Task type (e.g. creative_writing)")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COMMAND_MAP = {
    "analyze": cmd_analyze,
    "optimize": cmd_optimize,
    "test": cmd_test,
    "params": cmd_params,
    "system-prompt": cmd_system_prompt,
}


async def async_main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    _print_api_key_status()

    optimizer = PromptOptimizer(config)

    handler = COMMAND_MAP.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    await handler(optimizer, args)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
