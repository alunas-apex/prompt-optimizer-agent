"""Prompt Optimizer — core optimization logic.

Wires together the three MCP servers (prompt-analyzer, llm-router,
results-evaluator) to iteratively improve a user's prompt.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from agent_core.config import AgentConfig, load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy MCP server imports
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _import_module_from_path(name: str, path: Path):
    """Import a Python module from an absolute file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_analyzer():
    """Return the prompt-analyzer module."""
    return _import_module_from_path(
        "prompt_analyzer_server",
        _PROJECT_ROOT / "mcp-servers" / "prompt-analyzer" / "server.py",
    )


def _get_router():
    """Return the llm-router module."""
    return _import_module_from_path(
        "llm_router_server",
        _PROJECT_ROOT / "mcp-servers" / "llm-router" / "server.py",
    )


def _get_evaluator():
    """Return the results-evaluator module."""
    return _import_module_from_path(
        "results_evaluator_server",
        _PROJECT_ROOT / "mcp-servers" / "results-evaluator" / "server.py",
    )


# ---------------------------------------------------------------------------
# Optimization pipeline
# ---------------------------------------------------------------------------

class OptimizationResult:
    """Container for a single round of optimisation."""

    def __init__(
        self,
        round_number: int,
        original_prompt: str,
        optimized_prompt: str,
        analysis: dict[str, Any],
        suggestions: list[str],
        responses: dict[str, Any] | None = None,
        evaluation: dict[str, Any] | None = None,
    ):
        self.round_number = round_number
        self.original_prompt = original_prompt
        self.optimized_prompt = optimized_prompt
        self.analysis = analysis
        self.suggestions = suggestions
        self.responses = responses
        self.evaluation = evaluation

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_number,
            "original_prompt": self.original_prompt,
            "optimized_prompt": self.optimized_prompt,
            "analysis": self.analysis,
            "suggestions": self.suggestions,
            "responses": self.responses,
            "evaluation": self.evaluation,
        }


class PromptOptimizer:
    """Iteratively optimizes a prompt using the MCP tool pipeline.

    Pipeline per round:
        1. Analyze prompt quality  (prompt-analyzer)
        2. Generate optimized version  (prompt-analyzer)
        3. Optionally test across LLMs  (llm-router)
        4. Evaluate responses  (results-evaluator)
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or load_config()
        self._analyzer = _get_analyzer()
        self._router = _get_router()
        self._evaluator = _get_evaluator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self, prompt: str) -> dict[str, Any]:
        """Return a quality analysis of *prompt* without optimizing."""
        return await self._analyzer.analyze_prompt_quality(prompt)

    async def optimize(
        self,
        prompt: str,
        task_type: str = "",
        rounds: int | None = None,
        test_with_llms: bool = False,
    ) -> dict[str, Any]:
        """Run the full optimisation pipeline.

        Parameters
        ----------
        prompt:
            The original user prompt.
        task_type:
            Optional task-type hint (e.g. ``code_generation``).
        rounds:
            Number of optimise-evaluate cycles (defaults to config value).
        test_with_llms:
            If ``True``, send the optimised prompt to all LLMs and evaluate.
        """
        rounds = rounds or self.config.optimization_rounds
        results: list[OptimizationResult] = []
        current_prompt = prompt

        for i in range(1, rounds + 1):
            logger.info("=== Optimization round %d/%d ===", i, rounds)
            result = await self._run_round(
                round_number=i,
                prompt=current_prompt,
                task_type=task_type,
                test_with_llms=test_with_llms,
            )
            results.append(result)

            # Use the optimised prompt as input for the next round
            current_prompt = result.optimized_prompt

            # Stop early if quality is already high
            overall = result.analysis.get("overall_score", 0)
            if overall >= 90:
                logger.info("Prompt quality ≥ 90 — stopping early")
                break

        return {
            "original_prompt": prompt,
            "final_prompt": current_prompt,
            "rounds_completed": len(results),
            "rounds": [r.to_dict() for r in results],
            "final_analysis": results[-1].analysis if results else {},
        }

    async def suggest_parameters(self, task_type: str) -> dict[str, Any]:
        """Return recommended model parameters for *task_type*."""
        return await self._analyzer.suggest_model_parameters(task_type)

    async def suggest_system_prompt(self, task_type: str) -> dict[str, Any]:
        """Return a recommended system prompt for *task_type*."""
        return await self._analyzer.suggest_system_prompt(task_type)

    async def test_prompt(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Send *prompt* to all configured LLMs and return evaluated results."""
        temp = temperature if temperature is not None else self.config.model_defaults.temperature
        tokens = max_tokens if max_tokens is not None else self.config.model_defaults.max_tokens

        llm_results = await self._router.call_all_llms(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temp,
            max_tokens=tokens,
        )

        # Build response list for the evaluator
        response_list = [
            {
                "provider": provider,
                "model": data.get("model", "unknown"),
                "content": data.get("content", ""),
                "latency_seconds": data.get("latency_seconds", 0),
                "usage": data.get("usage", {}),
            }
            for provider, data in llm_results.get("responses", {}).items()
            if data.get("content")
        ]

        evaluation: dict[str, Any] = {}
        if len(response_list) >= 2:
            evaluation = await self._evaluator.compare_responses(
                prompt=prompt,
                responses=response_list,
            )

        return {
            "prompt": prompt,
            "llm_results": llm_results,
            "evaluation": evaluation,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_round(
        self,
        round_number: int,
        prompt: str,
        task_type: str,
        test_with_llms: bool,
    ) -> OptimizationResult:
        """Execute a single optimise → (test) → evaluate round."""
        # 1. Analyze
        analysis = await self._analyzer.analyze_prompt_quality(prompt)
        logger.info("Round %d — quality score: %s", round_number, analysis.get("overall_score"))

        # 2. Optimize
        opt = await self._analyzer.optimize_prompt(prompt, task_type)
        optimized = opt.get("optimized_prompt", prompt)
        suggestions = opt.get("suggestions", [])

        responses: dict[str, Any] | None = None
        evaluation: dict[str, Any] | None = None

        # 3. Optionally test with LLMs
        if test_with_llms:
            temp = self.config.model_defaults.temperature
            tokens = self.config.model_defaults.max_tokens

            responses = await self._router.call_all_llms(
                prompt=optimized,
                temperature=temp,
                max_tokens=tokens,
            )

            # 4. Evaluate
            resp_list = [
                {
                    "provider": prov,
                    "model": d.get("model", "unknown"),
                    "content": d.get("content", ""),
                    "latency_seconds": d.get("latency_seconds", 0),
                    "usage": d.get("usage", {}),
                }
                for prov, d in responses.get("responses", {}).items()
                if d.get("content")
            ]
            if len(resp_list) >= 2:
                evaluation = await self._evaluator.compare_responses(
                    prompt=optimized,
                    responses=resp_list,
                )

        return OptimizationResult(
            round_number=round_number,
            original_prompt=prompt,
            optimized_prompt=optimized,
            analysis=analysis,
            suggestions=suggestions,
            responses=responses,
            evaluation=evaluation,
        )
