"""Results Evaluator MCP Server.

Provides tools for comparing LLM responses, scoring quality, and
recommending the best model for a given task.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

mcp = FastMCP("results-evaluator", description="Evaluates and compares LLM outputs")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ResponseEntry(BaseModel):
    """A single LLM response to evaluate."""

    provider: str = Field(..., description="Provider name (anthropic, openai, google, perplexity)")
    model: str = Field(..., description="Model identifier")
    content: str = Field(..., description="The response text")
    latency_seconds: float = Field(default=0.0, description="Response latency in seconds")
    usage: dict[str, int] = Field(default_factory=dict, description="Token usage info")


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_relevance(content: str, prompt: str) -> dict[str, Any]:
    """Estimate how relevant the response is to the original prompt."""
    prompt_words = set(prompt.lower().split())
    content_words = set(content.lower().split())
    # Jaccard-like overlap
    if not prompt_words:
        return {"score": 50, "reason": "Empty prompt — cannot measure relevance"}

    overlap = prompt_words & content_words
    ratio = len(overlap) / len(prompt_words)
    score = min(100, int(ratio * 100) + 30)  # baseline 30
    return {
        "score": max(0, min(100, score)),
        "keyword_overlap": len(overlap),
        "prompt_keywords": len(prompt_words),
    }


def _score_coherence(content: str) -> dict[str, Any]:
    """Score the structural coherence of a response."""
    score = 50
    reasons: list[str] = []

    sentences = [s.strip() for s in re.split(r"[.!?]+", content) if s.strip()]
    if len(sentences) >= 3:
        score += 15
        reasons.append("Multiple well-formed sentences")
    elif len(sentences) == 0:
        score -= 20
        reasons.append("No complete sentences detected")

    # Paragraph structure
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        score += 10
        reasons.append("Uses paragraph breaks")

    # Headers / lists
    if re.search(r"^#{1,3}\s", content, re.MULTILINE):
        score += 10
        reasons.append("Uses markdown headers")
    if re.search(r"^[-*]\s", content, re.MULTILINE):
        score += 5
        reasons.append("Uses bullet lists")
    if re.search(r"^\d+\.\s", content, re.MULTILINE):
        score += 5
        reasons.append("Uses numbered lists")

    return {"score": max(0, min(100, score)), "reasons": reasons}


def _score_depth(content: str) -> dict[str, Any]:
    """Score the depth and thoroughness of a response."""
    score = 40
    reasons: list[str] = []

    word_count = len(content.split())
    if word_count < 20:
        score -= 10
        reasons.append("Very short response")
    elif word_count < 100:
        reasons.append("Moderate length")
    elif word_count < 500:
        score += 15
        reasons.append("Detailed response")
    else:
        score += 25
        reasons.append("Comprehensive response")

    # Check for examples
    if "example" in content.lower() or "e.g." in content.lower():
        score += 10
        reasons.append("Includes examples")

    # Check for code blocks
    if "```" in content:
        score += 10
        reasons.append("Includes code blocks")

    # Check for caveats / nuance
    nuance_words = ["however", "although", "on the other hand", "caveat", "note that", "keep in mind"]
    nuance_count = sum(1 for w in nuance_words if w in content.lower())
    if nuance_count:
        score += min(nuance_count * 5, 15)
        reasons.append(f"Shows nuance ({nuance_count} qualifier(s))")

    return {"score": max(0, min(100, score)), "reasons": reasons}


def _score_accuracy_signals(content: str) -> dict[str, Any]:
    """Heuristic check for accuracy signals (hedging, citations, etc.)."""
    score = 50
    reasons: list[str] = []

    # Positive signals
    if re.search(r"https?://", content):
        score += 10
        reasons.append("Contains URLs / references")
    if re.search(r"\[\d+\]", content):
        score += 10
        reasons.append("Uses citation markers")
    if "according to" in content.lower():
        score += 5
        reasons.append("Attributes claims to sources")

    # Appropriate hedging
    hedge_words = ["likely", "probably", "may", "might", "it seems", "appears to"]
    hedge_count = sum(1 for w in hedge_words if w in content.lower())
    if hedge_count:
        score += min(hedge_count * 3, 10)
        reasons.append("Uses appropriate hedging")

    return {"score": max(0, min(100, score)), "reasons": reasons}


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def compare_responses(
    prompt: str,
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare multiple LLM responses side-by-side on quality dimensions.

    Each item in *responses* should have keys: provider, model, content,
    and optionally latency_seconds and usage.
    """
    if len(responses) < 2:
        return {"error": "At least two responses are required for comparison"}

    comparisons: list[dict[str, Any]] = []

    for resp in responses:
        entry = ResponseEntry(**resp)
        relevance = _score_relevance(entry.content, prompt)
        coherence = _score_coherence(entry.content)
        depth = _score_depth(entry.content)
        accuracy = _score_accuracy_signals(entry.content)

        overall = round(
            (relevance["score"] + coherence["score"] + depth["score"] + accuracy["score"]) / 4
        )

        comparisons.append({
            "provider": entry.provider,
            "model": entry.model,
            "overall_score": overall,
            "dimensions": {
                "relevance": relevance,
                "coherence": coherence,
                "depth": depth,
                "accuracy_signals": accuracy,
            },
            "word_count": len(entry.content.split()),
            "latency_seconds": entry.latency_seconds,
            "usage": entry.usage,
        })

    # Sort best → worst
    comparisons.sort(key=lambda c: c["overall_score"], reverse=True)

    return {
        "prompt": prompt,
        "comparisons": comparisons,
        "ranking": [
            {"rank": i + 1, "provider": c["provider"], "model": c["model"], "score": c["overall_score"]}
            for i, c in enumerate(comparisons)
        ],
    }


@mcp.tool()
async def score_response_quality(
    prompt: str,
    response_content: str,
    provider: str = "unknown",
    model: str = "unknown",
) -> dict[str, Any]:
    """Score a single LLM response on relevance, coherence, depth, and
    accuracy signals.  Each dimension is scored 0-100.
    """
    relevance = _score_relevance(response_content, prompt)
    coherence = _score_coherence(response_content)
    depth = _score_depth(response_content)
    accuracy = _score_accuracy_signals(response_content)

    overall = round(
        (relevance["score"] + coherence["score"] + depth["score"] + accuracy["score"]) / 4
    )

    quality_label = "poor"
    if overall >= 80:
        quality_label = "excellent"
    elif overall >= 65:
        quality_label = "good"
    elif overall >= 50:
        quality_label = "fair"

    return {
        "provider": provider,
        "model": model,
        "overall_score": overall,
        "quality_label": quality_label,
        "dimensions": {
            "relevance": relevance,
            "coherence": coherence,
            "depth": depth,
            "accuracy_signals": accuracy,
        },
        "word_count": len(response_content.split()),
    }


@mcp.tool()
async def recommend_best_model(
    task_type: str,
    priorities: list[str] | None = None,
) -> dict[str, Any]:
    """Recommend the best LLM model for a given task type and priority set.

    Supported task types: creative_writing, code_generation, data_analysis,
    summarization, question_answering, translation, general.

    Priorities can include: quality, speed, cost, accuracy, creativity.
    """
    if priorities is None:
        priorities = ["quality"]

    priority_set = set(p.lower() for p in priorities)

    model_profiles: dict[str, dict[str, Any]] = {
        "anthropic": {
            "model": "claude-sonnet-4-20250514",
            "strengths": ["reasoning", "safety", "instruction_following", "code"],
            "quality": 90,
            "speed": 70,
            "cost": 65,
            "accuracy": 90,
            "creativity": 80,
        },
        "openai": {
            "model": "gpt-4o",
            "strengths": ["broad_knowledge", "tool_use", "multimodal", "code"],
            "quality": 88,
            "speed": 75,
            "cost": 60,
            "accuracy": 88,
            "creativity": 82,
        },
        "google": {
            "model": "gemini-1.5-flash",
            "strengths": ["speed", "long_context", "cost_effective", "multimodal"],
            "quality": 78,
            "speed": 95,
            "cost": 90,
            "accuracy": 78,
            "creativity": 72,
        },
        "perplexity": {
            "model": "sonar",
            "strengths": ["web_search", "current_info", "citations"],
            "quality": 75,
            "speed": 80,
            "cost": 70,
            "accuracy": 82,
            "creativity": 60,
        },
    }

    # Task-specific boosts
    task_boosts: dict[str, dict[str, int]] = {
        "code_generation": {"anthropic": 10, "openai": 8},
        "creative_writing": {"openai": 5, "anthropic": 5},
        "data_analysis": {"openai": 5, "anthropic": 5},
        "summarization": {"google": 5, "anthropic": 3},
        "question_answering": {"perplexity": 15, "anthropic": 3},
        "translation": {"openai": 5, "google": 5},
    }

    boosts = task_boosts.get(task_type, {})

    scored: list[dict[str, Any]] = []
    for provider, profile in model_profiles.items():
        # Weighted sum based on priorities
        total = 0.0
        count = 0
        for p in priority_set:
            if p in profile:
                total += profile[p]
                count += 1
        if count == 0:
            total = profile["quality"]
            count = 1

        avg = total / count + boosts.get(provider, 0)

        scored.append({
            "provider": provider,
            "model": profile["model"],
            "score": round(avg, 1),
            "strengths": profile["strengths"],
        })

    scored.sort(key=lambda s: s["score"], reverse=True)

    return {
        "task_type": task_type,
        "priorities": list(priority_set),
        "recommendations": scored,
        "best_pick": scored[0],
        "reasoning": (
            f"For {task_type} tasks with priorities {priorities}, "
            f"{scored[0]['provider']} ({scored[0]['model']}) scores highest "
            f"at {scored[0]['score']}."
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
