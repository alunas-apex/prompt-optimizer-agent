"""Prompt Analyzer MCP Server.

Provides tools for analysing prompt quality, suggesting improvements,
generating system prompts, and recommending model parameters.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

mcp = FastMCP("prompt-analyzer", description="Analyzes and optimizes prompt quality")


# ---------------------------------------------------------------------------
# Constants & heuristics
# ---------------------------------------------------------------------------

TASK_TYPES = {
    "creative_writing": {
        "description": "Fiction, poetry, storytelling",
        "system_prompt": (
            "You are a talented creative writer. Produce vivid, engaging, and "
            "original prose. Pay attention to narrative structure, character "
            "development, and literary style."
        ),
        "temperature": 0.9,
        "max_tokens": 2048,
    },
    "code_generation": {
        "description": "Writing or debugging code",
        "system_prompt": (
            "You are an expert software engineer. Write clean, efficient, and "
            "well-documented code. Follow best practices and include error "
            "handling. Explain your reasoning when appropriate."
        ),
        "temperature": 0.2,
        "max_tokens": 4096,
    },
    "data_analysis": {
        "description": "Analysing datasets, statistics, insights",
        "system_prompt": (
            "You are a data analysis expert. Provide accurate statistical "
            "analysis, identify patterns, and present insights clearly. Use "
            "precise numbers and explain methodology."
        ),
        "temperature": 0.3,
        "max_tokens": 2048,
    },
    "summarization": {
        "description": "Condensing long texts into summaries",
        "system_prompt": (
            "You are an expert summarizer. Create concise, accurate summaries "
            "that capture the key points without losing important nuance. "
            "Maintain the original tone and intent."
        ),
        "temperature": 0.3,
        "max_tokens": 1024,
    },
    "question_answering": {
        "description": "Factual Q&A, research",
        "system_prompt": (
            "You are a knowledgeable research assistant. Provide accurate, "
            "well-sourced answers. When uncertain, clearly state your "
            "confidence level and suggest ways to verify the information."
        ),
        "temperature": 0.4,
        "max_tokens": 1024,
    },
    "translation": {
        "description": "Translating between languages",
        "system_prompt": (
            "You are a professional translator. Produce natural, culturally "
            "appropriate translations that preserve the meaning, tone, and "
            "style of the original text."
        ),
        "temperature": 0.3,
        "max_tokens": 2048,
    },
    "general": {
        "description": "General-purpose assistance",
        "system_prompt": (
            "You are a helpful, accurate, and thoughtful assistant. Provide "
            "clear, well-structured responses. Ask clarifying questions when "
            "the request is ambiguous."
        ),
        "temperature": 0.7,
        "max_tokens": 1024,
    },
}

QUALITY_INDICATORS = {
    "specificity_keywords": [
        "specifically", "exactly", "precisely", "in detail",
        "step by step", "for example", "such as",
    ],
    "structure_markers": [
        "first", "second", "then", "finally", "step",
        "1.", "2.", "3.", "-", "*",
    ],
    "context_signals": [
        "context:", "background:", "given that", "assuming",
        "in the context of", "for the purpose of",
    ],
    "constraint_words": [
        "must", "should", "limit", "constraint", "requirement",
        "format", "length", "tone", "style",
    ],
    "role_indicators": [
        "act as", "you are", "pretend", "imagine you",
        "as a", "in the role of",
    ],
    "output_format_cues": [
        "json", "markdown", "bullet", "table", "list",
        "csv", "xml", "format",
    ],
    "vague_patterns": [
        "do something", "help me", "write something",
        "tell me about", "explain", "can you",
    ],
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_clarity(prompt: str) -> dict[str, Any]:
    """Score how clear and unambiguous the prompt is (0-100)."""
    score = 50  # baseline
    lower = prompt.lower()
    reasons: list[str] = []

    # Penalise very short prompts
    word_count = len(prompt.split())
    if word_count < 5:
        score -= 20
        reasons.append("Prompt is very short — may lack necessary detail")
    elif word_count > 15:
        score += 10
        reasons.append("Adequate length for context")

    # Reward question marks / clear intent
    if "?" in prompt:
        score += 10
        reasons.append("Contains a clear question")

    # Penalise vague language
    vague_hits = sum(1 for p in QUALITY_INDICATORS["vague_patterns"] if p in lower)
    if vague_hits:
        score -= vague_hits * 5
        reasons.append(f"Contains {vague_hits} vague phrase(s)")

    # Reward specificity keywords
    spec_hits = sum(1 for kw in QUALITY_INDICATORS["specificity_keywords"] if kw in lower)
    if spec_hits:
        score += spec_hits * 5
        reasons.append(f"Contains {spec_hits} specificity keyword(s)")

    return {"score": max(0, min(100, score)), "reasons": reasons}


def _score_specificity(prompt: str) -> dict[str, Any]:
    """Score how specific and well-defined the prompt is (0-100)."""
    score = 40
    lower = prompt.lower()
    reasons: list[str] = []

    # Check for numbers / quantities
    if re.search(r"\d+", prompt):
        score += 15
        reasons.append("Contains specific numbers or quantities")

    # Check for constraints
    constraint_hits = sum(1 for w in QUALITY_INDICATORS["constraint_words"] if w in lower)
    if constraint_hits:
        score += constraint_hits * 8
        reasons.append(f"Defines {constraint_hits} constraint(s)")

    # Check for output format cues
    format_hits = sum(1 for w in QUALITY_INDICATORS["output_format_cues"] if w in lower)
    if format_hits:
        score += format_hits * 10
        reasons.append(f"Specifies {format_hits} output format cue(s)")

    # Check for examples
    if "example" in lower or "e.g." in lower:
        score += 15
        reasons.append("Provides examples")

    return {"score": max(0, min(100, score)), "reasons": reasons}


def _score_structure(prompt: str) -> dict[str, Any]:
    """Score how well-structured the prompt is (0-100)."""
    score = 40
    lower = prompt.lower()
    reasons: list[str] = []

    # Newlines / sections
    line_count = len([l for l in prompt.splitlines() if l.strip()])
    if line_count > 3:
        score += 15
        reasons.append("Uses multi-line formatting")

    # Numbered or bulleted lists
    struct_hits = sum(1 for m in QUALITY_INDICATORS["structure_markers"] if m in lower)
    if struct_hits:
        score += min(struct_hits * 5, 20)
        reasons.append(f"Contains {struct_hits} structural marker(s)")

    # Role definition
    role_hits = sum(1 for r in QUALITY_INDICATORS["role_indicators"] if r in lower)
    if role_hits:
        score += 15
        reasons.append("Defines a role for the model")

    # Context section
    ctx_hits = sum(1 for c in QUALITY_INDICATORS["context_signals"] if c in lower)
    if ctx_hits:
        score += 10
        reasons.append("Provides contextual information")

    return {"score": max(0, min(100, score)), "reasons": reasons}


def _score_completeness(prompt: str) -> dict[str, Any]:
    """Score whether the prompt has all recommended elements (0-100)."""
    score = 30
    lower = prompt.lower()
    reasons: list[str] = []
    present: list[str] = []
    missing: list[str] = []

    checks = {
        "task_description": len(prompt.split()) >= 10,
        "context_or_background": any(c in lower for c in QUALITY_INDICATORS["context_signals"]),
        "output_format": any(f in lower for f in QUALITY_INDICATORS["output_format_cues"]),
        "constraints": any(c in lower for c in QUALITY_INDICATORS["constraint_words"]),
        "examples": "example" in lower or "e.g." in lower,
        "role_definition": any(r in lower for r in QUALITY_INDICATORS["role_indicators"]),
    }

    for name, found in checks.items():
        label = name.replace("_", " ").title()
        if found:
            score += 12
            present.append(label)
        else:
            missing.append(label)

    if present:
        reasons.append(f"Present: {', '.join(present)}")
    if missing:
        reasons.append(f"Missing: {', '.join(missing)}")

    return {"score": max(0, min(100, score)), "reasons": reasons, "present": present, "missing": missing}


def _detect_task_type(prompt: str) -> str:
    """Best-effort guess at the task type from prompt content."""
    lower = prompt.lower()
    signals: dict[str, int] = {t: 0 for t in TASK_TYPES}

    code_words = ["code", "function", "class", "debug", "program", "script", "api", "bug"]
    for w in code_words:
        if w in lower:
            signals["code_generation"] += 1

    creative_words = ["story", "poem", "fiction", "creative", "write a", "narrative"]
    for w in creative_words:
        if w in lower:
            signals["creative_writing"] += 1

    data_words = ["data", "analysis", "statistic", "chart", "graph", "dataset", "metric"]
    for w in data_words:
        if w in lower:
            signals["data_analysis"] += 1

    summary_words = ["summarize", "summary", "condense", "tldr", "brief"]
    for w in summary_words:
        if w in lower:
            signals["summarization"] += 1

    qa_words = ["what is", "who is", "when did", "how does", "why does", "explain"]
    for w in qa_words:
        if w in lower:
            signals["question_answering"] += 1

    translate_words = ["translate", "translation", "convert to", "in spanish", "in french"]
    for w in translate_words:
        if w in lower:
            signals["translation"] += 1

    best = max(signals, key=lambda k: signals[k])
    return best if signals[best] > 0 else "general"


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def analyze_prompt_quality(prompt: str) -> dict[str, Any]:
    """Analyze a prompt and return quality scores for clarity, specificity,
    structure, and completeness.  Each dimension is scored 0-100.
    """
    clarity = _score_clarity(prompt)
    specificity = _score_specificity(prompt)
    structure = _score_structure(prompt)
    completeness = _score_completeness(prompt)

    overall = round(
        (clarity["score"] + specificity["score"] + structure["score"] + completeness["score"]) / 4
    )
    task_type = _detect_task_type(prompt)

    return {
        "overall_score": overall,
        "detected_task_type": task_type,
        "dimensions": {
            "clarity": clarity,
            "specificity": specificity,
            "structure": structure,
            "completeness": completeness,
        },
        "word_count": len(prompt.split()),
        "line_count": len([l for l in prompt.splitlines() if l.strip()]),
    }


@mcp.tool()
async def optimize_prompt(prompt: str, task_type: str = "") -> dict[str, Any]:
    """Analyze the given prompt and return concrete suggestions to improve it,
    along with a rewritten optimized version.
    """
    if not task_type:
        task_type = _detect_task_type(prompt)

    analysis = await analyze_prompt_quality(prompt)
    suggestions: list[str] = []
    optimized_parts: list[str] = []

    # --- Role ---
    completeness = analysis["dimensions"]["completeness"]
    if "Role Definition" in completeness.get("missing", []):
        task_info = TASK_TYPES.get(task_type, TASK_TYPES["general"])
        role = task_info["system_prompt"].split(".")[0] + "."
        suggestions.append(f"Add a role definition, e.g.: \"{role}\"")
        optimized_parts.append(f"[Role]: {role}")

    # --- Context ---
    if "Context Or Background" in completeness.get("missing", []):
        suggestions.append("Add context or background information about your task")
        optimized_parts.append("[Context]: (Add relevant background here)")

    # --- Main task ---
    optimized_parts.append(f"[Task]: {prompt}")

    # --- Output format ---
    if "Output Format" in completeness.get("missing", []):
        suggestions.append("Specify the desired output format (e.g., JSON, markdown, bullet list)")
        optimized_parts.append("[Format]: (Specify desired output format)")

    # --- Constraints ---
    if "Constraints" in completeness.get("missing", []):
        suggestions.append("Add constraints such as length limits, tone, or style requirements")
        optimized_parts.append("[Constraints]: (Add any constraints)")

    # --- Examples ---
    if "Examples" in completeness.get("missing", []):
        suggestions.append("Include an example of the expected output")
        optimized_parts.append("[Example]: (Provide an example if possible)")

    # Clarity-specific tips
    if analysis["dimensions"]["clarity"]["score"] < 60:
        suggestions.append("Make the request more explicit — avoid vague phrases like 'help me' or 'tell me about'")

    # Specificity tips
    if analysis["dimensions"]["specificity"]["score"] < 50:
        suggestions.append("Add specific numbers, names, or quantities to ground the request")

    return {
        "original_prompt": prompt,
        "detected_task_type": task_type,
        "analysis": analysis,
        "suggestions": suggestions,
        "optimized_prompt": "\n".join(optimized_parts),
    }


@mcp.tool()
async def suggest_system_prompt(task_type: str) -> dict[str, Any]:
    """Return a recommended system prompt for the given task type.

    Supported task types: creative_writing, code_generation, data_analysis,
    summarization, question_answering, translation, general.
    """
    if task_type not in TASK_TYPES:
        available = list(TASK_TYPES.keys())
        return {
            "error": f"Unknown task type '{task_type}'",
            "available_types": available,
        }

    info = TASK_TYPES[task_type]
    return {
        "task_type": task_type,
        "description": info["description"],
        "system_prompt": info["system_prompt"],
        "recommended_temperature": info["temperature"],
        "recommended_max_tokens": info["max_tokens"],
    }


@mcp.tool()
async def suggest_model_parameters(task_type: str) -> dict[str, Any]:
    """Recommend temperature and max_tokens settings for a given task type,
    along with per-provider model suggestions.
    """
    if task_type not in TASK_TYPES:
        task_type = "general"

    info = TASK_TYPES[task_type]

    model_suggestions: dict[str, dict[str, Any]] = {
        "anthropic": {
            "model": "claude-sonnet-4-20250514",
            "strengths": "Strong reasoning, instruction following, safety",
        },
        "openai": {
            "model": "gpt-4o",
            "strengths": "Broad knowledge, tool use, multimodal",
        },
        "google": {
            "model": "gemini-1.5-flash",
            "strengths": "Speed, long context window, cost-effective",
        },
        "perplexity": {
            "model": "sonar",
            "strengths": "Real-time web search, up-to-date information",
        },
    }

    return {
        "task_type": task_type,
        "recommended_temperature": info["temperature"],
        "recommended_max_tokens": info["max_tokens"],
        "models": model_suggestions,
        "tips": [
            "Lower temperature (0.1-0.3) for factual/code tasks",
            "Higher temperature (0.7-1.0) for creative tasks",
            "Increase max_tokens for long-form outputs",
            "Use Perplexity for queries requiring current information",
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
