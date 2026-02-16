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

mcp = FastMCP("prompt-analyzer", instructions="Analyzes and optimizes prompt quality")


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
# Platform catalog — 15 platforms, current as of February 2026
# ---------------------------------------------------------------------------

PLATFORM_CATALOG: dict[str, dict] = {
    # ── Flagship ────────────────────────────────────────────────────────────
    "claude_sonnet_api": {
        "label": "Claude Sonnet 4.5 (API)",
        "model_id": "claude-sonnet-4-5-20250929",
        "tier": "flagship",
        "input_mtok": 3.00,
        "output_mtok": 15.00,
        "context_k": 200,
        "speed": "medium",
        "cost_note": "~$0.02/request",
        "api": True,
        "ui_free": False,
        "extended_thinking": False,
        "web_search": False,
        "vision": True,
        "task_scores": {
            "creative_writing": 93, "code_generation": 95, "data_analysis": 84,
            "summarization": 85, "question_answering": 78, "translation": 80,
            "general": 90,
        },
        "default_settings": {"temperature": 0.7, "max_tokens": 4096},
        "task_settings": {
            "creative_writing": {"temperature": 0.9, "max_tokens": 4096},
            "code_generation":  {"temperature": 0.2, "max_tokens": 8192},
            "data_analysis":    {"temperature": 0.3, "max_tokens": 4096},
            "summarization":    {"temperature": 0.3, "max_tokens": 2048},
            "question_answering": {"temperature": 0.5, "max_tokens": 2048},
            "translation":      {"temperature": 0.3, "max_tokens": 4096},
        },
        "strengths": ["instruction following", "nuanced writing", "coding", "safety"],
        "best_when": "Reliable high-quality output for most tasks",
        "when_to_skip": "Need real-time web data or have a very tight budget",
    },
    "claude_opus_api": {
        "label": "Claude Opus 4.6 (API)",
        "model_id": "claude-opus-4-6",
        "tier": "flagship",
        "input_mtok": 5.00,
        "output_mtok": 25.00,
        "context_k": 1000,
        "speed": "slow",
        "cost_note": "~$0.03/request",
        "api": True,
        "ui_free": False,
        "extended_thinking": True,
        "web_search": False,
        "vision": True,
        "task_scores": {
            "creative_writing": 88, "code_generation": 90, "data_analysis": 95,
            "summarization": 88, "question_answering": 85, "translation": 83,
            "general": 93,
        },
        "default_settings": {"temperature": 0.7, "max_tokens": 4096},
        "task_settings": {
            "creative_writing":   {"temperature": 0.9, "max_tokens": 4096},
            "code_generation":    {"temperature": 0.2, "max_tokens": 8192},
            "data_analysis":      {"temperature": 0.2, "max_tokens": 8192, "extended_thinking": True},
            "question_answering": {"temperature": 0.4, "max_tokens": 4096, "extended_thinking": True},
        },
        "strengths": ["deep reasoning", "1M context", "extended thinking", "complex analysis"],
        "best_when": "Maximum quality needed, cost is secondary, or analysing very long documents",
        "when_to_skip": "Simple tasks — Sonnet gives 95% of the quality at 40% of the cost",
    },
    "claude_sonnet_ui": {
        "label": "Claude Sonnet 4.5 (claude.ai)",
        "model_id": None,
        "tier": "flagship",
        "input_mtok": None,
        "output_mtok": None,
        "context_k": 200,
        "speed": "medium",
        "cost_note": "Free / Pro $20/mo",
        "api": False,
        "ui_free": True,
        "extended_thinking": True,
        "web_search": True,
        "vision": True,
        "task_scores": {
            "creative_writing": 91, "code_generation": 88, "data_analysis": 83,
            "summarization": 84, "question_answering": 82, "translation": 79,
            "general": 89,
        },
        "default_settings": {},
        "task_settings": {},
        "strengths": ["artifacts", "file upload", "web search", "computer use", "free tier"],
        "best_when": "Interactive use, quick experiments, or you want file/web/artifact features without code",
        "when_to_skip": "Automation or production pipelines — use the API instead",
    },
    "gpt52": {
        "label": "GPT-5.2 (OpenAI)",
        "model_id": "gpt-5.2",
        "tier": "flagship",
        "input_mtok": 1.75,
        "output_mtok": 14.00,
        "context_k": 256,
        "speed": "medium",
        "cost_note": "~$0.02/request",
        "api": True,
        "ui_free": True,
        "extended_thinking": True,
        "web_search": True,
        "vision": True,
        "task_scores": {
            "creative_writing": 87, "code_generation": 85, "data_analysis": 83,
            "summarization": 80, "question_answering": 84, "translation": 92,
            "general": 88,
        },
        "default_settings": {"temperature": 0.7, "max_tokens": 4096},
        "task_settings": {
            "creative_writing": {"temperature": 1.0, "max_tokens": 4096},
            "code_generation":  {"temperature": 0.2, "max_tokens": 8192},
            "data_analysis":    {"temperature": 0.3, "max_tokens": 4096},
            "translation":      {"temperature": 0.3, "max_tokens": 4096},
        },
        "strengths": ["multilingual", "image generation", "code interpreter", "broad knowledge"],
        "best_when": "Translation, multilingual tasks, or when you need DALL-E image generation alongside text",
        "when_to_skip": "Complex instruction-following — Claude edges it out; deep reasoning — use o3",
    },
    "o3": {
        "label": "o3 (OpenAI reasoning)",
        "model_id": "o3",
        "tier": "flagship",
        "input_mtok": 2.00,
        "output_mtok": 8.00,
        "context_k": 200,
        "speed": "slow",
        "cost_note": "~$0.01–$0.05/request (reasoning tokens multiply cost)",
        "api": True,
        "ui_free": False,
        "extended_thinking": True,
        "web_search": False,
        "vision": True,
        "task_scores": {
            "creative_writing": 45, "code_generation": 88, "data_analysis": 96,
            "summarization": 60, "question_answering": 80, "translation": 65,
            "general": 78,
        },
        "default_settings": {"temperature": 0.1, "max_tokens": 4096},
        "task_settings": {
            "code_generation": {"temperature": 0.1, "max_tokens": 8192},
            "data_analysis":   {"temperature": 0.1, "max_tokens": 8192},
        },
        "strengths": ["logic", "maths", "multi-step reasoning", "algorithm design"],
        "best_when": "Maths, algorithms, rigorous logical analysis, or structured data problems",
        "when_to_skip": "Creative writing, translation, or simple tasks — reasoning overhead wastes money",
    },
    "gemini25_pro": {
        "label": "Gemini 2.5 Pro (Google)",
        "model_id": "gemini-2.5-pro",
        "tier": "flagship",
        "input_mtok": 1.25,
        "output_mtok": 10.00,
        "context_k": 1000,
        "speed": "medium",
        "cost_note": "~$0.01/request",
        "api": True,
        "ui_free": False,
        "extended_thinking": True,
        "web_search": True,
        "vision": True,
        "task_scores": {
            "creative_writing": 76, "code_generation": 81, "data_analysis": 91,
            "summarization": 90, "question_answering": 88, "translation": 87,
            "general": 85,
        },
        "default_settings": {"temperature": 0.7, "max_tokens": 4096},
        "task_settings": {
            "data_analysis":    {"temperature": 0.2, "max_tokens": 8192},
            "summarization":    {"temperature": 0.3, "max_tokens": 4096},
            "question_answering": {"temperature": 0.4, "max_tokens": 4096},
            "translation":      {"temperature": 0.3, "max_tokens": 4096},
        },
        "strengths": ["1M token context", "Google Search grounding", "multimodal", "long documents"],
        "best_when": "Summarising or analysing huge documents, or when grounded Google Search answers matter",
        "when_to_skip": "Creative writing or coding — Claude Sonnet wins those",
    },
    "grok4": {
        "label": "Grok 4 (xAI)",
        "model_id": "grok-4",
        "tier": "flagship",
        "input_mtok": 3.00,
        "output_mtok": 15.00,
        "context_k": 256,
        "speed": "medium",
        "cost_note": "~$0.02/request",
        "api": True,
        "ui_free": False,
        "extended_thinking": True,
        "web_search": True,
        "vision": True,
        "task_scores": {
            "creative_writing": 72, "code_generation": 75, "data_analysis": 72,
            "summarization": 70, "question_answering": 88, "translation": 68,
            "general": 78,
        },
        "default_settings": {"temperature": 0.7, "max_tokens": 4096},
        "task_settings": {
            "question_answering": {"temperature": 0.4, "max_tokens": 2048},
        },
        "strengths": ["real-time X/Twitter data", "current events", "web search", "DeepSearch"],
        "best_when": "Questions about current events, social trends, or X/Twitter data",
        "when_to_skip": "Creative, coding, or analysis — better specialist models exist",
    },
    # ── Balanced ─────────────────────────────────────────────────────────────
    "claude_haiku_api": {
        "label": "Claude Haiku 4.5 (API)",
        "model_id": "claude-haiku-4-5-20251001",
        "tier": "balanced",
        "input_mtok": 1.00,
        "output_mtok": 5.00,
        "context_k": 200,
        "speed": "fast",
        "cost_note": "~$0.006/request",
        "api": True,
        "ui_free": False,
        "extended_thinking": False,
        "web_search": False,
        "vision": True,
        "task_scores": {
            "creative_writing": 70, "code_generation": 74, "data_analysis": 62,
            "summarization": 80, "question_answering": 62, "translation": 73,
            "general": 72,
        },
        "default_settings": {"temperature": 0.7, "max_tokens": 2048},
        "task_settings": {
            "creative_writing": {"temperature": 0.9, "max_tokens": 2048},
            "code_generation":  {"temperature": 0.2, "max_tokens": 4096},
            "summarization":    {"temperature": 0.3, "max_tokens": 2048},
            "translation":      {"temperature": 0.3, "max_tokens": 2048},
        },
        "strengths": ["speed", "cost-efficiency", "high throughput"],
        "best_when": "High-volume tasks, quick drafts, or when latency matters more than peak quality",
        "when_to_skip": "Complex reasoning or nuanced writing — spend the extra on Sonnet",
    },
    "gemini25_flash": {
        "label": "Gemini 2.5 Flash (Google)",
        "model_id": "gemini-2.5-flash",
        "tier": "balanced",
        "input_mtok": 0.15,
        "output_mtok": 0.60,
        "context_k": 1000,
        "speed": "fast",
        "cost_note": "~$0.001/request",
        "api": True,
        "ui_free": True,
        "extended_thinking": True,
        "web_search": True,
        "vision": True,
        "task_scores": {
            "creative_writing": 73, "code_generation": 72, "data_analysis": 76,
            "summarization": 88, "question_answering": 78, "translation": 86,
            "general": 80,
        },
        "default_settings": {"temperature": 0.7, "max_tokens": 2048},
        "task_settings": {
            "creative_writing": {"temperature": 1.0, "max_tokens": 2048},
            "code_generation":  {"temperature": 0.2, "max_tokens": 4096},
            "summarization":    {"temperature": 0.3, "max_tokens": 4096},
            "translation":      {"temperature": 0.3, "max_tokens": 4096},
        },
        "strengths": ["1M context at low cost", "speed", "Google Search grounding", "translation"],
        "best_when": "Budget-conscious tasks needing long context, summarisation, or translation at scale",
        "when_to_skip": "When you need peak quality — Flash trades some accuracy for speed/cost",
    },
    "deepseek_v3": {
        "label": "DeepSeek V3 (API)",
        "model_id": "deepseek-chat",
        "tier": "balanced",
        "input_mtok": 0.27,
        "output_mtok": 1.10,
        "context_k": 128,
        "speed": "fast",
        "cost_note": "~$0.001/request",
        "api": True,
        "ui_free": True,
        "extended_thinking": False,
        "web_search": False,
        "vision": False,
        "task_scores": {
            "creative_writing": 70, "code_generation": 88, "data_analysis": 75,
            "summarization": 83, "question_answering": 70, "translation": 90,
            "general": 78,
        },
        "default_settings": {"temperature": 0.7, "max_tokens": 4096},
        "task_settings": {
            "creative_writing": {"temperature": 0.9, "max_tokens": 4096},
            "code_generation":  {"temperature": 0.2, "max_tokens": 8192},
            "summarization":    {"temperature": 0.3, "max_tokens": 4096},
            "translation":      {"temperature": 0.2, "max_tokens": 4096},
        },
        "strengths": ["exceptional value", "strong coder", "best-in-class translation", "open-weight"],
        "best_when": "When you need GPT-4-class quality at ~10x lower cost, especially for code or translation",
        "when_to_skip": "Vision tasks (no image support) or real-time info (no web search)",
    },
    "deepseek_r1": {
        "label": "DeepSeek R1 (reasoning)",
        "model_id": "deepseek-reasoner",
        "tier": "balanced",
        "input_mtok": 0.55,
        "output_mtok": 2.19,
        "context_k": 128,
        "speed": "slow",
        "cost_note": "~$0.003/request",
        "api": True,
        "ui_free": True,
        "extended_thinking": True,
        "web_search": False,
        "vision": False,
        "task_scores": {
            "creative_writing": 48, "code_generation": 84, "data_analysis": 86,
            "summarization": 65, "question_answering": 72, "translation": 72,
            "general": 70,
        },
        "default_settings": {"temperature": 0.1, "max_tokens": 4096},
        "task_settings": {
            "code_generation": {"temperature": 0.1, "max_tokens": 8192},
            "data_analysis":   {"temperature": 0.1, "max_tokens": 8192},
        },
        "strengths": ["o1-level reasoning", "maths", "algorithms", "budget reasoning model"],
        "best_when": "o3-class reasoning tasks at ~75% lower cost — best value for logic and maths",
        "when_to_skip": "Creative, translation, or tasks that don't benefit from chain-of-thought",
    },
    "mistral_large": {
        "label": "Mistral Large 3 (API)",
        "model_id": "mistral-large-latest",
        "tier": "balanced",
        "input_mtok": 0.50,
        "output_mtok": 1.50,
        "context_k": 128,
        "speed": "fast",
        "cost_note": "~$0.002/request",
        "api": True,
        "ui_free": True,
        "extended_thinking": False,
        "web_search": False,
        "vision": False,
        "task_scores": {
            "creative_writing": 66, "code_generation": 80, "data_analysis": 68,
            "summarization": 73, "question_answering": 64, "translation": 79,
            "general": 72,
        },
        "default_settings": {"temperature": 0.7, "max_tokens": 4096},
        "task_settings": {
            "creative_writing": {"temperature": 0.9, "max_tokens": 4096},
            "code_generation":  {"temperature": 0.2, "max_tokens": 8192},
            "translation":      {"temperature": 0.3, "max_tokens": 4096},
        },
        "strengths": ["open-weight", "EU-hosted option", "strong EU languages", "cost-effective coding"],
        "best_when": "EU data residency requirements, European language tasks, or open-weight self-hosting",
        "when_to_skip": "Vision tasks or when you need web search — Mistral has neither",
    },
    # ── Specialized ──────────────────────────────────────────────────────────
    "perplexity_sonar_pro": {
        "label": "Perplexity Sonar Pro",
        "model_id": "sonar-pro",
        "tier": "specialized",
        "input_mtok": 3.00,
        "output_mtok": 15.00,
        "context_k": 200,
        "speed": "medium",
        "cost_note": "~$0.02/request + web search",
        "api": True,
        "ui_free": True,
        "extended_thinking": False,
        "web_search": True,
        "vision": False,
        "task_scores": {
            "creative_writing": 48, "code_generation": 38, "data_analysis": 68,
            "summarization": 62, "question_answering": 96, "translation": 55,
            "general": 65,
        },
        "default_settings": {"temperature": 0.3, "max_tokens": 2048},
        "task_settings": {
            "question_answering": {"temperature": 0.3, "max_tokens": 2048},
            "data_analysis":      {"temperature": 0.3, "max_tokens": 4096},
        },
        "strengths": ["real-time web search", "citations", "current events", "research reports"],
        "best_when": "Any question needing up-to-date information with cited sources",
        "when_to_skip": "Creative writing, coding, or anything that doesn't benefit from live web data",
    },
    "groq_llama": {
        "label": "Groq (Llama 3.3 70B)",
        "model_id": "llama-3.3-70b-versatile",
        "tier": "specialized",
        "input_mtok": 0.59,
        "output_mtok": 0.79,
        "context_k": 128,
        "speed": "very_fast",
        "cost_note": "~$0.001/request",
        "api": True,
        "ui_free": True,
        "extended_thinking": False,
        "web_search": False,
        "vision": False,
        "task_scores": {
            "creative_writing": 62, "code_generation": 70, "data_analysis": 58,
            "summarization": 72, "question_answering": 58, "translation": 64,
            "general": 65,
        },
        "default_settings": {"temperature": 0.7, "max_tokens": 2048},
        "task_settings": {
            "creative_writing": {"temperature": 0.9, "max_tokens": 2048},
            "code_generation":  {"temperature": 0.2, "max_tokens": 4096},
            "summarization":    {"temperature": 0.3, "max_tokens": 2048},
        },
        "strengths": ["600+ tokens/second", "ultra-low latency", "free tier", "open-weight"],
        "best_when": "Latency is critical (streaming, real-time apps) and quality can be mid-tier",
        "when_to_skip": "Complex tasks where quality matters — LPU speed doesn't offset the model size gap",
    },
    "github_copilot": {
        "label": "GitHub Copilot Pro+",
        "model_id": None,
        "tier": "specialized",
        "input_mtok": None,
        "output_mtok": None,
        "context_k": 200,
        "speed": "fast",
        "cost_note": "$39/mo flat (unlimited IDE use)",
        "api": False,
        "ui_free": False,
        "extended_thinking": False,
        "web_search": False,
        "vision": True,
        "task_scores": {
            "creative_writing": 28, "code_generation": 93, "data_analysis": 55,
            "summarization": 40, "question_answering": 35, "translation": 22,
            "general": 38,
        },
        "default_settings": {},
        "task_settings": {},
        "strengths": ["repo-aware", "multi-file agent mode", "IDE-native", "MCP integration"],
        "best_when": "Day-to-day coding in VS Code/JetBrains — context-aware across your whole repo",
        "when_to_skip": "Anything outside code — it's an IDE tool, not a general assistant",
    },
}


# ---------------------------------------------------------------------------
# Complete prompt generation — fills in all sections with usable defaults
# ---------------------------------------------------------------------------

_COMPLETE_DEFAULTS: dict[str, dict[str, str]] = {
    "creative_writing": {
        "role":        "You are a talented creative writer who crafts vivid, character-driven fiction with strong narrative arcs and compelling dialogue.",
        "context":     "Context: Standalone creative writing task. No prior work to continue.",
        "format":      "Format: Prose narrative, 500–800 words, third-person perspective.",
        "constraints": "Constraints: Open with a strong hook, build to a clear conflict, end with a satisfying resolution.",
        "example":     "Example style: Vivid sensory detail, show don't tell, varied sentence rhythm.",
    },
    "code_generation": {
        "role":        "You are an expert software engineer who writes clean, efficient, well-tested code with clear inline comments.",
        "context":     "Context: Software development task. Target audience: professional developers.",
        "format":      "Format: Code block with a brief explanation of the approach above and usage example below.",
        "constraints": "Constraints: Follow language best practices, include error handling, explain non-obvious decisions.",
        "example":     "Example: Include a minimal working example that can be run immediately.",
    },
    "data_analysis": {
        "role":        "You are a data analysis expert who delivers accurate statistical insights with clear methodology.",
        "context":     "Context: Data analysis task. Assume the audience has basic statistical literacy.",
        "format":      "Format: Structured report — Key Findings, Supporting Data, Methodology, Recommendations.",
        "constraints": "Constraints: Use precise numbers, state assumptions explicitly, flag any data quality issues.",
        "example":     "Example: Lead with the single most important insight, then support with data.",
    },
    "summarization": {
        "role":        "You are an expert summariser who distils content to its essential points without losing nuance or accuracy.",
        "context":     "Context: Text summarisation task. Preserve the author's intent and tone.",
        "format":      "Format: TL;DR (1–2 sentences) → Key Points (bullet list) → Brief Overview (2–3 sentences).",
        "constraints": "Constraints: Do not introduce information not in the source. Keep to 20% of original length.",
        "example":     "Example: Start with the single most important takeaway.",
    },
    "question_answering": {
        "role":        "You are a knowledgeable research assistant who gives accurate, well-sourced answers with appropriate caveats.",
        "context":     "Context: Factual Q&A task. The user needs a reliable, actionable answer.",
        "format":      "Format: Direct answer first (1–2 sentences), then supporting explanation, then sources or caveats.",
        "constraints": "Constraints: State your confidence level. If uncertain, say so and suggest how to verify.",
        "example":     "Example: Answer → Evidence → Confidence level → Follow-up suggestion.",
    },
    "translation": {
        "role":        "You are a professional translator with deep expertise in cultural nuance, idiom, and register.",
        "context":     "Context: Translation task. Preserve the source text's tone, style, and cultural meaning.",
        "format":      "Format: Translated text only, followed by brief Translator's Notes on significant choices.",
        "constraints": "Constraints: Match formality level of the original. Flag untranslatable terms with alternatives.",
        "example":     "Example: If idiomatic, choose a natural target-language equivalent over a literal translation.",
    },
    "general": {
        "role":        "You are a helpful, accurate, and thoughtful assistant who gives clear, well-structured responses.",
        "context":     "Context: General assistance task.",
        "format":      "Format: Clear response with headers if multi-part; bullet points for lists; plain prose otherwise.",
        "constraints": "Constraints: Be concise but complete. Ask one clarifying question if the request is ambiguous.",
        "example":     "Example: Address the core request first, then add relevant context or caveats.",
    },
}


def _generate_complete_prompt(prompt: str, task_type: str, analysis: dict) -> tuple[str, str]:
    """Return (complete_prompt, template_prompt) for the given prompt and analysis.

    complete_prompt  — fully formed, copy-paste ready, no placeholders.
    template_prompt  — same structure with [BRACKET] placeholders for customisation.
    """
    defaults = _COMPLETE_DEFAULTS.get(task_type, _COMPLETE_DEFAULTS["general"])
    missing = analysis["dimensions"]["completeness"].get("missing", [])

    # Build complete version — every section filled with a real default
    complete_parts: list[str] = []
    template_parts: list[str] = []

    # Role
    if "Role Definition" in missing:
        complete_parts.append(defaults["role"])
        template_parts.append(f"[Role]: {defaults['role'].split('.')[0]}.")

    # Context
    if "Context Or Background" in missing:
        complete_parts.append(defaults["context"])
        template_parts.append("[Context]: (Add relevant background — who you are, what this is for)")

    # Main task — always present
    complete_parts.append(f"Task: {prompt}")
    template_parts.append(f"[Task]: {prompt}")

    # Output format
    if "Output Format" in missing:
        complete_parts.append(defaults["format"])
        template_parts.append("[Format]: (e.g. prose / bullet list / JSON / markdown — include length)")

    # Constraints
    if "Constraints" in missing:
        complete_parts.append(defaults["constraints"])
        template_parts.append("[Constraints]: (e.g. tone, length limit, style requirements)")

    # Examples
    if "Examples" in missing:
        complete_parts.append(defaults["example"])
        template_parts.append("[Example]: (Provide a short example of the output you want)")

    return "\n".join(complete_parts), "\n".join(template_parts)


# ---------------------------------------------------------------------------
# Platform evaluation
# ---------------------------------------------------------------------------

def _evaluate_platforms(task_type: str, top_n: int = 5) -> list[dict]:
    """Return top_n platforms for task_type, sorted by quality score."""
    results = []
    for pid, p in PLATFORM_CATALOG.items():
        score = p["task_scores"].get(task_type, p["task_scores"].get("general", 50))
        settings = {**p["default_settings"], **p["task_settings"].get(task_type, {})}
        results.append({
            "id": pid,
            "label": p["label"],
            "model_id": p.get("model_id"),
            "tier": p["tier"],
            "quality_score": score,
            "speed": p["speed"],
            "cost_note": p["cost_note"],
            "api": p["api"],
            "ui_free": p["ui_free"],
            "extended_thinking": p.get("extended_thinking", False),
            "web_search": p.get("web_search", False),
            "settings": settings,
            "strengths": p["strengths"],
            "best_when": p["best_when"],
            "when_to_skip": p["when_to_skip"],
        })
    results.sort(key=lambda x: x["quality_score"], reverse=True)
    return results[:top_n]


@mcp.tool()
async def evaluate_platforms(prompt: str, task_type: str = "") -> dict[str, Any]:
    """Evaluate and rank all platforms for the given prompt and task type.

    Returns the top 5 platforms with quality scores, recommended settings,
    cost estimates, and plain-English guidance on when to use each.
    """
    if not task_type:
        task_type = _detect_task_type(prompt)
    platforms = _evaluate_platforms(task_type)
    return {
        "task_type": task_type,
        "top_platforms": platforms,
        "total_evaluated": len(PLATFORM_CATALOG),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
