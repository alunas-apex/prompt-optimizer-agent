"""Telegram bot command handlers for the Prompt Optimizer Agent.

Provides /start, /help, /analyze, /optimize commands, plus a default
handler that treats plain text messages as prompts to optimize.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import prompt-analyzer helpers directly (bypasses MCP decorator)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_analyzer():
    """Import the prompt-analyzer module from its file path."""
    mod_path = _PROJECT_ROOT / "mcp-servers" / "prompt-analyzer" / "server.py"
    spec = importlib.util.spec_from_file_location("prompt_analyzer_server", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load prompt-analyzer from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prompt_analyzer_server"] = mod
    spec.loader.exec_module(mod)
    return mod


_analyzer = _load_analyzer()


# ---------------------------------------------------------------------------
# Analysis / optimisation helpers (call internal scoring functions directly)
# ---------------------------------------------------------------------------

def _run_analysis(prompt: str) -> dict[str, Any]:
    """Analyze a prompt using the prompt-analyzer's scoring helpers."""
    clarity = _analyzer._score_clarity(prompt)
    specificity = _analyzer._score_specificity(prompt)
    structure = _analyzer._score_structure(prompt)
    completeness = _analyzer._score_completeness(prompt)

    overall = round(
        (clarity["score"] + specificity["score"]
         + structure["score"] + completeness["score"]) / 4
    )
    task_type = _analyzer._detect_task_type(prompt)

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
    }


def _run_optimization(prompt: str) -> dict[str, Any]:
    """Optimize a prompt and return before/after analysis."""
    task_type = _analyzer._detect_task_type(prompt)
    analysis = _run_analysis(prompt)

    suggestions: list[str] = []
    optimized_parts: list[str] = []

    completeness = analysis["dimensions"]["completeness"]
    task_info = _analyzer.TASK_TYPES.get(task_type, _analyzer.TASK_TYPES["general"])

    # Role
    if "Role Definition" in completeness.get("missing", []):
        role = task_info["system_prompt"].split(".")[0] + "."
        suggestions.append(f'Add a role definition, e.g.: "{role}"')
        optimized_parts.append(f"[Role]: {role}")

    # Context
    if "Context Or Background" in completeness.get("missing", []):
        suggestions.append("Add context or background information")
        optimized_parts.append("[Context]: (Add relevant background here)")

    # Main task
    optimized_parts.append(f"[Task]: {prompt}")

    # Output format
    if "Output Format" in completeness.get("missing", []):
        suggestions.append("Specify the desired output format")
        optimized_parts.append("[Format]: (Specify desired output format)")

    # Constraints
    if "Constraints" in completeness.get("missing", []):
        suggestions.append("Add constraints (length, tone, style)")
        optimized_parts.append("[Constraints]: (Add any constraints)")

    # Examples
    if "Examples" in completeness.get("missing", []):
        suggestions.append("Include an example of expected output")
        optimized_parts.append("[Example]: (Provide an example if possible)")

    # Clarity tips
    if analysis["dimensions"]["clarity"]["score"] < 60:
        suggestions.append("Avoid vague phrases like 'help me' or 'tell me about'")

    # Specificity tips
    if analysis["dimensions"]["specificity"]["score"] < 50:
        suggestions.append("Add specific numbers, names, or quantities")

    optimized_prompt = "\n".join(optimized_parts)
    optimized_analysis = _run_analysis(optimized_prompt)

    return {
        "original_prompt": prompt,
        "optimized_prompt": optimized_prompt,
        "detected_task_type": task_type,
        "before": analysis,
        "after": optimized_analysis,
        "suggestions": suggestions,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_analysis(analysis: dict[str, Any], prompt: str) -> str:
    """Format an analysis result as a Telegram-friendly message."""
    dims = analysis["dimensions"]
    overall = analysis["overall_score"]

    quality = "Poor"
    if overall >= 80:
        quality = "Excellent"
    elif overall >= 65:
        quality = "Good"
    elif overall >= 50:
        quality = "Fair"

    lines = [
        f"*Prompt Analysis* — {quality} ({overall}/100)\n",
        f"Prompt: _{_escape_md(prompt)}_\n",
        f"Task type: `{analysis['detected_task_type']}`\n",
        "*Dimension Scores:*",
        f"  Clarity: {dims['clarity']['score']}/100",
        f"  Specificity: {dims['specificity']['score']}/100",
        f"  Structure: {dims['structure']['score']}/100",
        f"  Completeness: {dims['completeness']['score']}/100",
    ]

    missing = dims["completeness"].get("missing", [])
    if missing:
        lines.append(f"\n*Missing elements:* {', '.join(missing)}")

    return "\n".join(lines)


def _format_optimization(result: dict[str, Any]) -> str:
    """Format an optimization result as a Telegram-friendly message."""
    before_score = result["before"]["overall_score"]
    after_score = result["after"]["overall_score"]
    diff = after_score - before_score

    lines = [
        "*Prompt Optimization Results*\n",
        f"Task type: `{result['detected_task_type']}`\n",
        f"*Original prompt:*\n_{_escape_md(result['original_prompt'])}_\n",
        f"*Optimized prompt:*\n`{result['optimized_prompt']}`\n",
        f"*Score: {before_score} → {after_score} \\(\\+{diff}\\)*\n",
        "*Breakdown:*",
    ]

    for dim_name in ("clarity", "specificity", "structure", "completeness"):
        b = result["before"]["dimensions"][dim_name]["score"]
        a = result["after"]["dimensions"][dim_name]["score"]
        change = a - b
        sign = "\\+" if change > 0 else ""
        label = dim_name.capitalize()
        lines.append(f"  {label}: {b} → {a} \\({sign}{change}\\)")

    if result["suggestions"]:
        lines.append("\n*Suggestions:*")
        for i, s in enumerate(result["suggestions"], 1):
            lines.append(f"  {i}\\. {_escape_md(s)}")

    return "\n".join(lines)


def _escape_md(text: str) -> str:
    """Escape special MarkdownV2 characters."""
    special = r"_*[]()~`>#+-=|{}.!"
    escaped = []
    for ch in text:
        if ch in special:
            escaped.append(f"\\{ch}")
        else:
            escaped.append(ch)
    return "".join(escaped)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — send a welcome message."""
    text = (
        "*Welcome to the Prompt Optimizer Bot\\!*\n\n"
        "I help you write better prompts for AI models\\.\n\n"
        "Send me any prompt and I'll analyze and optimize it, "
        "or use these commands:\n\n"
        "/analyze \\<prompt\\> — Analyze prompt quality\n"
        "/optimize \\<prompt\\> — Full optimization with suggestions\n"
        "/help — Show usage instructions"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help — show usage instructions."""
    text = (
        "*Prompt Optimizer Bot — Help*\n\n"
        "*Commands:*\n"
        "/start — Welcome message\n"
        "/help — This help text\n"
        "/analyze \\<prompt\\> — Score a prompt on clarity, specificity, structure, and completeness\n"
        "/optimize \\<prompt\\> — Get an optimized version with before/after scores\n\n"
        "*Plain text:* Send any message and I'll treat it as a prompt to optimize\\.\n\n"
        "*Scoring dimensions:*\n"
        "• *Clarity* — How clear and unambiguous is the prompt?\n"
        "• *Specificity* — Does it include concrete details?\n"
        "• *Structure* — Is it well\\-organized?\n"
        "• *Completeness* — Does it have role, context, format, constraints, examples?"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /analyze <prompt> — analyze prompt quality."""
    text = update.message.text or ""
    prompt = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else ""
    if not prompt:
        await update.message.reply_text(
            "Please provide a prompt to analyze\\.\n\nUsage: `/analyze write me a story about AI`",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    analysis = _run_analysis(prompt)
    text = _format_analysis(analysis, prompt)
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def optimize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /optimize <prompt> — run full optimization."""
    text = update.message.text or ""
    prompt = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else ""
    if not prompt:
        await update.message.reply_text(
            "Please provide a prompt to optimize\\.\n\nUsage: `/optimize write me a story about AI`",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    result = _run_optimization(prompt)
    text = _format_optimization(result)
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def plain_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain text messages — treat as prompts to optimize."""
    prompt = update.message.text
    if not prompt or not prompt.strip():
        return

    result = _run_optimization(prompt.strip())
    text = _format_optimization(result)
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)
