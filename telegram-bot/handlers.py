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
# Formatting helpers — conversational, emoji-rich, mobile-friendly
# ---------------------------------------------------------------------------

_TASK_EMOJI = {
    "creative_writing": "✍️",
    "code_generation": "💻",
    "data_analysis": "📊",
    "summarization": "📝",
    "question_answering": "🔎",
    "translation": "🌐",
    "general": "💬",
}

_TASK_LABEL = {
    "creative_writing": "Creative Writing",
    "code_generation": "Code Generation",
    "data_analysis": "Data Analysis",
    "summarization": "Summarization",
    "question_answering": "Q&A / Research",
    "translation": "Translation",
    "general": "General",
}

# temp values are pre-escaped for MarkdownV2 (. → \.)
_MODEL_ADVICE = {
    "creative_writing": {
        "model": "Claude Sonnet",
        "temp": "0\\.9",
        "why": "Creative tasks thrive with higher temperature — more vivid and original results",
    },
    "code_generation": {
        "model": "Claude Sonnet",
        "temp": "0\\.2",
        "why": "Code needs to be correct, not creative — low temperature keeps it precise and reliable",
    },
    "data_analysis": {
        "model": "Claude Sonnet",
        "temp": "0\\.3",
        "why": "Analysis needs consistency — low temperature reduces hallucination",
    },
    "summarization": {
        "model": "Claude Sonnet",
        "temp": "0\\.3",
        "why": "Summaries should be faithful to the source — keep temperature low",
    },
    "question_answering": {
        "model": "Claude Sonnet",
        "temp": "0\\.4",
        "why": "A slight creative touch helps explain things clearly and engagingly",
    },
    "translation": {
        "model": "Claude Sonnet",
        "temp": "0\\.3",
        "why": "Translation prioritises accuracy — low temperature stays close to the original",
    },
    "general": {
        "model": "Claude Sonnet",
        "temp": "0\\.7",
        "why": "A balanced temperature works well for general tasks",
    },
}


def _score_emoji(score: int) -> str:
    if score >= 75:
        return "✅"
    if score >= 50:
        return "⚠️"
    return "❌"


def _overall_verdict(score: int) -> str:
    if score >= 80:
        return "🌟 *Excellent* — this prompt is sharp and ready to go\\!"
    if score >= 65:
        return "✅ *Good* — a couple of tweaks will make it great"
    if score >= 50:
        return "⚠️ *Fair* — it'll work, but the AI might miss what you mean"
    if score >= 30:
        return "❌ *Needs work* — the AI will probably give you a vague answer"
    return "🚨 *Very weak* — let's fix this together"


def _dim_verdict(score: int, dim: str) -> str:
    thresholds: dict[str, dict[int, str]] = {
        "clarity": {
            75: "crystal clear",
            50: "a bit vague",
            25: "too ambiguous",
            0:  "very unclear",
        },
        "specificity": {
            75: "nicely specific",
            50: "could use more detail",
            25: "too generic",
            0:  "very generic",
        },
        "structure": {
            75: "well organised",
            50: "could be tidier",
            25: "a bit scattered",
            0:  "hard to follow",
        },
        "completeness": {
            75: "covers the bases",
            50: "missing a few things",
            25: "missing a lot",
            0:  "very incomplete",
        },
    }
    levels = thresholds.get(dim, {75: "good", 50: "okay", 25: "weak", 0: "poor"})
    for threshold in sorted(levels.keys(), reverse=True):
        if score >= threshold:
            return levels[threshold]
    return "poor"


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


def _format_analysis(analysis: dict[str, Any], prompt: str) -> str:
    """Format an analysis result — conversational, mobile-friendly."""
    dims = analysis["dimensions"]
    overall = analysis["overall_score"]
    task_type = analysis["detected_task_type"]
    task_emoji = _TASK_EMOJI.get(task_type, "💬")
    task_label = _TASK_LABEL.get(task_type, task_type.replace("_", " ").title())

    lines = [
        "🔍 *Prompt Check*\n",
        f"_{_escape_md(prompt)}_\n",
        _overall_verdict(overall) + "\n",
        f"Detected as: {task_emoji} {_escape_md(task_label)}\n",
        "*What I found:*",
    ]

    for dim_name, label in [
        ("clarity",      "Clarity"),
        ("specificity",  "Specificity"),
        ("structure",    "Structure"),
        ("completeness", "Completeness"),
    ]:
        score = dims[dim_name]["score"]
        emoji = _score_emoji(score)
        verdict = _dim_verdict(score, dim_name)
        lines.append(f"{emoji} *{label}* — {_escape_md(verdict)}")

    missing = dims["completeness"].get("missing", [])
    if missing:
        lines.append("\n💡 *Quick wins:*")
        for item in missing:
            lines.append(f"• Add {_escape_md(item.lower())}")

    lines.append("\n_Send your prompt without a command and I'll rewrite it for you\\!_")

    return "\n".join(lines)


def _format_optimization(result: dict[str, Any]) -> str:
    """Format an optimization result — conversational, mobile-friendly."""
    task_type = result["detected_task_type"]
    task_emoji = _TASK_EMOJI.get(task_type, "💬")
    task_label = _TASK_LABEL.get(task_type, task_type.replace("_", " ").title())

    before_score = result["before"]["overall_score"]
    after_score = result["after"]["overall_score"]
    diff = after_score - before_score
    diff_str = f"\\+{diff}" if diff >= 0 else f"\\-{abs(diff)}"

    advice = _MODEL_ADVICE.get(task_type, _MODEL_ADVICE["general"])

    lines = [
        f"✨ *Your upgraded prompt*\n",
        f"Detected: {task_emoji} {_escape_md(task_label)}\n",
        "📋 *Copy this:*",
        f"```\n{result['optimized_prompt']}\n```\n",
        f"📈 *Score: {before_score} → {after_score} \\({diff_str}\\)*",
    ]

    for dim_name, label in [
        ("clarity",      "Clarity"),
        ("specificity",  "Specificity"),
        ("structure",    "Structure"),
        ("completeness", "Completeness"),
    ]:
        b = result["before"]["dimensions"][dim_name]["score"]
        a = result["after"]["dimensions"][dim_name]["score"]
        change = a - b
        change_str = f"\\+{change}" if change >= 0 else f"\\-{abs(change)}"
        emoji = _score_emoji(a)
        lines.append(f"  {emoji} {label}: {b} → {a} \\({change_str}\\)")

    lines.append(f"\n🤖 *Best model for this:*")
    lines.append(
        f"*{_escape_md(advice['model'])}* at temperature {advice['temp']}\n"
        f"_{_escape_md(advice['why'])}_"
    )

    if result["suggestions"]:
        lines.append("\n💡 *Tips to make it even better:*")
        for i, s in enumerate(result["suggestions"], 1):
            lines.append(f"{i}\\. {_escape_md(s)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — send a welcome message."""
    text = (
        "👋 *Hey\\! I'm your Prompt Optimizer\\.*\n\n"
        "Send me any AI prompt and I'll tell you what's weak about it "
        "and give you a rewritten version that actually works\\.\n\n"
        "*How to use me:*\n"
        "💬 Just send any prompt as a message — I'll rewrite it\n"
        "🔍 /analyze \\<prompt\\> — See what's wrong with your prompt\n"
        "✨ /optimize \\<prompt\\> — Get the full upgrade \\+ model advice\n"
        "❓ /help — Tips and examples\n\n"
        "_Try it now — just paste a prompt you've been using\\!_"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help — show usage instructions."""
    text = (
        "❓ *How to get the best out of me*\n\n"
        "*The fastest way:* just send me a prompt as a plain message\\.\n"
        "I'll rewrite it and tell you which AI model to use\\.\n\n"
        "*Commands:*\n"
        "🔍 /analyze — Quick breakdown of what's strong and weak\n"
        "✨ /optimize — Full rewrite \\+ score comparison \\+ model recommendation\n\n"
        "*What I check:*\n"
        "✅ *Clarity* — Is it specific enough for the AI to understand\\?\n"
        "🎯 *Specificity* — Does it have concrete details, numbers, names\\?\n"
        "🏗 *Structure* — Is it organised with a clear ask\\?\n"
        "📋 *Completeness* — Does it have a role, context, format, constraints\\?\n\n"
        "*Example prompts to try:*\n"
        "`write me a story about AI`\n"
        "`summarize this article for me`\n"
        "`help me write a Python function`"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /analyze <prompt> — analyze prompt quality."""
    text = update.message.text or ""
    prompt = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else ""
    if not prompt:
        await update.message.reply_text(
            "🔍 Give me something to check\\!\n\n"
            "Usage: `/analyze write me a story about AI`",
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
            "✨ Give me a prompt to upgrade\\!\n\n"
            "Usage: `/optimize write me a story about AI`",
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
