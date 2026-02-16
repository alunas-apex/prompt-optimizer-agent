"""Telegram bot command handlers for the Prompt Optimizer Agent."""

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
# Load prompt-analyzer module directly
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_analyzer():
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
# Analysis / optimization logic
# ---------------------------------------------------------------------------

def _run_analysis(prompt: str) -> dict[str, Any]:
    clarity      = _analyzer._score_clarity(prompt)
    specificity  = _analyzer._score_specificity(prompt)
    structure    = _analyzer._score_structure(prompt)
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
            "clarity": clarity, "specificity": specificity,
            "structure": structure, "completeness": completeness,
        },
        "word_count": len(prompt.split()),
    }


def _run_optimization(prompt: str) -> dict[str, Any]:
    task_type    = _analyzer._detect_task_type(prompt)
    analysis     = _run_analysis(prompt)
    completeness = analysis["dimensions"]["completeness"]
    task_info    = _analyzer.TASK_TYPES.get(task_type, _analyzer.TASK_TYPES["general"])

    suggestions: list[str] = []

    if "Role Definition" in completeness.get("missing", []):
        role = task_info["system_prompt"].split(".")[0] + "."
        suggestions.append(f'Add a role definition, e.g.: "{role}"')
    if "Context Or Background" in completeness.get("missing", []):
        suggestions.append("Add context or background information")
    if "Output Format" in completeness.get("missing", []):
        suggestions.append("Specify the desired output format")
    if "Constraints" in completeness.get("missing", []):
        suggestions.append("Add constraints (length, tone, style)")
    if "Examples" in completeness.get("missing", []):
        suggestions.append("Include an example of expected output")
    if analysis["dimensions"]["clarity"]["score"] < 60:
        suggestions.append("Avoid vague phrases like 'help me' or 'tell me about'")
    if analysis["dimensions"]["specificity"]["score"] < 50:
        suggestions.append("Add specific numbers, names, or quantities")

    complete_prompt, template_prompt = _analyzer._generate_complete_prompt(
        prompt, task_type, analysis
    )
    after_analysis = _run_analysis(complete_prompt)
    platforms = _analyzer._evaluate_platforms(task_type, top_n=5)

    return {
        "original_prompt":  prompt,
        "complete_prompt":  complete_prompt,
        "template_prompt":  template_prompt,
        "detected_task_type": task_type,
        "before":           analysis,
        "after":            after_analysis,
        "suggestions":      suggestions,
        "platforms":        platforms,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_TASK_EMOJI = {
    "creative_writing":  "✍️",
    "code_generation":   "💻",
    "data_analysis":     "📊",
    "summarization":     "📝",
    "question_answering": "🔎",
    "translation":       "🌐",
    "general":           "💬",
}

_TASK_LABEL = {
    "creative_writing":  "Creative Writing",
    "code_generation":   "Code Generation",
    "data_analysis":     "Data Analysis",
    "summarization":     "Summarization",
    "question_answering": "Q&A / Research",
    "translation":       "Translation",
    "general":           "General",
}

_SPEED_LABEL = {
    "very_fast": "⚡⚡ ultra",
    "fast":      "⚡ fast",
    "medium":    "🕐 med",
    "slow":      "🐢 slow",
}


def _stars(score: int) -> str:
    filled = round(score / 20)   # 0-5
    return "★" * filled + "☆" * (5 - filled)


def _score_emoji(score: int) -> str:
    if score >= 75:
        return "✅"
    if score >= 50:
        return "⚠️"
    return "❌"


def _overall_verdict(score: int) -> str:
    if score >= 80:
        return "🌟 *Excellent* — sharp and ready to go\\!"
    if score >= 65:
        return "✅ *Good* — a couple of tweaks will make it great"
    if score >= 50:
        return "⚠️ *Fair* — it'll work, but the AI might miss what you mean"
    if score >= 30:
        return "❌ *Needs work* — the AI will probably give you a vague answer"
    return "🚨 *Very weak* — let's fix this together"


def _dim_verdict(score: int, dim: str) -> str:
    table: dict[str, list[tuple[int, str]]] = {
        "clarity":      [(75, "crystal clear"), (50, "a bit vague"),   (25, "too ambiguous"), (0, "very unclear")],
        "specificity":  [(75, "nicely specific"), (50, "needs more detail"), (25, "too generic"), (0, "very generic")],
        "structure":    [(75, "well organised"), (50, "could be tidier"), (25, "a bit scattered"), (0, "hard to follow")],
        "completeness": [(75, "covers the bases"), (50, "missing a few things"), (25, "missing a lot"), (0, "very incomplete")],
    }
    for threshold, label in table.get(dim, [(0, "okay")]):
        if score >= threshold:
            return label
    return "poor"


def _escape_md(text: str) -> str:
    special = r"_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{ch}" if ch in special else ch for ch in text)


# ---------------------------------------------------------------------------
# Message 1 — analysis check (/analyze)
# ---------------------------------------------------------------------------

def _format_analysis(analysis: dict[str, Any], prompt: str) -> str:
    dims     = analysis["dimensions"]
    overall  = analysis["overall_score"]
    task_type = analysis["detected_task_type"]
    label    = _TASK_LABEL.get(task_type, task_type.replace("_", " ").title())

    lines = [
        "🔍 *Prompt Check*\n",
        f"_{_escape_md(prompt)}_\n",
        _overall_verdict(overall) + "\n",
        f"Detected: {_TASK_EMOJI.get(task_type, '💬')} {_escape_md(label)}\n",
        "*What I found:*",
    ]
    for dim, lbl in [("clarity", "Clarity"), ("specificity", "Specificity"),
                     ("structure", "Structure"), ("completeness", "Completeness")]:
        score   = dims[dim]["score"]
        lines.append(f"{_score_emoji(score)} *{lbl}* — {_escape_md(_dim_verdict(score, dim))}")

    missing = dims["completeness"].get("missing", [])
    if missing:
        lines.append("\n💡 *Quick wins:*")
        for item in missing:
            lines.append(f"• Add {_escape_md(item.lower())}")

    lines.append("\n_Send your prompt as a plain message and I'll fully rewrite it\\!_")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Message 1 — optimized prompts + score (/optimize / plain text)
# ---------------------------------------------------------------------------

def _format_optimization(result: dict[str, Any]) -> str:
    task_type  = result["detected_task_type"]
    label      = _TASK_LABEL.get(task_type, task_type.replace("_", " ").title())
    before     = result["before"]["overall_score"]
    after      = result["after"]["overall_score"]
    diff       = after - before
    diff_str   = f"\\+{diff}" if diff >= 0 else f"\\-{abs(diff)}"

    lines = [
        f"✨ *Your upgraded prompt*\n",
        f"Detected: {_TASK_EMOJI.get(task_type, '💬')} {_escape_md(label)}\n",
        "📋 *Ready to use — copy \\& paste:*",
        f"```\n{result['complete_prompt']}\n```\n",
        "🔧 *Template \\(fill in the \\[brackets\\]\\):*",
        f"```\n{result['template_prompt']}\n```\n",
        f"📈 *Score: {before} → {after} \\({diff_str}\\)*",
    ]

    for dim, lbl in [("clarity", "Clarity"), ("specificity", "Specificity"),
                     ("structure", "Structure"), ("completeness", "Completeness")]:
        b      = result["before"]["dimensions"][dim]["score"]
        a      = result["after"]["dimensions"][dim]["score"]
        change = a - b
        chstr  = f"\\+{change}" if change >= 0 else f"\\-{abs(change)}"
        lines.append(f"  {_score_emoji(a)} {lbl}: {b} → {a} \\({chstr}\\)")

    if result["suggestions"]:
        lines.append("\n💡 *Tips to make it even better:*")
        for i, s in enumerate(result["suggestions"], 1):
            lines.append(f"{i}\\. {_escape_md(s)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Message 2 — platform comparison table
# ---------------------------------------------------------------------------

def _format_platform_table(result: dict[str, Any]) -> str:
    task_type = result["detected_task_type"]
    label     = _TASK_LABEL.get(task_type, task_type.replace("_", " ").title())
    platforms = result["platforms"]   # already top-5, sorted by quality

    if not platforms:
        return ""

    top = platforms[0]
    medals = ["🥇", "🥈", "🥉", "4\\.", "5\\."]

    lines = [
        f"🤖 *Where to run this prompt*\n",
        f"Task: {_TASK_EMOJI.get(task_type, '💬')} {_escape_md(label)}\n",
        f"🏆 *Best pick: {_escape_md(top['label'])}*",
    ]

    # API params for top pick (if it has an API)
    if top["api"] and top.get("model_id") and top["settings"]:
        s = top["settings"]
        lines.append("```")
        lines.append(f"model:       \"{top['model_id']}\"")
        if "temperature" in s:
            lines.append(f"temperature: {s['temperature']}")
        if "max_tokens" in s:
            lines.append(f"max_tokens:  {s['max_tokens']}")
        if s.get("extended_thinking"):
            lines.append("extended_thinking: true")
        if top.get("web_search"):
            lines.append("# enable web search / grounding tool")
        lines.append("```\n")
    elif not top["api"]:
        lines.append(f"_UI only — {_escape_md(top['cost_note'])}_\n")

    # Comparison table
    lines.append(f"📊 *Top 5 for {_escape_md(label)}:*")
    for i, p in enumerate(platforms):
        medal   = medals[i] if i < len(medals) else f"{i+1}\\."
        stars   = _escape_md(_stars(p["quality_score"]))
        speed   = _escape_md(_SPEED_LABEL.get(p["speed"], p["speed"]))
        cost    = _escape_md(p["cost_note"])
        plabel  = _escape_md(p["label"])
        search  = " 🌐" if p.get("web_search") else ""
        think   = " 🧠" if p.get("extended_thinking") else ""
        lines.append(f"{medal} {plabel}{search}{think}")
        lines.append(f"   {stars}  {speed}  💰{cost}")

    # "Use X instead if..." note for the most relevant specialist not in top-5
    lines.append("")
    top_ids = {p["id"] for p in platforms}

    # Surface Perplexity for non-research tasks
    if task_type != "question_answering" and "perplexity_sonar_pro" not in top_ids:
        lines.append(
            "💡 *Use Perplexity Sonar Pro instead* if your prompt needs\n"
            "   live web data or cited sources\\."
        )
    # Surface o3/DeepSeek R1 for non-reasoning tasks
    elif task_type not in ("data_analysis", "code_generation") and "o3" not in top_ids:
        lines.append(
            "💡 *Use o3 or DeepSeek R1 instead* if this becomes a\n"
            "   complex logic or maths problem\\."
        )
    # Surface Groq for any task where speed matters
    if "groq_llama" not in top_ids:
        lines.append(
            "⚡ *Use Groq \\(Llama 3\\.3 70B\\) instead* if you need\n"
            "   600\\+ tokens/sec for real\\-time streaming\\."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "👋 *Hey\\! I'm your Prompt Optimizer\\.*\n\n"
        "Send me any AI prompt and I'll rewrite it into a production\\-ready version "
        "— complete, no placeholders — plus tell you exactly which model and settings to use\\.\n\n"
        "*How to use me:*\n"
        "💬 Just send any prompt as a message — I'll rewrite it\n"
        "🔍 /analyze \\<prompt\\> — See what's weak about your prompt\n"
        "✨ /optimize \\<prompt\\> — Full rewrite \\+ platform recommendations\n"
        "❓ /help — Tips and examples\n\n"
        "_Try it now — just paste a prompt you've been using\\!_"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "❓ *How to get the best out of me*\n\n"
        "*Fastest way:* send any prompt as a plain message\\.\n"
        "You'll get a ready\\-to\\-use rewrite \\+ a customisable template \\+ platform picks\\.\n\n"
        "*Commands:*\n"
        "🔍 /analyze — Quick breakdown of what's strong and weak\n"
        "✨ /optimize — Full rewrite \\+ score comparison \\+ platform table\n\n"
        "*What I check:*\n"
        "✅ *Clarity* — Specific enough for the AI to understand\\?\n"
        "🎯 *Specificity* — Concrete details, numbers, names\\?\n"
        "🏗 *Structure* — Organised with a clear ask\\?\n"
        "📋 *Completeness* — Role, context, format, constraints\\?\n\n"
        "*Platforms I evaluate \\(15 total\\):*\n"
        "Claude Sonnet/Opus/Haiku, GPT\\-5\\.2, o3, Gemini 2\\.5 Pro/Flash,\n"
        "Grok 4, DeepSeek V3/R1, Mistral Large 3,\n"
        "Perplexity Sonar Pro, Groq, GitHub Copilot Pro\\+\n\n"
        "*Example prompts to try:*\n"
        "`write me a story about AI`\n"
        "`summarize this article for me`\n"
        "`help me write a Python function`"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text   = update.message.text or ""
    prompt = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else ""
    if not prompt:
        await update.message.reply_text(
            "🔍 Give me something to check\\!\n\nUsage: `/analyze write me a story about AI`",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return
    analysis = _run_analysis(prompt)
    await update.message.reply_text(
        _format_analysis(analysis, prompt), parse_mode=ParseMode.MARKDOWN_V2
    )


async def optimize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text   = update.message.text or ""
    prompt = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else ""
    if not prompt:
        await update.message.reply_text(
            "✨ Give me a prompt to upgrade\\!\n\nUsage: `/optimize write me a story about AI`",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return
    await _send_optimization(update, prompt)


async def plain_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    prompt = (update.message.text or "").strip()
    if not prompt:
        return
    await _send_optimization(update, prompt)


async def _send_optimization(update: Update, prompt: str) -> None:
    """Run optimization and send both messages."""
    result = _run_optimization(prompt)

    # Message 1 — prompts + score
    await update.message.reply_text(
        _format_optimization(result), parse_mode=ParseMode.MARKDOWN_V2
    )

    # Message 2 — platform table (best-effort; don't let a format error kill msg 1)
    try:
        table = _format_platform_table(result)
        if table:
            await update.message.reply_text(table, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception:
        logger.exception("Platform table send failed — skipping")
