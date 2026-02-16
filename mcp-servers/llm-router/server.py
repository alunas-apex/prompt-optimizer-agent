"""LLM Router MCP Server.

Provides tools to route prompts to different LLM APIs (Claude, GPT, Gemini,
Perplexity) and run parallel comparisons across all models.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

mcp = FastMCP("llm-router", instructions="Routes prompts to multiple LLM providers")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class LLMRequest(BaseModel):
    """Common request shape for every LLM call."""

    prompt: str = Field(..., description="The user prompt to send")
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System-level instructions",
    )
    model: str = Field(default="", description="Model name override")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=128_000)


class LLMResponse(BaseModel):
    """Standardised response returned from every provider."""

    provider: str
    model: str
    content: str
    usage: dict[str, int]
    latency_seconds: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

async def _call_claude_api(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> LLMResponse:
    """Call the Anthropic Claude API."""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        model = model or "claude-sonnet-4-20250514"

        start = time.perf_counter()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.perf_counter() - start

        return LLMResponse(
            provider="anthropic",
            model=model,
            content=response.content[0].text,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            latency_seconds=round(latency, 3),
        )
    except Exception as exc:
        logger.exception("Claude API error")
        return LLMResponse(
            provider="anthropic",
            model=model or "claude-sonnet-4-20250514",
            content="",
            usage={},
            latency_seconds=0.0,
            error=str(exc),
        )


async def _call_gpt_api(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> LLMResponse:
    """Call the OpenAI GPT API."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = model or "gpt-4o"

        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        latency = time.perf_counter() - start

        choice = response.choices[0]
        return LLMResponse(
            provider="openai",
            model=model,
            content=choice.message.content or "",
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            latency_seconds=round(latency, 3),
        )
    except Exception as exc:
        logger.exception("GPT API error")
        return LLMResponse(
            provider="openai",
            model=model or "gpt-4o",
            content="",
            usage={},
            latency_seconds=0.0,
            error=str(exc),
        )


async def _call_gemini_api(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> LLMResponse:
    """Call the Google Gemini API."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model_name = model or "gemini-1.5-flash"

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            generation_config=generation_config,
        )

        start = time.perf_counter()
        response = gemini_model.generate_content(prompt)
        latency = time.perf_counter() - start

        usage_meta = response.usage_metadata
        return LLMResponse(
            provider="google",
            model=model_name,
            content=response.text or "",
            usage={
                "input_tokens": getattr(usage_meta, "prompt_token_count", 0),
                "output_tokens": getattr(usage_meta, "candidates_token_count", 0),
            },
            latency_seconds=round(latency, 3),
        )
    except Exception as exc:
        logger.exception("Gemini API error")
        return LLMResponse(
            provider="google",
            model=model or "gemini-1.5-flash",
            content="",
            usage={},
            latency_seconds=0.0,
            error=str(exc),
        )


async def _call_perplexity_api(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> LLMResponse:
    """Call the Perplexity API (OpenAI-compatible)."""
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai",
        )
        model = model or "sonar"

        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        latency = time.perf_counter() - start

        choice = response.choices[0]
        return LLMResponse(
            provider="perplexity",
            model=model,
            content=choice.message.content or "",
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            latency_seconds=round(latency, 3),
        )
    except Exception as exc:
        logger.exception("Perplexity API error")
        return LLMResponse(
            provider="perplexity",
            model=model or "sonar",
            content="",
            usage={},
            latency_seconds=0.0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def call_claude(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Send a prompt to the Anthropic Claude API and return the response."""
    result = await _call_claude_api(prompt, system_prompt, model, temperature, max_tokens)
    return result.model_dump()


@mcp.tool()
async def call_gpt(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Send a prompt to the OpenAI GPT API and return the response."""
    result = await _call_gpt_api(prompt, system_prompt, model, temperature, max_tokens)
    return result.model_dump()


@mcp.tool()
async def call_gemini(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "gemini-1.5-flash",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Send a prompt to the Google Gemini API and return the response."""
    result = await _call_gemini_api(prompt, system_prompt, model, temperature, max_tokens)
    return result.model_dump()


@mcp.tool()
async def call_perplexity(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "sonar",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Send a prompt to the Perplexity API and return the response."""
    result = await _call_perplexity_api(prompt, system_prompt, model, temperature, max_tokens)
    return result.model_dump()


@mcp.tool()
async def call_all_llms(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Send a prompt to all LLM providers in parallel and return all responses.

    This is useful for comparing outputs across models on the same input.
    """
    tasks = [
        _call_claude_api(prompt, system_prompt, "", temperature, max_tokens),
        _call_gpt_api(prompt, system_prompt, "", temperature, max_tokens),
        _call_gemini_api(prompt, system_prompt, "", temperature, max_tokens),
        _call_perplexity_api(prompt, system_prompt, "", temperature, max_tokens),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    responses: dict[str, Any] = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error("Provider call failed: %s", result)
            continue
        responses[result.provider] = result.model_dump()

    return {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "parameters": {"temperature": temperature, "max_tokens": max_tokens},
        "responses": responses,
        "providers_called": len(responses),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
