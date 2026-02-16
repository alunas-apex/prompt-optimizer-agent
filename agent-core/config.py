"""Configuration loader for the Prompt Optimizer Agent.

Reads config/agent_config.yaml and exposes typed settings used by the
optimizer and main orchestrator.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "agent_config.yaml"


# ---------------------------------------------------------------------------
# Pydantic settings models
# ---------------------------------------------------------------------------

class ModelDefaults(BaseModel):
    """Default parameters for LLM calls."""

    temperature: float = 0.7
    max_tokens: int = 1024
    default_provider: str = "anthropic"
    default_model: str = "claude-sonnet-4-20250514"


class CostConfig(BaseModel):
    """Per-provider cost rates (USD per 1 000 tokens)."""

    anthropic_input: float = 0.003
    anthropic_output: float = 0.015
    openai_input: float = 0.005
    openai_output: float = 0.015
    google_input: float = 0.00035
    google_output: float = 0.00105
    perplexity_input: float = 0.001
    perplexity_output: float = 0.001


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file: str | None = None


class MCPServerConfig(BaseModel):
    """Settings for a single MCP server."""

    name: str
    module: str
    description: str = ""


class AgentConfig(BaseModel):
    """Top-level agent configuration."""

    model_defaults: ModelDefaults = Field(default_factory=ModelDefaults)
    cost: CostConfig = Field(default_factory=CostConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    optimization_rounds: int = 3
    parallel_evaluation: bool = True


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> AgentConfig:
    """Load and return an AgentConfig from a YAML file.

    Falls back to built-in defaults when the file is missing or unreadable.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    raw: dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path) as fh:
                raw = yaml.safe_load(fh) or {}
            logger.info("Loaded config from %s", config_path)
        except Exception:
            logger.warning("Failed to read %s — using defaults", config_path, exc_info=True)
    else:
        logger.info("Config file not found at %s — using defaults", config_path)

    return AgentConfig(**raw)


def setup_logging(config: AgentConfig) -> None:
    """Configure the root logger from *config.logging*."""
    log_cfg = config.logging
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_cfg.file:
        handlers.append(logging.FileHandler(log_cfg.file))

    logging.basicConfig(
        level=getattr(logging, log_cfg.level.upper(), logging.INFO),
        format=log_cfg.format,
        handlers=handlers,
        force=True,
    )


def get_api_keys() -> dict[str, str | None]:
    """Return a dict of available API keys from the environment."""
    return {
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "google": os.getenv("GOOGLE_API_KEY"),
        "perplexity": os.getenv("PERPLEXITY_API_KEY"),
    }


def validate_api_keys() -> dict[str, bool]:
    """Check which API keys are configured (non-empty)."""
    keys = get_api_keys()
    return {provider: bool(key) for provider, key in keys.items()}
