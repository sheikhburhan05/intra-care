"""Central configuration and path helpers."""

from __future__ import annotations

from pathlib import Path

# Anthropic Sonnet (kept for reference; currently disabled)
# CLAUDE_SONNET_4_6 = "claude-sonnet-4-6"
# CLAUDE_SONNET_4_5 = "claude-sonnet-4-5-20250929"
# CLAUDE_SONNET_4 = "claude-sonnet-4"
# DEFAULT_HUDDLE_MODEL = CLAUDE_SONNET_4_6

# Gemini model identifiers
GEMINI_2_5_FLASH = "gemini-2.5-flash"

# Groq model identifiers
GROQ_LLAMA_3_3_70B_VERSATILE = "llama-3.3-70b-versatile"

# Default model for huddle analysis
DEFAULT_HUDDLE_MODEL = GEMINI_2_5_FLASH
FALLBACK_HUDDLE_MODEL = GROQ_LLAMA_3_3_70B_VERSATILE


def project_root() -> Path:
    """Return repository root based on package location."""
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a relative path against repo root, leave absolute paths unchanged."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate
