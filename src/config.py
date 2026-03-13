"""Central configuration and path helpers."""

from __future__ import annotations

from pathlib import Path

# Claude model identifiers (Anthropic API)
CLAUDE_SONNET_4_6 = "claude-sonnet-4-6"
CLAUDE_SONNET_4_5 = "claude-sonnet-4-5-20250929"
CLAUDE_SONNET_4 = "claude-sonnet-4"

# Default model for huddle analysis
DEFAULT_HUDDLE_MODEL = CLAUDE_SONNET_4_6


def project_root() -> Path:
    """Return repository root based on package location."""
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a relative path against repo root, leave absolute paths unchanged."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate
