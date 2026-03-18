"""Central configuration and path helpers."""

from __future__ import annotations

from pathlib import Path

# Anthropic model identifiers
CLAUDE_SONNET_4_6 = "claude-sonnet-4-6"
CLAUDE_SONNET_4_5 = "claude-sonnet-4-5"   # primary

# Gemini model identifiers (used as fallback)
GEMINI_2_5_PRO   = "gemini-2.5-pro"
GEMINI_2_5_FLASH = "gemini-2.5-flash"

# Groq model identifiers
GROQ_LLAMA_3_3_70B_VERSATILE = "llama-3.3-70b-versatile"

# Default model for huddle analysis
DEFAULT_HUDDLE_MODEL  = CLAUDE_SONNET_4_5
FALLBACK_HUDDLE_MODEL = GEMINI_2_5_FLASH


def project_root() -> Path:
    """Return repository root based on package location."""
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a relative path against repo root, leave absolute paths unchanged."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def output_dir() -> Path:
    """Return (and create if needed) the shared output folder for JSON/PDF results."""
    path = project_root() / "output"
    path.mkdir(exist_ok=True)
    return path
