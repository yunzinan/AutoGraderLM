"""Path validation helpers for user-controlled filesystem keys."""

from __future__ import annotations


def is_safe_path_segment(value: str) -> bool:
    """Return True when value is a single relative path segment, not a path."""
    if not isinstance(value, str):
        return False
    if not value or value in {".", ".."}:
        return False
    if "\x00" in value or "/" in value or "\\" in value:
        return False
    return True


def require_safe_path_segment(value: str, label: str = "path segment") -> str:
    """Validate and return a user-controlled filesystem key."""
    if not is_safe_path_segment(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value
