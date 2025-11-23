from __future__ import annotations


def make_post_id(index: int, row: object | None = None) -> str:
    """Return a stable post identifier for embedding models."""
    return f"POST_{index}"


__all__ = ["make_post_id"]
