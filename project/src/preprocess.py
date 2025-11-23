from __future__ import annotations

import re
import unicodedata
import ast
from typing import Iterable, Mapping, TypedDict


MENTION_PATTERN = re.compile(r"@([A-Za-z0-9_]+)")
HASHTAG_PATTERN = re.compile(r"#([A-Za-z0-9_]+)")


class TextValue(TypedDict):
    original: str
    normalized: str
    lower: str

def remove_invisible_characters(text: str) -> str:
    if not text:
        return ""
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in {"Cc", "Cf"}
    )

def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def convert_entities(text: str) -> str:
    def _mention_repl(match: re.Match[str]) -> str:
        return f"user_{match.group(1).lower()}"
    def _hashtag_repl(match: re.Match[str]) -> str:
        return f"hashtag_{match.group(1).lower()}"
    return MENTION_PATTERN.sub(_mention_repl, HASHTAG_PATTERN.sub(_hashtag_repl, text))

def prepare_text(value: str | None) -> TextValue:
    original = value or ""
    cleaned = convert_entities(normalize_whitespace(remove_invisible_characters(original)))
    return {"original": original, "normalized": cleaned, "lower": cleaned.lower()}

def _collect_reply_sources(row: Mapping[str, str | None]) -> Iterable[str]:
    keys = ("reply_usernames", "in_reply_to_usernames", "in_reply_to_screen_name", "replying_to")
    for key in keys:
        value = row.get(key)
        if value:
            yield value

def extract_usernames(text: str) -> list[str]:
    if not text:
        return []
    mentions = MENTION_PATTERN.findall(text)
    if mentions:
        return [mention.lower() for mention in mentions]
    tokens = [token for token in re.split(r"[|,;\s]+", text) if token]
    return [token.lstrip("@").lower() for token in tokens if token]

def parse_keywords(value: str | None) -> list[str]:
    if not value:
        return []
    # Attempt to parse simple Python list literals commonly stored in CSV exports.
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = None

    items: list[str]
    if isinstance(parsed, (list, tuple)):
        items = [str(item) for item in parsed if isinstance(item, str)]
    else:
        cleaned = convert_entities(normalize_whitespace(remove_invisible_characters(value)))
        items = [piece for piece in re.split(r"[|,;]+", cleaned) if piece]

    return [
        convert_entities(normalize_whitespace(remove_invisible_characters(item))).lower()
        for item in items
        if item
    ]

def preprocess_row(row: Mapping[str, str | None]) -> dict[str, TextValue | list[str]]:
    username = prepare_text(row.get("username"))
    content = prepare_text(row.get("content"))
    date = prepare_text(row.get("date"))
    event_type = prepare_text(row.get("event_type"))
    reply_sources = " ".join(_collect_reply_sources(row))
    reply_usernames = extract_usernames(reply_sources)
    keywords = parse_keywords(row.get("matched_keywords"))
    return {
        "username": username,
        "content": content,
        "date": date,
        "reply_usernames": reply_usernames,
        "event_type": event_type,
        "keywords": keywords,
    }

__all__ = [
    "normalize_whitespace",
    "remove_invisible_characters",
    "convert_entities",
    "prepare_text",
    "extract_usernames",
    "parse_keywords",
    "preprocess_row",
]
