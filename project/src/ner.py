from __future__ import annotations

import re

from .preprocess import normalize_whitespace, remove_invisible_characters

USERNAME_PATTERN = re.compile(r"@([A-Za-z0-9_]+)")
HASHTAG_PATTERN = re.compile(r"#[A-Za-z0-9_]+")
CAPITALIZED_PHRASE_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z][a-z]+)\b"
)
NUMBER_PATTERN = re.compile(r"\b(?:\d+[\/\d]*|\d{1,3}(?:,\d{3})*)(?:\.\d+)?\b")
BIGRAM_PATTERN = re.compile(r"\b([A-Za-z]+\s+[A-Za-z]+)\b")

POSITIVE_WORDS = {
    "win",
    "winner",
    "great",
    "good",
    "amazing",
    "excited",
    "love",
    "hero",
    "champion",
    "victory",
    "celebrate",
}
NEGATIVE_WORDS = {
    "lose",
    "loss",
    "bad",
    "terrible",
    "awful",
    "hate",
    "sad",
    "angry",
    "defeat",
    "injury",
}

def _prepare_text(text: str | None) -> str:
    if not text:
        return ""
    return normalize_whitespace(remove_invisible_characters(text))

def extract_usernames(text: str | None) -> list[str]:
    cleaned = _prepare_text(text)
    return [match.group(1).lower() for match in USERNAME_PATTERN.finditer(cleaned)]

def extract_hashtags(text: str | None) -> list[str]:
    cleaned = _prepare_text(text)
    return [tag.lower() for tag in HASHTAG_PATTERN.findall(cleaned)]

def extract_capitalized_phrases(text: str | None) -> list[str]:
    cleaned = _prepare_text(text)
    phrases = [phrase.strip() for phrase in CAPITALIZED_PHRASE_PATTERN.findall(cleaned)]
    return sorted(set(phrases), key=str.lower)

def extract_numbers(text: str | None) -> list[str]:
    cleaned = _prepare_text(text)
    return NUMBER_PATTERN.findall(cleaned)

def extract_bigrams(text: str | None) -> list[str]:
    cleaned = _prepare_text(text)
    return [match.group(1).lower() for match in BIGRAM_PATTERN.finditer(cleaned)]

def extract_sentiment_bucket(text: str | None) -> str:
    cleaned = _prepare_text(text).lower()
    positive_hits = sum(1 for word in POSITIVE_WORDS if word in cleaned)
    negative_hits = sum(1 for word in NEGATIVE_WORDS if word in cleaned)
    if positive_hits > negative_hits:
        return "sent_positive"
    if negative_hits > positive_hits:
        return "sent_negative"
    return "sent_neutral"

def extract_event_entities(event_type: str | None) -> list[str]:
    cleaned = _prepare_text(event_type)
    tokens = re.split(r"\s+", cleaned)
    return [token for token in tokens if token]

__all__ = [
    "extract_usernames",
    "extract_hashtags",
    "extract_capitalized_phrases",
    "extract_numbers",
    "extract_bigrams",
    "extract_sentiment_bucket",
    "extract_event_entities",
]
