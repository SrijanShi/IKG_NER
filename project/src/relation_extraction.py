from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from .preprocess import TextValue

Relation = tuple[str, str, str]


def _get_text_field(row_processed: Mapping[str, TextValue | list[str]], key: str) -> TextValue:
    value = row_processed.get(key)
    if isinstance(value, dict):
        return value  # type: ignore[return-value]
    raise KeyError(f"Missing text field: {key}")


def _as_iterable(value: Mapping[str, TextValue | list[str]], key: str) -> Iterable[str]:
    items = value.get(key)
    if isinstance(items, list):
        return items
    raise KeyError(f"Missing list field: {key}")


def _extract_date_components(date_value: TextValue) -> tuple[str | None, str | None, str | None]:
    normalized = date_value["normalized"]
    parts = normalized.split()
    if not parts:
        return None, None, None
    month = parts[0]
    year = None
    hour = None
    for part in parts:
        if part.isdigit() and len(part) == 4:
            year = part
        if ":" in part:
            hour = part.split(":", 1)[0]
    return year, month, hour


def extract_relations(
    row_processed: Mapping[str, TextValue | list[str]],
    ner_result: Mapping[str, Sequence[str]] | Mapping[str, str | list[str]],
    post_id: str,
) -> list[Relation]:
    relations: list[Relation] = []

    username = _get_text_field(row_processed, "username")
    date = _get_text_field(row_processed, "date")
    event_type = _get_text_field(row_processed, "event_type")
    keywords = _as_iterable(row_processed, "keywords")
    reply_usernames = _as_iterable(row_processed, "reply_usernames")

    relations.append((username["normalized"], "posted", post_id))

    mention_usernames = set(reply_usernames)
    for mention in ner_result.get("usernames", []):
        mention_usernames.add(mention)
    for mention in mention_usernames:
        relations.append((post_id, "mentionsUser", mention))

    for hashtag in ner_result.get("hashtags", []):
        relations.append((post_id, "hasHashtag", hashtag))

    for keyword in keywords:
        relations.append((post_id, "hasKeyword", keyword))

    for entity in ner_result.get("capitalized_phrases", []):
        relations.append((post_id, "mentionsEntity", entity))

    sentiment = ner_result.get("sentiment_bucket")
    if isinstance(sentiment, str):
        relations.append((post_id, "hasSentiment", sentiment))

    for event_entity in ner_result.get("event_entities", []):
        relations.append((post_id, "isAboutEvent", event_entity))

    for number in ner_result.get("numbers", []):
        relations.append((post_id, "hasNumber", number))

    year, month, hour = _extract_date_components(date)
    if year:
        relations.append((post_id, "postedAtYear", year))
    if month:
        relations.append((post_id, "postedAtMonth", month))
    if hour:
        relations.append((post_id, "postedAtHour", hour))

    if isinstance(ner_result.get("bigrams"), list):
        for bigram in ner_result["bigrams"]:  # type: ignore[index]
            relations.append((post_id, "mentionsEntity", bigram))

    return relations

__all__ = ["extract_relations"]
