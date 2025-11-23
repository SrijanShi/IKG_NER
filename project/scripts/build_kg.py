from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import ner
from src.idgen import make_post_id
from src.preprocess import preprocess_row
from src.relation_extraction import extract_relations
from src.kg_utils import build_entity_relation_mappings, split_train_val_test

INPUT_PATH = PROJECT_ROOT / "data" / "Cricket.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "Cricket_triples.csv"


def _run_ner(row_preprocessed: Mapping[str, object]) -> dict[str, object]:
    content_text = row_preprocessed["content"]["original"]  # type: ignore[index]
    event_text = row_preprocessed["event_type"]["original"]  # type: ignore[index]

    return {
        "usernames": ner.extract_usernames(content_text),
        "hashtags": ner.extract_hashtags(content_text),
        "capitalized_phrases": ner.extract_capitalized_phrases(content_text),
        "numbers": ner.extract_numbers(content_text),
        "bigrams": ner.extract_bigrams(content_text),
        "sentiment_bucket": ner.extract_sentiment_bucket(content_text),
        "event_entities": ner.extract_event_entities(event_text),
    }


def build_kg() -> None:
    with INPUT_PATH.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        triples: set[tuple[str, str, str]] = set()

        for index, row in enumerate(reader, start=1):
            row_preprocessed = preprocess_row(row)
            ner_result = _run_ner(row_preprocessed)
            post_id = make_post_id(index, row)

            relations = extract_relations(row_preprocessed, ner_result, post_id)

            triples.update(relations)

    triples_list = sorted(triples)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["head", "relation", "tail"])
        for head, relation, tail in triples_list:
            writer.writerow([head, relation, tail])

    entity_map, relation_map = build_entity_relation_mappings(triples_list)
    train_split, val_split, test_split = split_train_val_test(
        triples_list,
        0.8,
        0.1,
        0.1,
        seed=42,
    )

    print(f"Number of triples: {len(triples_list)}")
    print(f"Number of entities: {len(entity_map)}")
    print(f"Number of relations: {len(relation_map)}")
    print(
        "Train/Val/Test sizes: "
        f"{len(train_split)}/{len(val_split)}/{len(test_split)}"
    )


if __name__ == "__main__":
    build_kg()
