from __future__ import annotations

import csv
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
# Ensure the src package is importable when running the script directly.
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src import ner
from src.idgen import make_post_id
from src.preprocess import preprocess_row
from src.relation_extraction import extract_relations


def main() -> None:
    csv_path = project_root / "data" / "Cricket.csv"

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        headers = reader.fieldnames or []

    print(f"Number of rows: {len(rows)}")
    print(f"Header columns: {headers}")
    print("First 3 rows:")
    for row in rows[:3]:
        print(row)
    
    first_row_preprocessed = preprocess_row(rows[0])
    print("\nPreprocessed first row:")
    print(first_row_preprocessed)

    content_text = first_row_preprocessed["content"]["original"]
    event_text = first_row_preprocessed["event_type"]["original"]

    print("\nExtracted hashtags from first row content:")
    print(ner.extract_hashtags(content_text))
    print("\nExtracted usernames from first row content:")
    print(ner.extract_usernames(content_text))

    ner_result = {
        "usernames": ner.extract_usernames(content_text),
        "hashtags": ner.extract_hashtags(content_text),
        "capitalized_phrases": ner.extract_capitalized_phrases(content_text),
        "numbers": ner.extract_numbers(content_text),
        "bigrams": ner.extract_bigrams(content_text),
        "sentiment_bucket": ner.extract_sentiment_bucket(content_text),
        "event_entities": ner.extract_event_entities(event_text),
    }

    post_id = make_post_id(1, rows[0])
    print("\nExtracted relations from preprocessed first row:")
    rels = extract_relations(first_row_preprocessed, ner_result, post_id)
    print(rels)
    print("\nGenerated post ID for first row:")
    print(post_id)
    




if __name__ == "__main__":
    main()
