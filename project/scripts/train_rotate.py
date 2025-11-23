from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kg_utils import build_entity_relation_mappings, split_train_val_test
from src.rotate import train_rotate

TRIPLES_PATH = PROJECT_ROOT / "data" / "Cricket_triples.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "rotate_model.json"


def load_triples(path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            head = row.get("head")
            relation = row.get("relation")
            tail = row.get("tail")
            if head and relation and tail:
                triples.append((head, relation, tail))
    return triples


def save_model(
    ent_re: list[list[float]],
    ent_im: list[list[float]],
    rel_phase: list[list[float]],
    path: Path,
) -> None:
    payload = {
        "ent_re": ent_re,
        "ent_im": ent_im,
        "rel_phase": rel_phase,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_file:
        json.dump(payload, out_file)


def main() -> None:
    triples = load_triples(TRIPLES_PATH)
    if not triples:
        print("No triples found; aborting training.")
        return

    entity_map, relation_map = build_entity_relation_mappings(triples)
    train_split, val_split, test_split = split_train_val_test(
        triples,
        0.8,
        0.1,
        0.1,
        seed=42,
    )

    print(
        "Training RotatE with "
        f"{len(train_split)} train / {len(val_split)} val / {len(test_split)} test triples"
    )

    ent_re, ent_im, rel_phase, metrics = train_rotate(
        train_split,
        val_split,
        entity_map,
        relation_map,
        epochs=50,
        dim=96,
        learning_rate=0.005,
        neg_ratio=5,
        seed=42,
        log_every=5,
    )

    save_model(ent_re, ent_im, rel_phase, MODEL_PATH)

    print(f"Final MRR: {metrics.get('MRR', 0.0):.4f}")
    print(f"Hits@1: {metrics.get('Hits@1', 0.0):.4f}")
    print(f"Hits@10: {metrics.get('Hits@10', 0.0):.4f}")


if __name__ == "__main__":
    main()
