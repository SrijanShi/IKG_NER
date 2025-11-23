from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple

Triple = Tuple[str, str, str]


def build_entity_relation_mappings(triples: Iterable[Triple]) -> tuple[dict[str, int], dict[str, int]]:
    entities: dict[str, int] = {}
    relations: dict[str, int] = {}

    for head, relation, tail in triples:
        if head not in entities:
            entities[head] = len(entities)
        if tail not in entities:
            entities[tail] = len(entities)
        if relation not in relations:
            relations[relation] = len(relations)

    return entities, relations


def split_train_val_test(
    triples: Sequence[Triple],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    seed: int | None = None,
) -> tuple[list[Triple], list[Triple], list[Triple]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    triples_list: List[Triple] = list(triples)
    if seed is not None:
        random.seed(seed)
    random.shuffle(triples_list)

    n = len(triples_list)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = triples_list[:train_end]
    val = triples_list[train_end:val_end]
    test = triples_list[val_end:]

    return train, val, test


__all__ = ["build_entity_relation_mappings", "split_train_val_test"]
