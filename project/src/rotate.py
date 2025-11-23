from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

Triple = Tuple[str, str, str]
IndexedTriple = Tuple[int, int, int]


def initialize_embeddings(
    num_entities: int,
    num_relations: int,
    dim: int,
    seed: int | None = None,
) -> tuple[List[List[float]], List[List[float]], List[List[float]]]:
    rng = random.Random(seed)
    ent_re = [[rng.uniform(-0.5, 0.5) for _ in range(dim)] for _ in range(num_entities)]
    ent_im = [[rng.uniform(-0.5, 0.5) for _ in range(dim)] for _ in range(num_entities)]
    rel_phase = [[rng.uniform(0.0, 2.0 * math.pi) for _ in range(dim)] for _ in range(num_relations)]
    return ent_re, ent_im, rel_phase


def rotate_entity(
    head_re: Sequence[float],
    head_im: Sequence[float],
    relation_phase: Sequence[float],
) -> tuple[List[float], List[float]]:
    rotated_re: List[float] = []
    rotated_im: List[float] = []
    for re, im, phase in zip(head_re, head_im, relation_phase):
        cos_phase = math.cos(phase)
        sin_phase = math.sin(phase)
        rotated_re.append(re * cos_phase - im * sin_phase)
        rotated_im.append(re * sin_phase + im * cos_phase)
    return rotated_re, rotated_im


def distance(
    ent_re: Sequence[Sequence[float]],
    ent_im: Sequence[Sequence[float]],
    rel_phase: Sequence[Sequence[float]],
    head_idx: int,
    relation_idx: int,
    tail_idx: int,
) -> float:
    head_re = ent_re[head_idx]
    head_im = ent_im[head_idx]
    tail_re = ent_re[tail_idx]
    tail_im = ent_im[tail_idx]
    phases = rel_phase[relation_idx]

    dist = 0.0
    for i in range(len(phases)):
        cos_phase = math.cos(phases[i])
        sin_phase = math.sin(phases[i])
        rotated_re = head_re[i] * cos_phase - head_im[i] * sin_phase
        rotated_im = head_re[i] * sin_phase + head_im[i] * cos_phase
        diff_re = rotated_re - tail_re[i]
        diff_im = rotated_im - tail_im[i]
        dist += diff_re * diff_re + diff_im * diff_im
    return dist


def sample_negative(entity_count: int, forbidden_index: int, rng: random.Random) -> int:
    if entity_count <= 1:
        return forbidden_index
    sampled = rng.randrange(entity_count)
    while sampled == forbidden_index:
        sampled = rng.randrange(entity_count)
    return sampled


def hinge_loss(margin: float, positive_distance: float, negative_distance: float) -> float:
    value = margin + positive_distance - negative_distance
    return value if value > 0.0 else 0.0


def _encode_triples(
    triples: Iterable[Triple],
    entity_map: Mapping[str, int],
    relation_map: Mapping[str, int],
) -> List[IndexedTriple]:
    encoded: List[IndexedTriple] = []
    for head, relation, tail in triples:
        if head not in entity_map or tail not in entity_map or relation not in relation_map:
            continue
        encoded.append((entity_map[head], relation_map[relation], entity_map[tail]))
    return encoded


def _forward(
    ent_re: Sequence[List[float]],
    ent_im: Sequence[List[float]],
    rel_phase: Sequence[List[float]],
    head_idx: int,
    relation_idx: int,
    tail_idx: int,
) -> dict[str, List[float] | float]:
    head_re = ent_re[head_idx]
    head_im = ent_im[head_idx]
    tail_re = ent_re[tail_idx]
    tail_im = ent_im[tail_idx]
    phases = rel_phase[relation_idx]

    cos_values: List[float] = []
    sin_values: List[float] = []
    diff_re: List[float] = []
    diff_im: List[float] = []
    dist = 0.0

    for i in range(len(phases)):
        cos_phase = math.cos(phases[i])
        sin_phase = math.sin(phases[i])
        cos_values.append(cos_phase)
        sin_values.append(sin_phase)

        rotated_real = head_re[i] * cos_phase - head_im[i] * sin_phase
        rotated_imag = head_re[i] * sin_phase + head_im[i] * cos_phase

        delta_re = rotated_real - tail_re[i]
        delta_im = rotated_imag - tail_im[i]
        diff_re.append(delta_re)
        diff_im.append(delta_im)
        dist += delta_re * delta_re + delta_im * delta_im

    return {
        "dist": dist,
        "cos": cos_values,
        "sin": sin_values,
        "diff_re": diff_re,
        "diff_im": diff_im,
        "head_re": list(head_re),
        "head_im": list(head_im),
    }


def _compute_gradients(cache: dict[str, List[float] | float]) -> tuple[List[float], List[float], List[float], List[float], List[float]]:
    cos_values = cache["cos"]  # type: ignore[assignment]
    sin_values = cache["sin"]  # type: ignore[assignment]
    diff_re = cache["diff_re"]  # type: ignore[assignment]
    diff_im = cache["diff_im"]  # type: ignore[assignment]
    head_re = cache["head_re"]  # type: ignore[assignment]
    head_im = cache["head_im"]  # type: ignore[assignment]

    grad_head_re: List[float] = []
    grad_head_im: List[float] = []
    grad_tail_re: List[float] = []
    grad_tail_im: List[float] = []
    grad_phase: List[float] = []

    for i in range(len(diff_re)):
        grad_rot_re = 2.0 * diff_re[i]
        grad_rot_im = 2.0 * diff_im[i]
        cos_phase = cos_values[i]
        sin_phase = sin_values[i]

        grad_head_re.append(grad_rot_re * cos_phase + grad_rot_im * sin_phase)
        grad_head_im.append(-grad_rot_re * sin_phase + grad_rot_im * cos_phase)
        grad_tail_re.append(-grad_rot_re)
        grad_tail_im.append(-grad_rot_im)
        d_phase = (
            grad_rot_re * (-head_re[i] * sin_phase - head_im[i] * cos_phase)
            + grad_rot_im * (head_re[i] * cos_phase - head_im[i] * sin_phase)
        )
        grad_phase.append(d_phase)

    return grad_head_re, grad_head_im, grad_tail_re, grad_tail_im, grad_phase


def _apply_updates(
    ent_re: List[List[float]],
    ent_im: List[List[float]],
    rel_phase: List[List[float]],
    head_idx: int,
    relation_idx: int,
    tail_idx: int,
    neg_tail_idx: int,
    grad_pos: tuple[List[float], List[float], List[float], List[float], List[float]],
    grad_neg: tuple[List[float], List[float], List[float], List[float], List[float]],
    learning_rate: float,
) -> None:
    pos_head_re, pos_head_im, pos_tail_re, pos_tail_im, pos_phase = grad_pos
    neg_head_re, neg_head_im, neg_tail_re, neg_tail_im, neg_phase = grad_neg

    dim = len(pos_head_re)
    two_pi = 2.0 * math.pi

    for i in range(dim):
        ent_re[head_idx][i] -= learning_rate * (pos_head_re[i] - neg_head_re[i])
        ent_im[head_idx][i] -= learning_rate * (pos_head_im[i] - neg_head_im[i])
        ent_re[tail_idx][i] -= learning_rate * pos_tail_re[i]
        ent_im[tail_idx][i] -= learning_rate * pos_tail_im[i]
        ent_re[neg_tail_idx][i] += learning_rate * neg_tail_re[i]
        ent_im[neg_tail_idx][i] += learning_rate * neg_tail_im[i]
        rel_phase[relation_idx][i] -= learning_rate * (pos_phase[i] - neg_phase[i])
        rel_phase[relation_idx][i] %= two_pi


def _renorm(entity_re: List[float], entity_im: List[float]) -> None:
    norm = math.sqrt(sum((re * re + im * im) for re, im in zip(entity_re, entity_im)))
    if norm > 1.0:
        inv = 1.0 / norm
        for i in range(len(entity_re)):
            entity_re[i] *= inv
            entity_im[i] *= inv


def _evaluate(
    triples: Sequence[IndexedTriple],
    ent_re: Sequence[Sequence[float]],
    ent_im: Sequence[Sequence[float]],
    rel_phase: Sequence[Sequence[float]],
) -> Dict[str, float]:
    if not triples:
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@10": 0.0}

    num_entities = len(ent_re)
    mrr = 0.0
    hits_at_1 = 0.0
    hits_at_10 = 0.0

    for head_idx, relation_idx, tail_idx in triples:
        true_distance = distance(ent_re, ent_im, rel_phase, head_idx, relation_idx, tail_idx)
        rank = 1
        for candidate in range(num_entities):
            if candidate == tail_idx:
                continue
            candidate_distance = distance(ent_re, ent_im, rel_phase, head_idx, relation_idx, candidate)
            if candidate_distance + 1e-9 < true_distance:
                rank += 1
        mrr += 1.0 / rank
        if rank == 1:
            hits_at_1 += 1.0
        if rank <= 10:
            hits_at_10 += 1.0

    total = float(len(triples))
    return {
        "MRR": mrr / total,
        "Hits@1": hits_at_1 / total,
        "Hits@10": hits_at_10 / total,
    }


def train_rotate(
    triples_train: Sequence[Triple],
    triples_val: Sequence[Triple],
    entity_map: Mapping[str, int],
    relation_map: Mapping[str, int],
    *,
    dim: int = 32,
    margin: float = 6.0,
    learning_rate: float = 0.01,
    epochs: int = 50,
    neg_ratio: int = 1,
    seed: int | None = 42,
    log_every: int = 10,
) -> tuple[List[List[float]], List[List[float]], List[List[float]], Dict[str, float]]:
    if neg_ratio < 1:
        raise ValueError("neg_ratio must be >= 1")

    rng = random.Random(seed)
    ent_re, ent_im, rel_phase = initialize_embeddings(
        len(entity_map),
        len(relation_map),
        dim,
        seed,
    )

    train_encoded = _encode_triples(triples_train, entity_map, relation_map)
    val_encoded = _encode_triples(triples_val, entity_map, relation_map)

    if not train_encoded:
        metrics = _evaluate(val_encoded, ent_re, ent_im, rel_phase)
        return ent_re, ent_im, rel_phase, metrics

    num_entities = len(entity_map)

    best_metrics: Dict[str, float] | None = None
    best_epoch = 0
    best_state: tuple[List[List[float]], List[List[float]], List[List[float]]] | None = None

    if val_encoded:
        initial_metrics = _evaluate(val_encoded, ent_re, ent_im, rel_phase)
        best_metrics = initial_metrics
        best_state = (deepcopy(ent_re), deepcopy(ent_im), deepcopy(rel_phase))
        print(
            f"[RotatE] Epoch 0/{epochs} "
            f"MRR={initial_metrics['MRR']:.4f} "
            f"Hits@1={initial_metrics['Hits@1']:.4f} "
            f"Hits@10={initial_metrics['Hits@10']:.4f}"
        )

    for epoch in range(epochs):
        rng.shuffle(train_encoded)
        epoch_loss = 0.0
        epoch_updates = 0
        for head_idx, relation_idx, tail_idx in train_encoded:
            for _ in range(neg_ratio):
                neg_tail_idx = sample_negative(num_entities, tail_idx, rng)
                pos_cache = _forward(ent_re, ent_im, rel_phase, head_idx, relation_idx, tail_idx)
                neg_cache = _forward(ent_re, ent_im, rel_phase, head_idx, relation_idx, neg_tail_idx)
                loss_value = margin + float(pos_cache["dist"]) - float(neg_cache["dist"])
                if loss_value <= 0.0:
                    continue
                grad_pos = _compute_gradients(pos_cache)
                grad_neg = _compute_gradients(neg_cache)
                _apply_updates(
                    ent_re,
                    ent_im,
                    rel_phase,
                    head_idx,
                    relation_idx,
                    tail_idx,
                    neg_tail_idx,
                    grad_pos,
                    grad_neg,
                    learning_rate,
                )
                _renorm(ent_re[head_idx], ent_im[head_idx])
                _renorm(ent_re[tail_idx], ent_im[tail_idx])
                _renorm(ent_re[neg_tail_idx], ent_im[neg_tail_idx])
                epoch_loss += loss_value
                epoch_updates += 1

        if log_every > 0 and ((epoch + 1) % log_every == 0 or epoch == 0):
            avg_loss = epoch_loss / epoch_updates if epoch_updates else 0.0
            if val_encoded:
                interim_metrics = _evaluate(val_encoded, ent_re, ent_im, rel_phase)
                is_best = False
                if best_metrics is None or interim_metrics["MRR"] > best_metrics["MRR"]:
                    best_metrics = interim_metrics
                    best_epoch = epoch + 1
                    best_state = (deepcopy(ent_re), deepcopy(ent_im), deepcopy(rel_phase))
                    is_best = True
                print(
                    f"[RotatE] Epoch {epoch + 1}/{epochs} "
                    f"loss={avg_loss:.4f} "
                    f"MRR={interim_metrics['MRR']:.4f} "
                    f"Hits@1={interim_metrics['Hits@1']:.4f} "
                    f"Hits@10={interim_metrics['Hits@10']:.4f}"
                    + (" *" if is_best else "")
                )
            else:
                print(
                    f"[RotatE] Epoch {epoch + 1}/{epochs} "
                    f"loss={avg_loss:.4f}"
                )

    if best_state is not None and best_metrics is not None:
        ent_re, ent_im, rel_phase = (
            deepcopy(best_state[0]),
            deepcopy(best_state[1]),
            deepcopy(best_state[2]),
        )
        metrics = best_metrics
        print(
            f"[RotatE] Best epoch {best_epoch} "
            f"MRR={metrics['MRR']:.4f} "
            f"Hits@1={metrics['Hits@1']:.4f} "
            f"Hits@10={metrics['Hits@10']:.4f}"
        )
    else:
        metrics = _evaluate(val_encoded, ent_re, ent_im, rel_phase)

    return ent_re, ent_im, rel_phase, metrics


__all__ = [
    "initialize_embeddings",
    "rotate_entity",
    "distance",
    "sample_negative",
    "hinge_loss",
    "train_rotate",
]
