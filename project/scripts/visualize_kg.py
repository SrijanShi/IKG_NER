from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRIPLES_PATH = PROJECT_ROOT / "data" / "Cricket_triples.csv"
OUTPUT_DOT_PATH = PROJECT_ROOT / "data" / "Cricket_sample.dot"

MAX_TRIPLES = 300

COLOR_MAP = {
    "post": "#1f77b4",
    "user": "#ff7f0e",
    "hashtag": "#2ca02c",
    "keyword": "#9467bd",
    "event": "#d62728",
    "sentiment": "#8c564b",
    "time": "#e377c2",
    "number": "#7f7f7f",
    "default": "#17becf",
}


def infer_node_type(node: str) -> str:
    if node.startswith("POST_"):
        return "post"
    if node.startswith("user_"):
        return "user"
    if node.startswith("#"):
        return "hashtag"
    if node in {"sent_positive", "sent_negative", "sent_neutral"}:
        return "sentiment"
    if node.isdigit() or any(ch.isdigit() for ch in node) and node.replace(":", "").isdigit():
        return "number"
    if node in {
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    }:
        return "time"
    if node.lower() in {"world", "cup", "cricket", "2025"}:
        return "event"
    if " " in node:
        return "keyword"
    return "default"


def load_top_triples(path: Path, limit: int) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for idx, row in enumerate(reader, start=1):
            head = row.get("head")
            relation = row.get("relation")
            tail = row.get("tail")
            if head and relation and tail:
                triples.append((head, relation, tail))
            if idx >= limit:
                break
    return triples


def select_top_nodes(triples: Iterable[tuple[str, str, str]], max_nodes: int = 120) -> set[str]:
    counts = Counter()
    for head, _, tail in triples:
        counts[head] += 1
        counts[tail] += 1
    most_common = {node for node, _ in counts.most_common(max_nodes)}
    return most_common


def write_dot(triples: list[tuple[str, str, str]], nodes_to_keep: set[str], path: Path) -> None:
    lines: list[str] = []
    lines.append("digraph CricketKG {")
    lines.append("  rankdir=LR;")
    lines.append("  node [style=filled, fontname=Helvetica];")

    for node in nodes_to_keep:
        node_type = infer_node_type(node)
        color = COLOR_MAP.get(node_type, COLOR_MAP["default"])
        label = node.replace("\"", "\"")
        lines.append(f'  "{node}" [fillcolor="{color}", label="{label}"];')

    for head, relation, tail in triples:
        if head in nodes_to_keep and tail in nodes_to_keep:
            safe_relation = relation.replace("\"", "'")
            lines.append(f'  "{head}" -> "{tail}" [label="{safe_relation}"];')

    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not TRIPLES_PATH.exists():
        print("Triple file not found. Run build_kg.py first.")
        return

    triples = load_top_triples(TRIPLES_PATH, MAX_TRIPLES)
    if not triples:
        print("No triples available to visualize.")
        return

    nodes_to_keep = select_top_nodes(triples)
    write_dot(triples, nodes_to_keep, OUTPUT_DOT_PATH)

    print("Visualization data generated:")
    print(f" - DOT file: {OUTPUT_DOT_PATH}")
    print("Render with Graphviz, e.g.:")
    print(f"   dot -Tpng {OUTPUT_DOT_PATH.name} -o Cricket_sample.png")


if __name__ == "__main__":
    main()
