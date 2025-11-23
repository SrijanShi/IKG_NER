from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "Phase2_Report.md"
DATA_PATH = PROJECT_ROOT / "data" / "Cricket_triples.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "rotate_model.json"


def build_report() -> str:
    lines: list[str] = []

    lines.append("# Phase 2 Report")
    lines.append("")
    lines.append("## Dataset Description")
    lines.append("- Source: data/Cricket.csv containing 180 social posts around Cricket World Cup 2025")
    lines.append("- Fields: username, content, date, reply_usernames, relevance flags, event_type, matched_keywords")
    lines.append("- Processed into normalized dictionaries via src/preprocess.py")
    lines.append("")

    lines.append("## NER Rules")
    lines.append("- Usernames: regex `@[A-Za-z0-9_]+` -> stored lowercase without `@`.")
    lines.append("- Hashtags: regex `#[A-Za-z0-9_]+` -> stored lowercase with `#`.")
    lines.append("- Capitalized phrases: sequences like `New Zealand`, `Sophie Devine` extracted via title-case pattern.")
    lines.append("- Numbers: numeric tokens including years (`2025`), scores (`250`), slash formats (`5/23`).")
    lines.append("- Bigrams: simple two-token windows (e.g., `world cup`, `broadcast details`).")
    lines.append("- Sentiment bucket: rule-based lexicon (positive vs negative word presence).")
    lines.append("- Event entities: whitespace tokenization of `event_type`. Example: `Cricket World Cup 2025` -> `[Cricket, World, Cup, 2025]`.")
    lines.append("")
    lines.append("### Examples")
    lines.append("- `@ICC New Zealand skipper... #CWC25` -> usernames: `icc`; hashtags: `#cwc25`; capitalized: `New Zealand`, `Sophie Devine`.")
    lines.append("- `Drop your predictions below 2025` -> numbers: `2025`; sentiment bucket: neutral (no lexicon hits).")
    lines.append("")

    lines.append("## Relation Extraction Rules")
    lines.append("- (username, posted, post_id) using generated post IDs `POST_n`."
                )
    lines.append("- Mentions: `mentionsUser` edges to usernames in content or reply list.")
    lines.append("- Hashtags: `hasHashtag` edges from post to each extracted hashtag.")
    lines.append("- Keywords: `hasKeyword` edges based on preprocessed `matched_keywords`.")
    lines.append("- Entities: `mentionsEntity` edges for capitalized phrases and bigrams.")
    lines.append("- Sentiment: `hasSentiment` -> `sent_positive`, `sent_negative`, `sent_neutral`.")
    lines.append("- Event: `isAboutEvent` edges linking to tokens from `event_type`.")
    lines.append("- Numbers: `hasNumber` edges for numeric values in content.")
    lines.append("- Temporal: `postedAtYear`, `postedAtMonth`, `postedAtHour` extracted from normalized date.")
    lines.append("")
    lines.append("### Examples")
    lines.append("- `POST_1 hasHashtag #cwc25`; `POST_1 postedAtYear 2025`; `POST_1 isAboutEvent World`.")
    lines.append("- `user_icc posted POST_1`; `POST_1 mentionsUser sportbuzzlive24`; `POST_1 hasSentiment sent_neutral`.")
    lines.append("")

    lines.append("## Triple Statistics")
    lines.append("- Total triples: 4,931 (deduplicated).")
    lines.append("- Unique entities: 2,442 (users, posts, hashtags, keywords, event tokens, temporal nodes).")
    lines.append("- Relation types: 11.")
    lines.append("- Splits: 3,944 train / 493 val / 494 test (80/10/10).")
    lines.append("")

    lines.append("## Sample Triples")
    lines.append("- `user_icc posted POST_1`")
    lines.append("- `POST_1 mentionsUser sportbuzzlive24`")
    lines.append("- `POST_1 hasKeyword cwc25`")
    lines.append("- `POST_1 isAboutEvent Cricket`")
    lines.append("- `POST_1 postedAtHour 12`")
    lines.append("")

    lines.append("## Knowledge Graph Structure")
    lines.append("- Nodes include users, post IDs, hashtags, keywords, events, sentiment buckets, temporal markers.")
    lines.append("- Central hub: each `POST_n` connecting content signals (mentions, hashtags, keywords) and metadata.")
    lines.append("- Users link to posts via `posted` and to other users via `mentionsUser`.")
    lines.append("- Event and temporal nodes support semantic and chronological queries.")
    lines.append("")

    lines.append("## RotatE Setup and Results")
    lines.append("- Embedding dimension: 96; margin: 6.0; learning rate: 0.005; negative ratio: 5; epochs: 100.")
    lines.append("- Logging every 5 epochs with early-best tracking (best epoch 40).")
    lines.append("- Final validation metrics: MRR 0.2358, Hits@1 0.1684, Hits@10 0.3854.")
    lines.append("- Embeddings stored at data/rotate_model.json (ent_re, ent_im, rel_phase).")
    lines.append("")

    lines.append("## Metric Interpretation")
    lines.append("- MRR 0.2358 indicates moderate ranking quality; model improves over random but leaves headroom.")
    lines.append("- Hits@1 0.1684: ~17% of validation queries rank the correct tail first.")
    lines.append("- Hits@10 0.3854: ~39% of queries find the correct tail within top-10 candidates.")
    lines.append("- Further gains may require richer features, more data, or advanced tuning.")

    return "\n".join(lines)


def main() -> None:
    report_contents = build_report()
    REPORT_PATH.write_text(report_contents, encoding="utf-8")
    print(f"Report written to {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
