# Phase 2 Report

## Dataset Description
- Source: data/Cricket.csv containing 180 social posts around Cricket World Cup 2025
- Fields: username, content, date, reply_usernames, relevance flags, event_type, matched_keywords
- Processed into normalized dictionaries via src/preprocess.py

## NER Rules
- Usernames: regex `@[A-Za-z0-9_]+` -> stored lowercase without `@`.
- Hashtags: regex `#[A-Za-z0-9_]+` -> stored lowercase with `#`.
- Capitalized phrases: sequences like `New Zealand`, `Sophie Devine` extracted via title-case pattern.
- Numbers: numeric tokens including years (`2025`), scores (`250`), slash formats (`5/23`).
- Bigrams: simple two-token windows (e.g., `world cup`, `broadcast details`).
- Sentiment bucket: rule-based lexicon (positive vs negative word presence).
- Event entities: whitespace tokenization of `event_type`. Example: `Cricket World Cup 2025` -> `[Cricket, World, Cup, 2025]`.

### Examples
- `@ICC New Zealand skipper... #CWC25` -> usernames: `icc`; hashtags: `#cwc25`; capitalized: `New Zealand`, `Sophie Devine`.
- `Drop your predictions below 2025` -> numbers: `2025`; sentiment bucket: neutral (no lexicon hits).

## Relation Extraction Rules
- (username, posted, post_id) using generated post IDs `POST_n`.
- Mentions: `mentionsUser` edges to usernames in content or reply list.
- Hashtags: `hasHashtag` edges from post to each extracted hashtag.
- Keywords: `hasKeyword` edges based on preprocessed `matched_keywords`.
- Entities: `mentionsEntity` edges for capitalized phrases and bigrams.
- Sentiment: `hasSentiment` -> `sent_positive`, `sent_negative`, `sent_neutral`.
- Event: `isAboutEvent` edges linking to tokens from `event_type`.
- Numbers: `hasNumber` edges for numeric values in content.
- Temporal: `postedAtYear`, `postedAtMonth`, `postedAtHour` extracted from normalized date.

### Examples
- `POST_1 hasHashtag #cwc25`; `POST_1 postedAtYear 2025`; `POST_1 isAboutEvent World`.
- `user_icc posted POST_1`; `POST_1 mentionsUser sportbuzzlive24`; `POST_1 hasSentiment sent_neutral`.

## Triple Statistics
- Total triples: 4,931 (deduplicated).
- Unique entities: 2,442 (users, posts, hashtags, keywords, event tokens, temporal nodes).
- Relation types: 11.
- Splits: 3,944 train / 493 val / 494 test (80/10/10).

## Sample Triples
- `user_icc posted POST_1`
- `POST_1 mentionsUser sportbuzzlive24`
- `POST_1 hasKeyword cwc25`
- `POST_1 isAboutEvent Cricket`
- `POST_1 postedAtHour 12`

## Knowledge Graph Structure
- Nodes include users, post IDs, hashtags, keywords, events, sentiment buckets, temporal markers.
- Central hub: each `POST_n` connecting content signals (mentions, hashtags, keywords) and metadata.
- Users link to posts via `posted` and to other users via `mentionsUser`.
- Event and temporal nodes support semantic and chronological queries.

## RotatE Setup and Results
- Embedding dimension: 96; margin: 6.0; learning rate: 0.005; negative ratio: 5; epochs: 100.
- Logging every 5 epochs with early-best tracking (best epoch 40).
- Final validation metrics: MRR 0.2358, Hits@1 0.1684, Hits@10 0.3854.
- Embeddings stored at data/rotate_model.json (ent_re, ent_im, rel_phase).

## Metric Interpretation
- MRR 0.2358 indicates moderate ranking quality; model improves over random but leaves headroom.
- Hits@1 0.1684: ~17% of validation queries rank the correct tail first.
- Hits@10 0.3854: ~39% of queries find the correct tail within top-10 candidates.
- Further gains may require richer features, more data, or advanced tuning.