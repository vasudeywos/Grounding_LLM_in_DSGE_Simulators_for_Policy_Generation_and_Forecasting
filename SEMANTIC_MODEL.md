# Temporal Semantic Shock Encoder

This module trains an encoder-only BERT model to convert time-indexed economic discourse into a reusable latent shock embedding. It does not yet connect the embedding to Qwen or Snowdrop.

## Input

Training data uses one JSON object per time bucket and source:

```json
{"timestamp":"2024-01-02","source":"news","texts":["Document one","Document two"],"shock_targets":{"inflation":0.45,"monetary":0.27},"semantic_volatility":0.62}
```

Fields:

- `timestamp` is an ISO-8601 time identifying the bucket.
- `source` identifies an outlet, platform, community, or other discourse stream.
- `texts` contains the documents observed within that bucket.
- `shock_targets` contains any available continuous economic shock labels. Missing shock dimensions are masked from supervised loss.
- `semantic_volatility` is an optional measured shift target. It is also masked when unavailable.

Shock targets should be economically identified and standardized using only the training period. Examples include high-frequency monetary surprises, structural VAR residuals, forecast revisions, or calibrated shock estimates. Raw asset returns should not automatically be treated as structural shocks.

## Model

For each time bucket, BERT encodes every document. The model averages the configured final hidden layers, pools tokens into document states, and pools documents into a time-specific semantic state. Consecutive states produce:

- A directional semantic-shift vector.
- Cosine semantic distance.
- A learned temporal semantic state.
- A compact latent shock embedding `z_t`.
- Named continuous shock estimates.
- A semantic-volatility estimate.

The training objective combines masked shock regression, masked volatility regression, and a temporal contrastive objective that predicts the next discourse state from the previous one. The chronological split prevents future discourse from leaking into training.

Economic shock labels provide identification. Without them, the temporal objective can still learn when discourse changes, but the latent dimensions cannot reliably be interpreted as monetary, demand, supply, technology, expectations, or inflation shocks.

## Training

Copy or replace the example data:

```bash
cp data/economic_discourse.example.jsonl data/economic_discourse.jsonl
semantic-train --config configs/semantic_bert.yaml
```

The best checkpoint contains:

- `bert_encoder/` with the fine-tuned BERT layers.
- `semantic_heads.pt` with temporal, shock, and volatility heads.
- `semantic_config.json` with the architecture and shock ordering.
- `tokenizer/` with the matching tokenizer.

## Encoding

```bash
semantic-encode --checkpoint outputs/semantic_bert/best --input data/economic_discourse.jsonl --output outputs/semantic_embeddings.jsonl
```

Each output row contains the reusable `shock_embedding`, directional `semantic_shift`, semantic volatility, cosine shift, and named shock estimates. The fine-tuned final BERT layers can later be connected to the policy model as a separate latent feature source.

