# N0 DATA_AUDIT

- gate_pass: **True**
- n_matched_pairs: 51 (discover floor 30, development floor 20)
- n_benign: 70 (floor 30)

## Counts vs targets

| split | n | target | floor | meets_floor |
|---|---:|---:|---:|---|
| discover | 30 | 50 | 30 | True |
| development | 20 | 30 | 20 | True |
| confirm | 1 | 30 | 25 | False |
| attack_test | 40 | 40 | 30 | True |

## RELATED_SAFE contamination

Gate uses matched-pair SAFE sides (must be <5% refuse/theme-denial). Legacy RELATED_SAFE slots are diagnostic only.

- n_legacy_related_safe: 67
- n_core_safe_answer: 66
- legacy_refuse_or_deny_frac: 0.0
- pair_refuse_or_deny_frac: 0.0 (gate < 0.05)
- theme_denial_frac (legacy): 0.0

## Duplicates

- duplicate_pair_id: 0 (0.0)
- duplicate_query_image: 0 (0.0)

## Split intersections

Query IDs must be disjoint. Image IDs may overlap because only four P0 carriers (c01–c04) exist; that is recorded, not a query leak.

- `query_discover_x_development`: []
- `query_discover_x_confirm`: []
- `query_discover_x_attack_test`: []
- `query_development_x_confirm`: []
- `query_development_x_attack_test`: []
- `query_confirm_x_attack_test`: []
- `sealed_confirm_used`: []
- `sealed_attack_used_in_pairs`: []

Image-ID overlaps (shared carriers, expected):
- `image_discover_x_development`: ['c01', 'c02', 'c03', 'c04']
- `image_discover_x_confirm`: ['c04']
- `image_discover_x_attack_test`: []
- `image_development_x_confirm`: ['c04']
- `image_development_x_attack_test`: []
- `image_confirm_x_attack_test`: []

## Relabel (single annotator, interval rule re-label of 20%)

- annotators: 1 (interval_rule_relabel)
- n_sample: 10
- agreement: {'safety': 1.0, 'response_mode': 1.0, 'grounding': 1.0, 'quality': 1.0}
- dispute rule: rule-based four_axes is source of truth; keep after-label on disagreement

## Sealed old-line IDs

- h83–h90 unused (old confirm, never loaded into N0 pairs)
- h91–h130 reserved as attack-test catalog IDs only; generations were not read

## Gate reasons

- (none)

N0 passed. N1 representation scan is allowed on discover only.

## Notes

- Confirm has 1 pair (floor 25 / target 30). That does **not** fail N0, but N3 cannot run until confirm is filled.
- Discover 30 and development 20 meet the N0 hard floors. Targets 50/30 are not required to open N1.
- Attack-test lists catalog IDs h91–h130 only; P0 generations were not read.
- `traces.json` remains the frozen P0-S snapshot. N0 working traces are `outputs/p0_qwen/full/native/traces_n0.json`.
- This ticket did not run representation scan, patching, or attack.
