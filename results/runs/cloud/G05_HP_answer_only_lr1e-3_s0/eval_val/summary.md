n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 55.1 [52.2, 58.0] | 39.3 | 54.7 | ref | ref | ref | ref |
| swapped | none | 35.3 [32.6, 38.1] | 20.1 | 36.5 | +19.8 | 1.2e-29 | +0.322 [+0.282, +0.363] | 48.5 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 47.8% (accuracy-implied bound 81.7%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 46.2 | swapped: +23.4 [+19.1, +27.5] |
| swapped | none | 22.8 | - |
