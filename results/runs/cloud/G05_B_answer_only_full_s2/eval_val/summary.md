n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 67.2 [64.4, 69.8] | 54.3 | 67.8 | ref | ref | ref | ref |
| swapped | none | 35.1 [32.4, 37.9] | 18.2 | 34.2 | +32.1 | 2.2e-60 | +0.668 [+0.606, +0.729] | 57.7 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 41.7% (accuracy-implied bound 68.6%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 62.4 | swapped: +36.8 [+32.3, +41.2] |
| swapped | none | 25.5 | - |
