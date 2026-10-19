n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 67.0 [64.2, 69.6] | 54.2 | 67.2 | ref | ref | ref | ref |
| swapped | none | 36.2 [33.5, 39.1] | 18.4 | 35.8 | +30.7 | 2.7e-57 | +0.656 [+0.593, +0.719] | 56.9 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 43.6% (accuracy-implied bound 69.2%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 61.5 | swapped: +37.0 [+32.4, +41.3] |
| swapped | none | 24.6 | - |
