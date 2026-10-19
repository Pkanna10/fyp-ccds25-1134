n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 67.0 [64.2, 69.6] | 56.0 | 68.3 | ref | ref | ref | ref |
| swapped | none | 34.7 [32.0, 37.5] | 18.9 | 34.9 | +32.3 | 2.1e-64 | +0.673 [+0.609, +0.739] | 57.2 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 43.1% (accuracy-implied bound 68.1%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 61.0 | swapped: +36.8 [+32.3, +41.2] |
| swapped | none | 24.2 | - |
