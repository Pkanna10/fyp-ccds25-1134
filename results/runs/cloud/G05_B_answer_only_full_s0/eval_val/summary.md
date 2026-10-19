n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 67.6 [64.8, 70.2] | 55.1 | 68.9 | ref | ref | ref | ref |
| swapped | none | 35.3 [32.6, 38.1] | 18.8 | 35.8 | +32.3 | 1.1e-62 | +0.665 [+0.602, +0.728] | 56.4 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 42.8% (accuracy-implied bound 67.5%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 62.0 | swapped: +38.0 [+33.5, +42.4] |
| swapped | none | 23.9 | - |
