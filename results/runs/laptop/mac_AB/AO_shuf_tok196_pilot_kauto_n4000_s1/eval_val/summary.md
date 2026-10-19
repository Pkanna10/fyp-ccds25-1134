n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 56.9 [54.0, 59.7] | 41.6 | 61.1 | ref | ref | ref | ref |
| swapped | none | 33.1 [30.4, 35.9] | 14.8 | 35.4 | +23.8 | 7.3e-38 | +0.503 [+0.448, +0.560] | 54.5 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 48.2% (accuracy-implied bound 75.3%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 48.9 | swapped: +27.7 [+23.2, +32.3] |
| swapped | none | 21.2 | - |
