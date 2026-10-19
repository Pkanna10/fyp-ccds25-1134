n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 58.5 [55.6, 61.3] | 41.2 | 61.6 | ref | ref | ref | ref |
| swapped | none | 33.4 [30.8, 36.2] | 14.9 | 34.6 | +25.1 | 3.1e-41 | +0.505 [+0.448, +0.562] | 55.3 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 48.4% (accuracy-implied bound 74.8%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 51.5 | swapped: +29.3 [+24.7, +33.7] |
| swapped | none | 22.3 | - |
