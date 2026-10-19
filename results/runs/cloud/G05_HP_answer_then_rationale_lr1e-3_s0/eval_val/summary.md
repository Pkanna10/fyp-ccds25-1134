n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 56.2 [53.4, 59.1] | 38.0 | 59.3 | ref | ref | ref | ref |
| swapped | none | 33.6 [30.9, 36.4] | 15.8 | 35.1 | +22.6 | 7.1e-40 | +0.392 [+0.350, +0.435] | 46.6 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 50.8% (accuracy-implied bound 77.1%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 45.7 | swapped: +25.1 [+21.0, +29.1] |
| swapped | none | 20.6 | - |
