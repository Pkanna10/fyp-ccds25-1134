n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 63.9 [61.1, 66.7] | 52.1 | 65.2 | ref | ref | ref | ref |
| swapped | none | 34.7 [32.0, 37.5] | 18.7 | 35.5 | +29.3 | 3.7e-48 | +0.972 [+0.835, +1.107] | 61.7 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 41.0% (accuracy-implied bound 71.2%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 59.5 | swapped: +33.4 [+28.6, +38.0] |
| swapped | none | 26.1 | - |
