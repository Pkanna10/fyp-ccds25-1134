n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 55.2 [52.3, 58.1] | 55.2 | 55.2 | ref | ref | ref | ref |
| swapped | none | 32.3 [29.7, 35.1] | 32.3 | 32.3 | +22.9 | 3.1e-36 | +0.770 [+0.650, +0.892] | 54.1 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 40.2% (accuracy-implied bound 81.2%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 49.2 | swapped: +25.3 [+20.9, +29.7] |
| swapped | none | 23.9 | - |
