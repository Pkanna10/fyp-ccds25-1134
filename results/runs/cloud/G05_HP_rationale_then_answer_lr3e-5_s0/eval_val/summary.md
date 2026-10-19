n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 50.7 [47.8, 53.6] | 50.7 | 50.7 | ref | ref | ref | ref |
| swapped | none | 33.1 [30.4, 35.9] | 33.1 | 33.1 | +17.6 | 8.1e-23 | +0.495 [+0.411, +0.582] | 53.3 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 42.0% (accuracy-implied bound 85.7%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 43.5 | swapped: +20.2 [+15.8, +24.5] |
| swapped | none | 23.4 | - |
