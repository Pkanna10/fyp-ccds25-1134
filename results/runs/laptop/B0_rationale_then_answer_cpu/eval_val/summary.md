n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 49.8 [46.9, 52.7] | 49.8 | 49.8 | ref | ref | ref | ref |
| swapped | none | 30.7 [28.1, 33.4] | 30.7 | 30.7 | +19.1 | 1.3e-24 | +0.853 [+0.723, +0.983] | 60.0 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 41.2% (accuracy-implied bound 86.6%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 43.1 | swapped: +19.2 [+15.0, +23.4] |
| swapped | none | 23.9 | - |
