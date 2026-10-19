n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 68.2 [65.5, 70.8] | 54.5 | 69.5 | ref | ref | ref | ref |
| swapped | none | 34.8 [32.1, 37.7] | 18.4 | 34.9 | +33.4 | 3.9e-65 | +0.666 [+0.604, +0.729] | 58.8 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 43.1% (accuracy-implied bound 66.9%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 62.4 | swapped: +37.8 [+33.2, +42.2] |
| swapped | none | 24.6 | - |
