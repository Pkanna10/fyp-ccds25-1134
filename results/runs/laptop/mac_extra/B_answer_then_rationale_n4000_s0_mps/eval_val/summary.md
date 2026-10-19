n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 51.6 [48.7, 54.5] | 31.2 | 57.7 | ref | ref | ref | ref |
| swapped | none | 31.2 [28.6, 33.9] | 11.3 | 34.6 | +20.4 | 1.6e-30 | +0.375 [+0.336, +0.417] | 52.1 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 50.7% (accuracy-implied bound 78.7%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 43.3 | swapped: +23.4 [+19.0, +27.6] |
| swapped | none | 19.9 | - |
