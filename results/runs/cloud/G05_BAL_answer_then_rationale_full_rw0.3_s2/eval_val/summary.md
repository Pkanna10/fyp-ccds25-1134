n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 65.9 [63.1, 68.5] | 54.3 | 67.3 | ref | ref | ref | ref |
| swapped | none | 34.1 [31.5, 36.9] | 19.8 | 33.6 | +31.7 | 2.9e-60 | +0.669 [+0.607, +0.731] | 58.8 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 43.1% (accuracy-implied bound 69.1%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 60.2 | swapped: +36.0 [+31.5, +40.4] |
| swapped | none | 24.2 | - |
