n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.7 [58.9, 64.5] | 45.5 | 64.5 | ref | ref | ref | ref |
| swapped | none | 34.5 [31.8, 37.3] | 15.2 | 35.9 | +27.2 | 1e-49 | +0.530 [+0.476, +0.586] | 54.2 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 47.7% (accuracy-implied bound 71.9%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 52.9 | swapped: +30.5 [+26.1, +34.8] |
| swapped | none | 22.4 | - |
