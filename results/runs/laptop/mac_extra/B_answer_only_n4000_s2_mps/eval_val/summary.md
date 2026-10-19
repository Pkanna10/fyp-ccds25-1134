n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 60.9 [58.0, 63.7] | 47.4 | 63.8 | ref | ref | ref | ref |
| swapped | none | 33.1 [30.4, 35.9] | 14.9 | 35.5 | +27.8 | 5.1e-47 | +0.563 [+0.505, +0.622] | 57.9 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 44.1% (accuracy-implied bound 72.6%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 54.1 | swapped: +31.5 [+26.9, +36.0] |
| swapped | none | 22.7 | - |
