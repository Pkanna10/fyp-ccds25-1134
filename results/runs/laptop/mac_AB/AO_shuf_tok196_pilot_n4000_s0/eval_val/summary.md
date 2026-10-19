n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 58.1 [55.2, 60.9] | 41.2 | 61.1 | ref | ref | ref | ref |
| swapped | none | 33.4 [30.8, 36.2] | 14.7 | 33.4 | +24.6 | 2.1e-40 | +0.504 [+0.447, +0.561] | 55.6 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 48.6% (accuracy-implied bound 75.3%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 50.3 | swapped: +27.2 [+22.8, +31.6] |
| swapped | none | 23.1 | - |
