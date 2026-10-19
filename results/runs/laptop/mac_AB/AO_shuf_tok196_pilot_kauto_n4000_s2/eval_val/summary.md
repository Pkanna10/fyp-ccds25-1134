n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 58.1 [55.2, 60.9] | 41.7 | 61.4 | ref | ref | ref | ref |
| swapped | none | 34.1 [31.4, 36.9] | 14.7 | 35.1 | +24.0 | 1.8e-38 | +0.488 [+0.434, +0.545] | 53.4 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 49.2% (accuracy-implied bound 75.0%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 50.1 | swapped: +27.1 [+22.5, +31.5] |
| swapped | none | 23.1 | - |
