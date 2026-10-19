n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 52.1 [49.2, 54.9] | 32.4 | 58.1 | ref | ref | ref | ref |
| swapped | none | 32.7 [30.0, 35.4] | 11.3 | 34.5 | +19.4 | 2.7e-28 | +0.367 [+0.328, +0.409] | 50.2 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 50.5% (accuracy-implied bound 78.3%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 42.7 | swapped: +20.9 [+16.6, +25.0] |
| swapped | none | 21.8 | - |
