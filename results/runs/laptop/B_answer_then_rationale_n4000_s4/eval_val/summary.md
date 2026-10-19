n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 50.7 [47.8, 53.6] | 31.7 | 57.4 | ref | ref | ref | ref |
| swapped | none | 31.9 [29.2, 34.6] | 10.6 | 35.1 | +18.9 | 1.6e-26 | +0.349 [+0.313, +0.386] | 52.1 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 51.3% (accuracy-implied bound 79.0%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 41.9 | swapped: +21.3 [+17.0, +25.5] |
| swapped | none | 20.6 | - |
