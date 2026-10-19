n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 44.1 [41.3, 47.0] | 15.3 | 54.8 | ref | ref | ref | ref |
| swapped | none | 29.9 [27.3, 32.6] | 5.1 | 33.4 | +14.2 | 1.3e-16 | +0.197 [+0.176, +0.220] | 53.5 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 52.1% (accuracy-implied bound 81.7%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 37.9 | swapped: +15.5 [+11.4, +19.5] |
| swapped | none | 22.4 | - |
