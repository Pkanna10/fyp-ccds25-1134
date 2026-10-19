n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 60.9 [58.0, 63.7] | 45.3 | 62.9 | ref | ref | ref | ref |
| swapped | none | 34.8 [32.1, 37.6] | 16.2 | 34.0 | +26.1 | 1.7e-44 | +0.580 [+0.521, +0.640] | 53.7 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 45.5% (accuracy-implied bound 73.5%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 53.7 | swapped: +30.2 [+25.7, +34.6] |
| swapped | none | 23.5 | - |
