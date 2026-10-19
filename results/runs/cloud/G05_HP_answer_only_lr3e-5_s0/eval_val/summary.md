n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 59.0 [56.2, 61.9] | 43.8 | 62.5 | ref | ref | ref | ref |
| swapped | none | 32.8 [30.1, 35.5] | 13.0 | 33.6 | +26.3 | 1e-43 | +0.510 [+0.459, +0.563] | 57.3 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 45.5% (accuracy-implied bound 73.9%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 51.9 | swapped: +29.0 [+24.3, +33.5] |
| swapped | none | 22.9 | - |
