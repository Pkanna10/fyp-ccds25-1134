n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 52.2 [49.3, 55.1] | 32.1 | 59.0 | ref | ref | ref | ref |
| swapped | none | 32.3 [29.7, 35.1] | 12.1 | 35.2 | +19.9 | 1.9e-30 | +0.345 [+0.308, +0.385] | 50.2 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 50.7% (accuracy-implied bound 77.5%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 43.3 | swapped: +22.1 [+18.0, +26.1] |
| swapped | none | 21.2 | - |
