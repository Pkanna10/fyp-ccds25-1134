n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.0 [58.1, 63.7] | 45.4 | 64.2 | ref | ref | ref | ref |
| swapped | none | 34.2 [31.5, 37.0] | 15.7 | 35.3 | +26.7 | 5.9e-46 | +0.609 [+0.536, +0.681] | 54.8 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 45.9% (accuracy-implied bound 72.2%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 52.3 | swapped: +29.1 [+24.6, +33.5] |
| swapped | none | 23.2 | - |
