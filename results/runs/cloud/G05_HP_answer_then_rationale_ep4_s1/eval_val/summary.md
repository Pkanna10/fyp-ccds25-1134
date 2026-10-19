n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.0 [58.1, 63.7] | 45.1 | 64.2 | ref | ref | ref | ref |
| swapped | none | 34.5 [31.8, 37.3] | 15.0 | 35.6 | +26.5 | 2.3e-43 | +0.605 [+0.534, +0.678] | 57.5 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 46.8% (accuracy-implied bound 72.2%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 52.3 | swapped: +29.3 [+24.7, +33.8] |
| swapped | none | 23.1 | - |
