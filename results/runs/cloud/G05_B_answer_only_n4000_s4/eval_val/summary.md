n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 62.2 [59.3, 64.9] | 46.8 | 64.6 | ref | ref | ref | ref |
| swapped | none | 33.7 [31.0, 36.5] | 15.6 | 35.7 | +28.5 | 3.4e-50 | +0.551 [+0.496, +0.605] | 57.6 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 44.8% (accuracy-implied bound 71.8%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 55.9 | swapped: +33.9 [+29.4, +38.3] |
| swapped | none | 22.0 | - |
