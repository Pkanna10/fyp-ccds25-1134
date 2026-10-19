n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.8 [59.0, 64.6] | 45.9 | 64.1 | ref | ref | ref | ref |
| swapped | none | 33.5 [30.9, 36.3] | 15.3 | 35.3 | +28.3 | 2.3e-53 | +0.532 [+0.480, +0.585] | 54.7 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 47.9% (accuracy-implied bound 72.3%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 52.9 | swapped: +29.9 [+25.5, +34.2] |
| swapped | none | 22.9 | - |
