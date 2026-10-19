n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.8 [59.0, 64.6] | 47.4 | 64.5 | ref | ref | ref | ref |
| swapped | none | 33.1 [30.4, 35.9] | 15.6 | 34.8 | +28.7 | 1.9e-53 | +0.563 [+0.508, +0.620] | 55.4 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 43.8% (accuracy-implied bound 72.0%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 55.6 | swapped: +33.1 [+28.6, +37.5] |
| swapped | none | 22.5 | - |
