n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.8 [59.0, 64.6] | 45.2 | 63.7 | ref | ref | ref | ref |
| swapped | none | 32.9 [30.3, 35.7] | 14.9 | 35.2 | +28.9 | 1.2e-53 | +0.526 [+0.474, +0.580] | 54.6 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 48.5% (accuracy-implied bound 72.8%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 52.6 | swapped: +31.3 [+26.9, +35.6] |
| swapped | none | 21.3 | - |
