n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 54.4 [51.5, 57.3] | 37.6 | 57.2 | ref | ref | ref | ref |
| swapped | none | 33.2 [30.5, 36.0] | 15.0 | 34.9 | +21.2 | 1.5e-32 | +0.393 [+0.344, +0.443] | 50.8 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 53.4% (accuracy-implied bound 79.2%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 44.4 | swapped: +23.8 [+19.5, +28.0] |
| swapped | none | 20.6 | - |
