n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.7 [58.8, 64.4] | 47.5 | 64.8 | ref | ref | ref | ref |
| swapped | none | 33.1 [30.4, 35.9] | 15.8 | 34.3 | +28.6 | 9.3e-53 | +0.564 [+0.508, +0.622] | 55.5 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 43.6% (accuracy-implied bound 71.6%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 55.5 | swapped: +33.1 [+28.6, +37.4] |
| swapped | none | 22.4 | - |
