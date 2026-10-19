n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.3 [58.5, 64.1] | 47.2 | 64.6 | ref | ref | ref | ref |
| swapped | none | 32.9 [30.3, 35.7] | 15.8 | 34.7 | +28.4 | 2.6e-52 | +0.566 [+0.510, +0.623] | 54.9 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 43.8% (accuracy-implied bound 71.8%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 55.1 | swapped: +33.0 [+28.4, +37.4] |
| swapped | none | 22.1 | - |
