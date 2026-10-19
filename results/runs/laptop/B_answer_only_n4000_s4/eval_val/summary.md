n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 62.1 [59.2, 64.9] | 46.7 | 65.0 | ref | ref | ref | ref |
| swapped | none | 33.8 [31.1, 36.6] | 15.6 | 35.5 | +28.3 | 4.1e-49 | +0.555 [+0.500, +0.610] | 58.0 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 45.1% (accuracy-implied bound 71.4%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 55.5 | swapped: +33.0 [+28.4, +37.4] |
| swapped | none | 22.5 | - |
