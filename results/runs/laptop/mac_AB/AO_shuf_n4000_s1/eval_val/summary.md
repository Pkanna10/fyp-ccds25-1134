n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.6 [58.7, 64.3] | 47.8 | 64.3 | ref | ref | ref | ref |
| swapped | none | 33.5 [30.9, 36.3] | 15.4 | 34.8 | +28.0 | 8.6e-50 | +0.590 [+0.530, +0.651] | 56.6 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 43.5% (accuracy-implied bound 72.1%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 55.1 | swapped: +30.9 [+26.4, +35.3] |
| swapped | none | 24.2 | - |
