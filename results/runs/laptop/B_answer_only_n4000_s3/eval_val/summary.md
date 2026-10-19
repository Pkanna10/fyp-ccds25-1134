n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.0 [58.1, 63.7] | 47.1 | 63.6 | ref | ref | ref | ref |
| swapped | none | 32.9 [30.3, 35.7] | 16.4 | 34.8 | +28.0 | 1.1e-51 | +0.557 [+0.501, +0.615] | 55.0 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 45.3% (accuracy-implied bound 72.8%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 53.3 | swapped: +32.1 [+27.7, +36.5] |
| swapped | none | 21.2 | - |
