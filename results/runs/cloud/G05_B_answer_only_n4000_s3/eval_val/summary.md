n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.4 [58.5, 64.2] | 47.1 | 64.3 | ref | ref | ref | ref |
| swapped | none | 33.2 [30.5, 36.0] | 16.8 | 34.5 | +28.2 | 2.2e-51 | +0.553 [+0.498, +0.611] | 56.1 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 45.7% (accuracy-implied bound 72.1%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 54.1 | swapped: +32.6 [+28.0, +37.0] |
| swapped | none | 21.6 | - |
