n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 60.1 [57.2, 62.9] | 45.0 | 63.1 | ref | ref | ref | ref |
| swapped | none | 33.4 [30.8, 36.2] | 16.0 | 34.5 | +26.6 | 8.1e-47 | +0.562 [+0.504, +0.621] | 54.3 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 44.2% (accuracy-implied bound 73.3%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 53.0 | swapped: +30.2 [+25.7, +34.5] |
| swapped | none | 22.8 | - |
