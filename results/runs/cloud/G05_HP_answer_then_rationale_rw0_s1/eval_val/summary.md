n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.1 [58.3, 63.9] | 46.9 | 64.2 | ref | ref | ref | ref |
| swapped | none | 34.1 [31.5, 36.9] | 16.6 | 34.9 | +27.0 | 2.4e-48 | +0.575 [+0.516, +0.633] | 54.5 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 44.5% (accuracy-implied bound 72.2%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 53.7 | swapped: +29.8 [+25.4, +34.1] |
| swapped | none | 23.9 | - |
