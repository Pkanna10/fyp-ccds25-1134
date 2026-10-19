n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 64.7 [61.9, 67.4] | 52.6 | 65.9 | ref | ref | ref | ref |
| swapped | none | 35.7 [33.0, 38.5] | 19.8 | 35.8 | +29.0 | 1.3e-53 | +0.601 [+0.544, +0.659] | 53.4 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 43.8% (accuracy-implied bound 70.5%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 58.1 | swapped: +32.3 [+27.9, +36.5] |
| swapped | none | 25.8 | - |
