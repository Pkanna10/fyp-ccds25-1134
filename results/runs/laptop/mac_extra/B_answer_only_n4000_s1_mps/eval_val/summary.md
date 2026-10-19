n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.0 [58.1, 63.7] | 47.0 | 63.9 | ref | ref | ref | ref |
| swapped | none | 33.2 [30.5, 36.0] | 15.2 | 34.5 | +27.8 | 3.8e-49 | +0.564 [+0.506, +0.624] | 55.2 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 43.5% (accuracy-implied bound 72.5%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 54.9 | swapped: +32.0 [+27.3, +36.4] |
| swapped | none | 22.9 | - |
