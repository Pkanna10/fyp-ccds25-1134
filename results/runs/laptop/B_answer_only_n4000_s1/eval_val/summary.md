n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 60.9 [58.0, 63.7] | 47.4 | 63.8 | ref | ref | ref | ref |
| swapped | none | 33.4 [30.8, 36.2] | 15.3 | 34.5 | +27.4 | 2e-47 | +0.565 [+0.507, +0.625] | 55.7 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 43.7% (accuracy-implied bound 72.6%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 54.8 | swapped: +31.9 [+27.2, +36.4] |
| swapped | none | 22.9 | - |
