n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.0 [58.1, 63.7] | 47.2 | 63.0 | ref | ref | ref | ref |
| swapped | none | 33.6 [30.9, 36.4] | 15.4 | 36.0 | +27.3 | 5.2e-45 | +0.563 [+0.506, +0.622] | 57.8 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 44.2% (accuracy-implied bound 73.4%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 54.3 | swapped: +31.3 [+26.6, +35.9] |
| swapped | none | 22.9 | - |
