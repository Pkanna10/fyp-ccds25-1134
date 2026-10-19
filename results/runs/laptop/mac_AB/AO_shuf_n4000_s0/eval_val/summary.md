n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.5 [58.6, 64.3] | 45.8 | 64.5 | ref | ref | ref | ref |
| swapped | none | 33.7 [31.0, 36.5] | 14.8 | 35.3 | +27.8 | 1.2e-47 | +0.570 [+0.512, +0.628] | 57.6 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 45.1% (accuracy-implied bound 71.9%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 55.1 | swapped: +31.9 [+27.2, +36.3] |
| swapped | none | 23.2 | - |
