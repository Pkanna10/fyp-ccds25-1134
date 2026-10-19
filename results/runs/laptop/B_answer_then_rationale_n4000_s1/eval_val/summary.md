n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 52.0 [49.1, 54.8] | 32.4 | 58.2 | ref | ref | ref | ref |
| swapped | none | 32.8 [30.1, 35.5] | 11.3 | 34.8 | +19.2 | 4.1e-27 | +0.369 [+0.330, +0.412] | 50.8 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 50.6% (accuracy-implied bound 78.3%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 43.1 | swapped: +21.2 [+16.9, +25.3] |
| swapped | none | 22.0 | - |
