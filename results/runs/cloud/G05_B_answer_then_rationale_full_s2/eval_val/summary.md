n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.3 [58.5, 64.1] | 45.6 | 63.8 | ref | ref | ref | ref |
| swapped | none | 34.0 [31.3, 36.8] | 15.4 | 34.9 | +27.3 | 1.2e-47 | +0.538 [+0.483, +0.594] | 54.9 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 47.9% (accuracy-implied bound 72.6%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 52.5 | swapped: +30.6 [+26.2, +34.9] |
| swapped | none | 21.8 | - |
