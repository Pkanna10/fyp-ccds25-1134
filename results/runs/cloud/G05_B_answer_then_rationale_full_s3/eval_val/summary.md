n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.3 [58.5, 64.1] | 45.6 | 63.8 | ref | ref | ref | ref |
| swapped | none | 34.4 [31.7, 37.2] | 15.2 | 34.9 | +26.9 | 8.4e-47 | +0.537 [+0.484, +0.593] | 54.7 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 49.0% (accuracy-implied bound 72.7%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 52.6 | swapped: +29.9 [+25.5, +34.3] |
| swapped | none | 22.7 | - |
