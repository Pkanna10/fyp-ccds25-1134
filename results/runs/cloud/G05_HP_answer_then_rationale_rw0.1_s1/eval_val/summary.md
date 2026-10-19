n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 60.5 [57.7, 63.3] | 47.2 | 63.4 | ref | ref | ref | ref |
| swapped | none | 34.1 [31.4, 36.9] | 16.7 | 34.3 | +26.5 | 4.7e-47 | +0.571 [+0.513, +0.629] | 54.6 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 44.7% (accuracy-implied bound 73.0%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 53.3 | swapped: +29.9 [+25.5, +34.3] |
| swapped | none | 23.4 | - |
