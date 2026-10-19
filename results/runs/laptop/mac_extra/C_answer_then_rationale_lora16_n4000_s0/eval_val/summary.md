n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 61.3 [58.5, 64.1] | 47.7 | 63.5 | ref | ref | ref | ref |
| swapped | none | 34.8 [32.1, 37.7] | 17.6 | 35.7 | +26.5 | 1.8e-48 | +0.559 [+0.503, +0.616] | 51.0 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 47.0% (accuracy-implied bound 72.9%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 52.7 | swapped: +29.7 [+25.4, +33.9] |
| swapped | none | 23.1 | - |
