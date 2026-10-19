n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 62.1 [59.2, 64.9] | 47.6 | 65.0 | ref | ref | ref | ref |
| swapped | none | 35.6 [32.9, 38.5] | 17.1 | 36.3 | +26.5 | 5.9e-46 | +0.573 [+0.512, +0.633] | 52.5 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 43.2% (accuracy-implied bound 71.4%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 56.2 | swapped: +31.3 [+26.8, +35.6] |
| swapped | none | 24.9 | - |
