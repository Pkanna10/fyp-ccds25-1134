n = 64 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| blank | none | 39.1 [28.1, 51.3] | 18.8 | 37.5 | +20.3 | 0.019 | +0.319 [+0.180, +0.464] | 54.7 |
| none | none | 40.6 [29.5, 52.9] | 14.1 | 39.1 | +18.8 | 0.036 | +0.353 [+0.165, +0.557] | 62.5 |
| real | none | 59.4 [47.1, 70.5] | 40.6 | 57.8 | ref | ref | ref | ref |

**Arbitration, legacy_rot0 (real vs no image, own NONE prior at rotation 0; quote the Prior-wrong subset P block below instead)**

- Acc with image when prior is right (n=26): 69.2%
- Acc with image when prior is wrong (n=38): 52.6%
- Blind agreement (same answer with/without image): 37.5%
- Prior-subtracted accuracy (rotation 0): a=0: 59.4, a=0.25: 42.2, a=0.5: 39.1, a=0.75: 39.1, a=1: 37.5
- Debiased accuracy, NONE prior, rotation-averaged: a=0: 57.8, a=0.25: 57.8, a=0.5: 50.0, a=0.75: 43.8, a=1: 40.6 (ties at a=1: 0.0%)
- Debiased accuracy, BLANK prior, rotation-averaged: a=0: 57.8, a=0.25: 57.8, a=0.5: 57.8, a=0.75: 56.2, a=1: 50.0 (ties at a=1: 0.0%)

**Prior-wrong subset P** (prior: own, 4 rotation(s); n_P = 39 of 64). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 53.1% (accuracy-implied bound 81.2%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| blank | none | 17.9 | - |
| none | none | 20.5 | - |
| real | none | 48.7 | blank: +30.8 [+12.8, +48.7] |
