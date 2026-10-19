n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| blank | none | 31.8 [29.2, 34.5] | 9.5 | 35.9 | +20.3 | 3.1e-28 | +0.319 [+0.287, +0.352] | 55.4 |
| none | none | 29.2 [26.6, 31.9] | 6.4 | 36.4 | +22.9 | 2e-33 | +0.399 [+0.351, +0.448] | 60.6 |
| real | none | 52.1 [49.2, 54.9] | 31.0 | 59.4 | ref | ref | ref | ref |
| swapped | none | 30.3 [27.7, 33.0] | 8.3 | 32.4 | +21.7 | 4.6e-33 | +0.360 [+0.324, +0.396] | 55.5 |

**Arbitration, legacy_rot0 (real vs no image, own NONE prior at rotation 0; quote the Prior-wrong subset P block below instead)**

- Acc with image when prior is right (n=334): 65.3%
- Acc with image when prior is wrong (n=811): 46.6%
- Blind agreement (same answer with/without image): 39.4%
- Prior-subtracted accuracy (rotation 0): a=0: 52.1, a=0.25: 49.6, a=0.5: 43.2, a=0.75: 39.5, a=1: 34.9
- Debiased accuracy, NONE prior, rotation-averaged: a=0: 59.4, a=0.25: 57.4, a=0.5: 53.2, a=0.75: 49.8, a=1: 45.1 (ties at a=1: 0.0%)
- Debiased accuracy, BLANK prior, rotation-averaged: a=0: 59.4, a=0.25: 59.1, a=0.5: 58.4, a=0.75: 54.8, a=1: 53.2 (ties at a=1: 0.0%)

**Prior-wrong subset P** (prior: own, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 47.2% (accuracy-implied bound 77.0%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| blank | none | 17.2 | - |
| none | none | 17.6 | - |
| real | none | 45.7 | swapped: +23.8 [+19.4, +27.9]; blank: +28.6 [+24.3, +32.8] |
| swapped | none | 22.0 | - |
