n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 60.6 [57.8, 63.4] | 60.6 | 60.6 | ref | ref | ref | ref |
| swapped | none | 34.5 [31.8, 37.3] | 34.5 | 34.5 | +26.1 | 1.4e-47 | +0.992 [+0.853, +1.134] | 51.6 |

**Prior-wrong subset P** (prior: external, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

- Blind agreement vs this prior: 40.4% (accuracy-implied bound 75.8%)

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| real | none | 54.9 | swapped: +28.6 [+24.3, +32.7] |
| swapped | none | 26.4 | - |
