n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| none | caption | 72.8 [70.1, 75.3] | 60.6 | 74.3 | -27.8 | 5.1e-55 | -0.902 [-1.082, -0.714] | 47.5 |
| none | none | 45.0 [42.1, 47.9] | 23.8 | 45.9 | ref | ref | ref | ref |
| none | rationale | 96.3 [95.1, 97.3] | 93.5 | 96.5 | -51.4 | 6.2e-160 | -1.828 [-1.996, -1.652] | 54.3 |
| none | rationale_masked | 82.9 [80.6, 85.0] | 73.3 | 84.2 | -37.9 | 3.3e-94 | -1.350 [-1.519, -1.177] | 49.8 |

**Prior-wrong subset P** (prior: own, 4 rotation(s); n_P = 619 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| none | caption | 62.4 | - |
| none | none | 13.1 | - |
| none | rationale | 94.8 | - |
| none | rationale_masked | 74.6 | - |

**Leak-free subset** (context has no stem of the gold option or of a direct answer)

- none / caption: Acc 63.2% on n = 642
- none / rationale: Acc 90.7% on n = 151
- none / rationale_masked: Acc 82.9% on n = 1145
