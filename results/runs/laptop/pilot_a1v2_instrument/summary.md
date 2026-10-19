n = 1145 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| none | caption | 55.9 [53.0, 58.7] | 38.8 | 60.9 | -26.7 | 1e-53 | -0.490 [-0.547, -0.432] | 49.0 |
| none | none | 29.2 [26.6, 31.9] | 6.4 | 36.4 | ref | ref | ref | ref |
| none | rationale | 86.6 [84.5, 88.4] | 74.5 | 92.2 | -57.4 | 2.4e-186 | -1.139 [-1.190, -1.089] | 62.3 |
| none | rationale_masked | 57.8 [54.9, 60.6] | 40.4 | 63.1 | -28.6 | 6.1e-58 | -0.520 [-0.572, -0.467] | 52.5 |

**Prior-wrong subset P** (prior: own, 4 rotation(s); n_P = 728 of 1145). P uses the rotation-averaged prior while Acc|P is at rotation 0, so the NONE row can have Acc|P > 0.

| image | context | Acc\|P % | real - control on P (pp) [95% CI] |
|---|---|---|---|
| none | caption | 48.4 | - |
| none | none | 17.6 | - |
| none | rationale | 83.2 | - |
| none | rationale_masked | 45.7 | - |

**Leak-free subset** (context has no stem of the gold option or of a direct answer)

- none / caption: Acc 39.6% on n = 642
- none / rationale: Acc 74.8% on n = 151
- none / rationale_masked: Acc 57.8% on n = 1145
