# mmstar (bridge:Qwen/Qwen2-7B-Instruct)

n = 1500 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 36.5 [34.1, 39.0] | 36.5 | 36.5 | ref | ref | ref | ref |
| swapped | none | 27.5 [25.3, 29.8] | 27.5 | 27.5 | +9.0 | 1.3e-08 | +0.142 [+0.043, +0.240] | 61.1 |

Chance accuracy (uniform guess): 26.9%

| category | n | Acc real % | Acc swapped % |
|---|---|---|---|
| coarse perception | 250 | 52.8 | 25.6 |
| fine-grained perception | 250 | 30.4 | 24.0 |
| instance reasoning | 250 | 48.0 | 30.8 |
| logical reasoning | 250 | 35.2 | 26.8 |
| math | 250 | 28.4 | 32.4 |
| science & technology | 250 | 24.4 | 25.6 |
