# mmstar (bridge:Qwen/Qwen2-0.5B-Instruct)

n = 1500 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 35.6 [33.2, 38.1] | 35.6 | 35.6 | ref | ref | ref | ref |
| swapped | none | 26.9 [24.7, 29.2] | 26.9 | 26.9 | +8.7 | 3.7e-10 | +0.139 [+0.102, +0.176] | 48.5 |

Chance accuracy (uniform guess): 26.9%

| category | n | Acc real % | Acc swapped % |
|---|---|---|---|
| coarse perception | 250 | 51.2 | 26.4 |
| fine-grained perception | 250 | 29.2 | 24.8 |
| instance reasoning | 250 | 42.0 | 27.2 |
| logical reasoning | 250 | 36.0 | 28.0 |
| math | 250 | 25.6 | 27.2 |
| science & technology | 250 | 29.6 | 28.0 |
