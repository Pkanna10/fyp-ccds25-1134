# mmstar (bridge:Qwen/Qwen2-0.5B-Instruct)

n = 1500 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 35.7 [33.3, 38.2] | 35.7 | 35.7 | ref | ref | ref | ref |
| swapped | none | 26.1 [24.0, 28.4] | 26.1 | 26.1 | +9.6 | 2.7e-12 | +0.146 [+0.109, +0.184] | 47.5 |

Chance accuracy (uniform guess): 26.9%

| category | n | Acc real % | Acc swapped % |
|---|---|---|---|
| coarse perception | 250 | 51.2 | 24.8 |
| fine-grained perception | 250 | 29.2 | 24.8 |
| instance reasoning | 250 | 43.2 | 23.6 |
| logical reasoning | 250 | 33.6 | 28.4 |
| math | 250 | 27.6 | 28.8 |
| science & technology | 250 | 29.6 | 26.4 |
