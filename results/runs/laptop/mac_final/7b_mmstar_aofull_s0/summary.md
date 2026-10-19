# mmstar (bridge:Qwen/Qwen2-7B-Instruct)

n = 1500 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 37.5 [35.1, 40.0] | 37.5 | 37.5 | ref | ref | ref | ref |
| swapped | none | 26.3 [24.2, 28.6] | 26.3 | 26.3 | +11.2 | 3.3e-14 | +0.287 [+0.222, +0.351] | 53.9 |

Chance accuracy (uniform guess): 26.9%

| category | n | Acc real % | Acc swapped % |
|---|---|---|---|
| coarse perception | 250 | 53.2 | 26.0 |
| fine-grained perception | 250 | 24.4 | 18.0 |
| instance reasoning | 250 | 48.4 | 25.6 |
| logical reasoning | 250 | 39.6 | 29.6 |
| math | 250 | 38.8 | 37.2 |
| science & technology | 250 | 20.8 | 21.6 |
