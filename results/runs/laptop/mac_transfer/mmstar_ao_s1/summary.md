# mmstar (bridge:Qwen/Qwen2-0.5B-Instruct)

n = 1500 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 35.5 [33.1, 37.9] | 35.5 | 35.5 | ref | ref | ref | ref |
| swapped | none | 27.0 [24.8, 29.3] | 27.0 | 27.0 | +8.5 | 2.5e-10 | +0.132 [+0.096, +0.170] | 44.7 |

Chance accuracy (uniform guess): 26.9%

| category | n | Acc real % | Acc swapped % |
|---|---|---|---|
| coarse perception | 250 | 52.0 | 24.8 |
| fine-grained perception | 250 | 27.2 | 24.0 |
| instance reasoning | 250 | 43.2 | 27.2 |
| logical reasoning | 250 | 33.6 | 30.8 |
| math | 250 | 26.4 | 26.8 |
| science & technology | 250 | 30.4 | 28.4 |
