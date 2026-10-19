# mmstar (bridge:Qwen/Qwen2-0.5B-Instruct)

n = 1500 questions (paired)

| image | context | Acc % [95% CI] | Circ % | Perm % | dAcc vs ref (pp) | McNemar p | VIG nats [95% CI] | Flip % |
|---|---|---|---|---|---|---|---|---|
| real | none | 33.8 [31.5, 36.2] | 33.8 | 33.8 | ref | ref | ref | ref |
| swapped | none | 26.5 [24.4, 28.8] | 26.5 | 26.5 | +7.3 | 1.5e-10 | +0.105 [+0.071, +0.138] | 35.4 |

Chance accuracy (uniform guess): 26.9%

| category | n | Acc real % | Acc swapped % |
|---|---|---|---|
| coarse perception | 250 | 50.8 | 25.2 |
| fine-grained perception | 250 | 24.4 | 24.4 |
| instance reasoning | 250 | 42.0 | 25.2 |
| logical reasoning | 250 | 32.0 | 28.8 |
| math | 250 | 26.4 | 27.6 |
| science & technology | 250 | 27.2 | 28.0 |
