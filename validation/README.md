# Locked post-outcome rerun

This is NOT preregistration. Historical results were inspected first. It cannot repair a missing protocol predating those outcomes or establish strict promotion.

The exact dataset, source, split body IDs, seeds, thresholds, environment and gate are frozen in protocol.json and checked by run_locked.py. Seeds are applied before model construction, fixing the old runner's initialization gap. The gate is copied from the historical baseline comparison; at least one descriptive nomination is required, not the historical count of 97. No new holdout, pair-grouped split, independent annotation confirmation or biological experiment is claimed. README's earlier 14-class count is wrong: the actual aligned task has 16 classes.

Run: `python validation/run_locked.py`. Results go only to validation/rerun.json; historical results remain unchanged. The lock commit precedes this rerun locally but follows historical outcomes.

## Pair-grouped split check (methods note, post-outcome)

2442 of 2578 nodes have a bilateral mate. In the locked split, 606 of 1037 test nodes have their mate in train. `pair_grouped.py` keeps mates on the same side of a class-stratified ~60/40 split (split seed 11; model seeds 11, 22, 33). Result in `pair_grouped.json`: GCN 0.529, CNN 0.440, linear 0.426; the baseline gate still passes. Pair leakage inflates GCN by about 1.7 points (0.546 locked vs 0.529) and does not explain the roughly 9-point GCN lead. Caveats: single split seed, run after outcomes were known, descriptive only, not part of the frozen protocol, no preregistration claimed. Candidate lists are init-sensitive: 75-106 candidates across seeds 1-10, 64 of the locked 92 present in all 10 seeds.
