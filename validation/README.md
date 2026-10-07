# Locked post-outcome rerun

This is NOT preregistration. Historical results were inspected first. It cannot repair a missing protocol predating those outcomes or establish strict promotion.

The exact dataset, source, split body IDs, seeds, thresholds, environment and gate are frozen in protocol.json and checked by run_locked.py. Seeds are applied before model construction, fixing the old runner's initialization gap. The gate is copied from the historical baseline comparison; at least one descriptive nomination is required, not the historical count of 97. No new holdout, pair-grouped split, independent annotation confirmation or biological experiment is claimed. README's earlier 14-class count is wrong: the actual aligned task has 16 classes.

Run: `python validation/run_locked.py`. Results go only to validation/rerun.json; historical results remain unchanged. The lock commit precedes this rerun locally but follows historical outcomes.
