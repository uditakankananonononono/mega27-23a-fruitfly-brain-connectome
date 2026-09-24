# mega27-23a-fruitfly-brain-connectome
GNN analysis of the real larval Drosophila brain connectome (Winding et al. 2023, Science; Supplementary-Data-S1, 2,953-neuron all-all matrix).
Task: neuron cell-type classification (14 classes, >=40 members). GCN on the binarized synaptic graph vs 1D-CNN and linear baselines on log morphology features, stratified node split, 3 seeds.
Result: GCN 0.547 vs CNN 0.451 vs linear 0.425 accuracy - graph structure carries the signal. Hermetic unit tests validate the GCN on a planted SBM where structure must dominate.
Run: `pip install -e . && pytest && python experiments/run_connectome.py`
