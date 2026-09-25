import sys
import numpy as np
sys.path.insert(0, "src")
from flygnn.tools40 import ALL_TOOLS, GRAPH_TOOLS, ML_TOOLS, SEQ_TOOLS

def test_exactly_40():
    assert len(ALL_TOOLS) == 40
    assert len(GRAPH_TOOLS) == 20 and len(ML_TOOLS) == 16 and len(SEQ_TOOLS) == 4

def toy():
    A = np.zeros((6, 6))
    A[0, 1] = A[1, 2] = A[2, 0] = 1   # directed 3-cycle
    A[3, 4] = A[4, 5] = A[5, 3] = 1   # second 3-cycle
    A[0, 3] = 1
    return A

def test_reciprocity_zero_on_cycles():
    assert GRAPH_TOOLS["reciprocity"](toy()) == 0.0

def test_wcc_two_components():
    A = toy(); A[0, 3] = 0  # remove the bridge -> two 3-cycles
    r = GRAPH_TOOLS["wcc"](A)
    assert r["n_wcc"] == 2 and r["largest"] == 3
    assert GRAPH_TOOLS["wcc"](toy())["n_wcc"] == 1  # bridged -> one

def test_pagerank_sums_to_one():
    pr = GRAPH_TOOLS["pagerank"](toy())
    assert abs(pr.sum() - 1.0) < 1e-6

def test_jaccard():
    assert ML_TOOLS["jaccard_stability"]([1, 2, 3], [2, 3, 4]) == 0.5

def test_threshold_sweep_monotone():
    rng = np.random.RandomState(0)
    q = rng.rand(50, 4); q /= q.sum(1, keepdims=True)
    y = rng.randint(0, 4, 50)
    sw = ML_TOOLS["threshold_sweep"](q, y)
    assert sw["0.4"] >= sw["0.5"] >= sw["0.6"]

def test_homophily_perfect():
    A = np.zeros((4, 4)); A[0, 1] = A[2, 3] = 1
    y = np.array(["a", "a", "b", "b"], dtype=object)
    assert ML_TOOLS["homophily"](A, y) == 1.0

def test_seq_tools():
    assert SEQ_TOOLS["seq_gc"]("GGCC") == 1.0
    assert SEQ_TOOLS["seq_kmer_distance"]("AAAA", "AAAA") == 0.0
    assert abs(SEQ_TOOLS["seq_entropy"]("ACGT") - 2.0) < 1e-9
    assert SEQ_TOOLS["seq_codon_usage"]("AAATTT")["AAA"] == 1
