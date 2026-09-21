"""M2c: replication driver over the unchanged M2b harness plus the C3 rank-sum test."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from organism_sim.benchmarks import m2b, m2c  # noqa: E402


def test_rank_sum_perm_test_directions():
    small = [1, 2, 3, 4, 5, 6, 7, 8]
    big = [10, 11, 12, 13, 14, 15, 16, 17]
    assert m2c.rank_sum_perm_test(small, big, n_perm=2000)["p_one_sided"] < 0.01
    assert m2c.rank_sum_perm_test(big, small, n_perm=2000)["p_one_sided"] > 0.99
    same = m2c.rank_sum_perm_test(small, small, n_perm=2000)["p_one_sided"]
    assert 0.3 < same < 0.8
    # ties are handled (average ranks): identical-valued samples are not "smaller"
    assert m2c.rank_sum_perm_test([5] * 6, [5] * 6, n_perm=500)["p_one_sided"] > 0.9


def test_seeds_are_fresh_and_harness_is_m2b():
    assert set(m2c.SEEDS) == set(range(30, 35))
    assert not set(m2c.SEEDS) & (set(range(0, 30)) | set(range(100, 110)))
    assert m2c.run_arm is m2b.run_arm and m2c.ARMS == m2b.ARMS and m2c.FAMILIES == m2b.FAMILIES


def test_compare_structure_on_the_cheap_family():
    res = m2c.compare(seeds=(30,), families=("6bit",), arms=("covering", "periodic", "shape", "covering-matched"))
    fam = res["families"]["6bit"]
    v = fam["verdict"]
    assert set(v) >= {"C1", "C2", "C3", "C3_tests", "pass", "n"}
    assert v["n"] == 1 and len(fam["rows"][0]["arms"]["shape"]["scores"]) == 4
    assert 0 < v["C3_tests"]["shape_vs_covering"]["p_one_sided"] <= 1
    assert res["verdict"]["judged_on"] == "16bit" and res["verdict"]["pass"] is False
