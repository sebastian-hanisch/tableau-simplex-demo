"""Jede Zahl in den Grenzen der App und im README, gegen die echten Auswertungsfunktionen (feste Seeds, plattformstabil dank random.Random; Bänder bei Zählgrößen)."""

from dataclasses import replace
from functools import lru_cache

import pytest

import tsx_algorithm as A
import tsx_constants as C
import tsx_evaluation as ev
import tsx_scenario as S


@lru_cache(maxsize=None)
def _curve(kind):
    return ev.pivot_curve(kind)


def _text():
    return open("app.py", encoding="utf-8").read()


def test_random_le_instances_need_about_half_a_pivot_per_resource():
    rows = _curve("random")
    assert [r["size"] for r in rows] == [5, 10, 20, 40, 60]
    for r, want in zip(rows, (3.0, 5.5, 10.5, 20.0, 32.5)):
        assert r["pivots"] == pytest.approx(want, abs=max(1.0, 0.06 * want))
    assert all(0.45 <= r["per_m"] <= 0.65 for r in rows) and all(r["phase1_share"] == 0.0 for r in rows)
    _t = _text()
    for v in ("3/5.5/10.5/20/32.5 Pivots", "0.5 bis 0.6 je Ressource"):
        assert v in _t, v


def test_mixed_instances_need_more_pivots_per_resource_and_more_with_size():
    rows = _curve("mixed")
    for r, want in zip(rows, (5.0, 13.0, 38.0, 105.5, 199.0)):
        assert r["pivots"] == pytest.approx(want, rel=0.06, abs=1.0)
    per_m = [r["per_m"] for r in rows]
    assert per_m == sorted(per_m) and per_m[0] == pytest.approx(1.0, abs=0.1) and per_m[-1] == pytest.approx(3.3, abs=0.15)
    share = [r["phase1_share"] for r in rows]
    assert min(share) == pytest.approx(0.25, abs=0.03) and max(share) == pytest.approx(0.39, abs=0.03) and share[-1] == pytest.approx(0.25, abs=0.03)
    _t = _text()
    for v in ("5/13/38/105.5/199", "1.0 bis 3.3 je Ressource", "25 bis 39 %", "25 % von 199 Pivots"):
        assert v in _t, v


def test_ties_are_common_but_zero_steps_never_happen_in_the_tie_instances():
    got = []
    for k in (10, 20, 40):
        r = ev.run_config(ev.Settings("degenerate", k, k, 0.5, 0))
        assert r["degenerate_share"] == 0.0
        got.append(round(100 * r["tie_share"], 1))
    assert got[0] == pytest.approx(16.7, abs=3.0) and got[1] == pytest.approx(36.4, abs=3.0) and got[2] == pytest.approx(15.8, abs=3.0)
    _t = _text()
    for v in ("16.7/36.4/15.8 %", "aber keinen einzigen Nullschritt"):
        assert v in _t, v


def test_the_bland_safety_net_is_never_needed_on_generated_instances():
    for kind in ("random", "mixed", "degenerate"):
        for seed in range(200000, 200040):
            for m, n in ((5, 5), (20, 20), (40, 30)):
                assert not A.tableau_simplex(S.generate(kind, m, n, 0.5, seed)).switched_to_bland
    assert "die Bland-Regel als Notbremse wurde nie gebraucht" in _text()


def test_a_dense_pivot_costs_columns_plus_one_times_one_plus_two_m():
    a = ev.analyse(ev.Settings("random", 60, 60, 0.5, 35))
    assert a.res.flops_per_pivot == (a.res.n_cols + 1) * (1 + 2 * a.res.m) == 14641
    for kind in ("random", "mixed", "degenerate"):
        for m, n in ((3, 4), (10, 7), (25, 30)):
            r = A.tableau_simplex(S.generate(kind, m, n, 0.5, 3))
            assert r.flops_per_pivot == (r.n_cols + 1) * (1 + 2 * r.m)
    assert "bei m = n = 60 sind das 14 641" in _text()


def test_every_generated_result_agrees_with_highs():
    from tests.test_scenario import _highs
    count = 0
    for kind in ("random", "mixed", "degenerate"):
        for seed in range(10):
            for m, n in ((4, 4), (12, 9), (25, 20)):
                inst = S.generate(kind, m, n, 0.5, 500 + seed)
                assert A.tableau_simplex(inst).obj == pytest.approx(-_highs(inst).fun, rel=1e-7, abs=1e-7)
                count += 1
    assert count == 90


def test_densities_and_sizes_do_not_change_the_status_of_generated_instances():
    base = ev.Settings("random", 10, 10, 0.5, 0)
    for d in C.DENSITY_OPTIONS:
        assert ev.run_config(replace(base, density=d))["solved"] == 5


def test_zero_steps_occur_only_in_mixed_instances_and_rarely():
    zero = {"random": 0, "mixed": 0, "degenerate": 0}
    total = 0
    for kind in zero:
        for seed in range(200000, 200060):
            for m, n in ((5, 5), (10, 8), (20, 20), (40, 30)):
                r = A.tableau_simplex(S.generate(kind, m, n, 0.5, seed))
                zero[kind] += r.degenerate_pivots
                total += r.total_pivots
    assert zero["random"] == 0 and zero["degenerate"] == 0 and 8 <= zero["mixed"] <= 60 and total > 10000
    assert "22 in gut 11 000 Pivots auf 720 Instanzen" in open("README.md", encoding="utf-8").read()


def test_pivots_grow_slower_than_the_size_when_only_one_dimension_grows():
    S0 = ev.Settings
    fixed_m = [ev.run_config(S0("random", 20, n, 0.5, 0))["pivots"] for n in (5, 20, 60)]
    fixed_n = [ev.run_config(S0("random", m, 20, 0.5, 0))["pivots"] for m in (5, 20, 60)]
    assert fixed_m == pytest.approx([4, 7, 15], abs=2) and fixed_n == pytest.approx([5, 7, 13], abs=2)
    assert fixed_m[2] / fixed_m[0] < 60 / 5 and fixed_n[2] / fixed_n[0] < 60 / 5
    assert "4/7/15" in open("README.md", encoding="utf-8").read()
