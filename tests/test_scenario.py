"""Instanzen: Bauart, Determinismus, Fixtures, Zulässigkeit der erzeugten Mischinstanzen."""

import numpy as np
import pytest
from scipy.optimize import linprog

import tsx_scenario as S


def _highs(inst, zero_objective=False):
    A, b, c = inst.arrays()
    if zero_objective:
        c = np.zeros_like(c)
    ub_a, ub_b, eq_a, eq_b = [], [], [], []
    for i, s in enumerate(inst.senses):
        if s == S.LE:
            ub_a.append(A[i]), ub_b.append(b[i])
        elif s == S.GE:
            ub_a.append(-A[i]), ub_b.append(-b[i])
        else:
            eq_a.append(A[i]), eq_b.append(b[i])
    return linprog(-c, A_ub=np.array(ub_a) if ub_a else None, b_ub=ub_b or None, A_eq=np.array(eq_a) if eq_a else None, b_eq=eq_b or None, bounds=(0, None), method="highs")


def test_textbook_is_the_classic_two_service_example():
    inst = S.textbook_instance()
    assert (inst.m, inst.n) == (3, 2) and inst.senses == (S.LE,) * 3 and inst.b == (4.0, 12.0, 18.0) and inst.c == (3.0, 5.0)
    assert -_highs(inst).fun == pytest.approx(36.0)


def test_fixtures_have_the_status_they_claim():
    assert _highs(S.infeasible_instance()).status == 2
    assert _highs(S.unbounded_instance()).status == 3
    d = S.degenerate_instance()
    assert -_highs(d).fun == pytest.approx(36.0) and d.A[3] == (1.0, 1.0) and d.b[3] == 8.0


def test_generation_is_deterministic_and_seed_dependent():
    a = S.generate("random", 8, 6, 0.5, 3)
    assert a == S.generate("random", 8, 6, 0.5, 3) and a != S.generate("random", 8, 6, 0.5, 4)
    assert hash(a) == hash(S.generate("random", 8, 6, 0.5, 3))


@pytest.mark.parametrize("kind", ["random", "mixed", "degenerate"])
def test_generated_instances_are_feasible_and_bounded(kind):
    for seed in range(40):
        for m, n in ((2, 2), (6, 4), (4, 9), (15, 15), (30, 20)):
            inst = S.generate(kind, m, n, 0.4, seed)
            assert (inst.m, inst.n) == (m, n) and len(inst.senses) == m and len(inst.names) == n and len(inst.row_names) == m
            assert _highs(inst).status == 0, (kind, seed, m, n)


def test_every_row_and_column_is_used_and_row_zero_is_dense():
    for seed in range(20):
        inst = S.generate("random", 10, 8, 0.1, seed)
        A = np.array(inst.A)
        assert (A.sum(axis=1) > 0).all() and (A.sum(axis=0) > 0).all() and (A[0] > 0).all() and (A >= 0).all()


def test_mixed_instances_contain_all_three_senses_sometimes():
    seen = set()
    for seed in range(30):
        seen |= set(S.generate("mixed", 12, 8, 0.5, seed).senses)
    assert seen == {S.LE, S.GE, S.EQ}
    for seed in range(30):
        inst = S.generate("mixed", 10, 4, 0.5, seed)
        assert sum(s == S.EQ for s in inst.senses) <= 2 and inst.senses[0] == S.LE


def test_degenerate_instances_repeat_rows_as_sums():
    inst = S.generate("degenerate", 8, 5, 0.6, 1)
    A, b = np.array(inst.A), np.array(inst.b)
    base = (8 + 1) // 2
    for i in range(base, 8):
        assert any(np.allclose(A[i], A[p] + A[q]) and abs(b[i] - (b[p] + b[q])) < 1e-6 for p in range(base) for q in range(base)) or any(np.allclose(A[i], A[p]) and abs(b[i] - b[p]) < 1e-6 for p in range(base))


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        S.generate("nope", 3, 3, 0.5, 1)


def reference_status(inst):
    """Status laut HiGHS; HiGHS meldet bei zulässigen, unbeschränkten LPs gelegentlich "unzulässig" (Präsolve), darum wird der Status über ein Zulässigkeitsproblem ohne Zielfunktion abgesichert."""
    h = _highs(inst)
    if h.status == 0:
        return "optimal"
    return "infeasible" if _highs(inst, zero_objective=True).status == 2 else "unbounded"


def test_mixed_equations_with_the_same_support_stay_feasible():
    """Regression: gerundete Gleichungs-Bedingungen mit gleichem Träger (etwa zwei Zeilen nur in x3) machten 9 von 9000 Mischinstanzen unzulässig."""
    for seed, m, n, d in ((1, 6, 4, 0.3), (23, 6, 4, 0.3), (31, 6, 4, 0.5), (76, 6, 4, 0.5), (110, 6, 4, 0.3), (133, 10, 8, 0.3), (142, 10, 8, 0.3), (163, 6, 4, 0.3), (168, 6, 4, 0.3)):
        assert _highs(S.generate("mixed", m, n, d, seed)).status == 0, (seed, m, n, d)
