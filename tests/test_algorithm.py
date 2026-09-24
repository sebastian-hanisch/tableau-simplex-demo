"""Korrektheitskette des Tableau-Simplex: HiGHS-Gegenprobe, Dualität, Unzulässig/Unbeschränkt, Ecken, Lehrbuchbeispiel von Hand, Buchführung, Sonderfälle."""

import numpy as np
import pytest

import tsx_algorithm as A
import tsx_scenario as S
from tests.test_scenario import _highs, reference_status

STATUS = {0: "optimal", 2: "infeasible", 3: "unbounded"}


def _solve(inst, **kw):
    return A.tableau_simplex(inst, **kw)


def _custom(rows, b, c, senses):
    n = len(c)
    return S.Instance(tuple(tuple(float(v) for v in r) for r in rows), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(senses), tuple(f"x{j}" for j in range(n)), tuple(f"r{i}" for i in range(len(b))), "custom")


@pytest.mark.parametrize("kind", ["random", "mixed", "degenerate"])
def test_optimum_equals_highs_and_certificates_hold(kind):
    count = 0
    for seed in range(40):
        for m, n in ((2, 2), (3, 3), (8, 5), (5, 12), (20, 20), (30, 25)):
            inst = S.generate(kind, m, n, 0.5, seed)
            r, h = _solve(inst), _highs(inst)
            assert r.status == STATUS[h.status] == "optimal", (kind, seed, m, n)
            assert r.obj == pytest.approx(-h.fun, rel=1e-7, abs=1e-7)
            assert A.primal_violation(inst, r.x) < 1e-7                                     # primal zulässig
            assert A.dual_violation(inst, r.duals) < 1e-6                                   # dual zulässig
            b = np.array(inst.b)
            assert float(np.dot(r.duals, b)) == pytest.approx(r.obj, rel=1e-7, abs=1e-7)    # starke Dualität
            slack = np.array(inst.arrays()[0]) @ np.array(r.x) - b
            assert all(abs(y * s) < 1e-6 * max(1.0, abs(r.obj)) for y, s in zip(r.duals, slack))    # komplementärer Schlupf
            count += 1
    assert count == 240


def test_infeasible_and_unbounded_are_detected_and_never_reported_as_optimal():
    r = _solve(S.infeasible_instance())
    assert r.status == "infeasible" and r.x == () and np.isnan(r.obj)
    r = _solve(S.unbounded_instance())
    assert r.status == "unbounded" and r.x == ()
    rng = np.random.default_rng(5)
    for _ in range(100):
        m, n = int(rng.integers(2, 7)), int(rng.integers(2, 6))
        base = S.generate("random", m, n, 0.6, int(rng.integers(0, 10 ** 6)))
        rows = [list(row) for row in base.A] + [[1.0] + [0.0] * (n - 1)]
        inf = _custom(rows, list(base.b) + [1e4], base.c, list(base.senses) + [S.GE])          # Mindestmenge, die keine Ressource hergibt
        assert _solve(inf).status == reference_status(inf) == "infeasible"
        rows2 = [[0.0] + [float(v) for v in np.round(rng.uniform(0.5, 4.0, n - 1), 2)]]
        unb = _custom(rows2, [50.0], [1.0] + [0.0] * (n - 1), [S.LE])                          # die erste Variable kommt in keiner Bedingung vor
        assert _solve(unb).status == reference_status(unb) == "unbounded"


def test_random_small_lps_agree_with_highs_in_all_three_statuses():
    rng = np.random.default_rng(11)
    seen = set()
    for _ in range(300):
        m, n = int(rng.integers(1, 6)), int(rng.integers(1, 5))
        Am = np.round(rng.uniform(-3, 4, (m, n)), 1)
        b = np.round(rng.uniform(-5, 20, m), 1)
        c = np.round(rng.uniform(-3, 6, n), 1)
        senses = [str(s) for s in rng.choice([S.LE, S.GE, S.EQ], size=m, p=[0.55, 0.3, 0.15])]
        inst = _custom(Am, b, c, senses)
        r, h = _solve(inst), _highs(inst)
        assert r.status == reference_status(inst), (Am, b, c, senses)
        seen.add(r.status)
        if r.status == "optimal":
            assert r.obj == pytest.approx(-h.fun, rel=1e-6, abs=1e-6) and A.primal_violation(inst, r.x) < 1e-6 and A.dual_violation(inst, r.duals) < 1e-5
    assert seen == {"optimal", "infeasible", "unbounded"}


def test_textbook_by_hand():
    r = _solve(S.textbook_instance(), keep=True)
    assert r.status == "optimal" and r.obj == 36.0 and r.x == (2.0, 6.0) and r.duals == (0.0, 1.5, 1.0) and r.total_pivots == 2 and r.phase1_pivots == 0
    p1, p2 = r.pivots
    assert (p1.enter, p1.leave_row, p1.ratio, p1.obj, p1.x) == (1, 1, 6.0, 30.0, (0.0, 6.0)) and (p2.enter, p2.leave_row, p2.ratio, p2.obj, p2.x) == (0, 2, 2.0, 36.0, (2.0, 6.0))
    t0, t1, t2 = (s["T"] for s in r.snapshots)
    assert np.allclose(t0, [[1, 0, 1, 0, 0, 4], [0, 2, 0, 1, 0, 12], [3, 2, 0, 0, 1, 18], [-3, -5, 0, 0, 0, 0]])
    assert np.allclose(t1, [[1, 0, 1, 0, 0, 4], [0, 1, 0, 0.5, 0, 6], [3, 0, 0, -1, 1, 6], [-3, 0, 0, 2.5, 0, 30]])
    assert np.allclose(t2, [[0, 0, 1, 1 / 3, -1 / 3, 2], [0, 1, 0, 0.5, 0, 6], [1, 0, 0, -1 / 3, 1 / 3, 2], [0, 0, 0, 1.5, 1, 36]])
    assert [s["basis"] for s in r.snapshots] == [(2, 3, 4), (2, 1, 4), (2, 1, 0)] and r.col_names == ("x1", "x2", "s1", "s2", "s3")


def test_every_basic_solution_is_a_vertex_and_the_objective_never_falls():
    for kind in ("random", "degenerate"):
        for seed in range(30):
            inst = S.generate(kind, 5, 2, 0.7, seed)
            verts = A.vertices_2d(inst)
            r = _solve(inst)
            path = [(0.0, 0.0)] + [p.x for p in r.pivots]
            for x in path:
                assert any(np.allclose(x, v, atol=1e-7) for v in verts), (kind, seed, x)
            objs = [float(np.dot(inst.c, x)) for x in path]
            assert all(b >= a - 1e-9 for a, b in zip(objs, objs[1:]))
            for a, b, p in zip(objs, objs[1:], r.pivots):
                assert b > a + 1e-9 or p.degenerate
            assert r.obj == pytest.approx(max(float(np.dot(inst.c, v)) for v in verts), abs=1e-7)    # das Optimum ist die beste Ecke


def test_vertices_of_the_textbook_polygon():
    assert sorted(A.vertices_2d(S.textbook_instance())) == [(0.0, 0.0), (0.0, 6.0), (2.0, 6.0), (4.0, 0.0), (4.0, 3.0)]
    with pytest.raises(ValueError):
        A.vertices_2d(S.generate("random", 3, 3, 0.5, 1))


def test_degenerate_fixture_creates_a_tie_and_a_zero_basic_variable():
    inst = S.degenerate_instance()
    r = _solve(inst)
    assert r.obj == 36.0 and r.x == (2.0, 6.0)
    assert [p.ties for p in r.pivots] == [1, 2] and r.pivots[-1].zero_basics >= 1 and r.degenerate_pivots == 0
    rb = _solve(inst, rule="bland")
    assert rb.obj == 36.0 and not rb.switched_to_bland


def test_bland_rule_gives_the_same_optimum_and_terminates():
    for seed in range(20):
        for kind in ("random", "mixed", "degenerate"):
            inst = S.generate(kind, 10, 8, 0.5, seed)
            assert _solve(inst, rule="bland").obj == pytest.approx(_solve(inst).obj, rel=1e-8)


def test_bookkeeping_pivots_flops_phases_and_snapshots():
    for kind in ("random", "mixed", "degenerate"):
        for seed in range(10):
            inst = S.generate(kind, 6, 4, 0.6, seed)
            r = _solve(inst, keep=True)
            assert r.total_pivots == len(r.pivots) and [p.k for p in r.pivots] == list(range(1, r.total_pivots + 1))
            assert r.flops == r.total_pivots * ((r.n_cols + 1) + 2 * r.m * (r.n_cols + 1))
            assert r.phase1_pivots == sum(p.phase == 1 for p in r.pivots) and all(p.phase == 2 for p in r.pivots[r.phase1_pivots:])
            for s in r.snapshots:
                T, basis = s["T"], s["basis"]
                for i, j in enumerate(basis):                                               # Basisspalten sind Einheitsvektoren
                    expected = np.zeros(T.shape[0] - 1)
                    expected[i] = 1.0
                    assert np.allclose(T[:-1, j], expected, atol=1e-8)
    inst = S.generate("random", 5, 4, 0.5, 2)
    assert _solve(inst).snapshots == [] and len(_solve(inst, keep=True).snapshots) == _solve(inst).total_pivots + 1


def test_phase1_ends_at_a_feasible_basis_and_all_le_instances_skip_it():
    saw_phase1 = False
    for seed in range(20):
        inst = S.generate("mixed", 8, 5, 0.5, seed)
        p1 = [p for p in _solve(inst).pivots if p.phase == 1]
        if p1:
            saw_phase1 = True
            assert p1[-1].feasible and p1[-1].obj == pytest.approx(0.0, abs=1e-7)
    assert saw_phase1
    for seed in range(20):
        assert _solve(S.generate("random", 12, 8, 0.5, seed)).phase1_pivots == 0


def test_special_cases():
    r = _solve(_custom([[2.0]], [10.0], [3.0], [S.LE]))
    assert r.status == "optimal" and r.x == (5.0,) and r.obj == 15.0 and r.duals == (1.5,)
    zero_c = _custom([[1.0, 1.0]], [4.0], [0.0, 0.0], [S.LE])
    assert _solve(zero_c).obj == 0.0 and _solve(zero_c).total_pivots == 0
    full_deg = _custom([[1.0, 1.0], [1.0, 2.0]], [0.0, 0.0], [1.0, 1.0], [S.LE, S.LE])
    assert _solve(full_deg).obj == 0.0 and _solve(full_deg).status == "optimal"
    dup = _custom([[1.0, 2.0], [1.0, 2.0], [2.0, 4.0]], [6.0, 6.0, 12.0], [1.0, 1.0], [S.EQ, S.EQ, S.EQ])
    rd = _solve(dup)
    assert rd.status == "optimal" and rd.obj == pytest.approx(6.0) and rd.redundant_rows == 2
    negative_b = _custom([[1.0, 1.0]], [-3.0], [1.0, 1.0], [S.GE])
    assert _solve(negative_b).status == reference_status(negative_b)
    neg_le = _custom([[1.0, 1.0], [-1.0, -2.0]], [10.0, -3.0], [1.0, 2.0], [S.LE, S.LE])                # zweite Zeile: x1 + 2 x2 >= 3
    assert _solve(neg_le).obj == pytest.approx(-_highs(neg_le).fun) and _solve(neg_le).phase1_pivots > 0


def test_determinism():
    inst = S.generate("mixed", 10, 7, 0.5, 4)
    a, b = _solve(inst), _solve(inst)
    assert a.x == b.x and [(p.enter, p.leave_row) for p in a.pivots] == [(p.enter, p.leave_row) for p in b.pivots]
