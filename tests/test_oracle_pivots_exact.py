"""Unabhängiges Orakel: jeder Pivot in exakter Bruchrechnung (fractions) aus der Basis neu berechnet, Ecken der Zeichnung exakt, Zahl der redundanten Zeilen über den Rang.

Aus der Basis vor dem Pivot wird B^-1 [A | b] exakt gelöst, daraus die reduzierten Kosten, die Kandidatenmenge, die Wahl der eintretenden Spalte (Dantzig bzw. Bland), die
Zeile des Quotiententests (kleinster Quotient, Gleichstand nach kleinstem Index), Schrittweite, Zielwert, Entartung und Gleichstandszahl. Dazu Status und Optimum gegen HiGHS
auf zufälligen kleinen LPs (alle drei Status), Beales Zyklus (Notbremse greift) und die Ecken von Zwei-Dienste-Instanzen gegen Schnittpunkte in Brüchen.
"""

import itertools
import random
from fractions import Fraction as F

import numpy as np
import pytest

import tsx_algorithm as A
import tsx_scenario as S


def _frac(v):
    return F(repr(float(v)))


def _exact_form(inst):
    m, n = inst.m, inst.n
    rows, rhs, sen = [], [], []
    for i in range(m):
        a, r, s = [_frac(v) for v in inst.A[i]], _frac(inst.b[i]), inst.senses[i]
        if r < 0:
            a, r, s = [-v for v in a], -r, {S.LE: S.GE, S.GE: S.LE, S.EQ: S.EQ}[s]
        rows.append(a), rhs.append(r), sen.append(s)
    nc, slack, art = n, {}, {}
    for i, s in enumerate(sen):
        if s in (S.LE, S.GE):
            slack[i], nc = nc, nc + 1
    for i, s in enumerate(sen):
        if s in (S.GE, S.EQ):
            art[i], nc = nc, nc + 1
    M = [[F(0)] * (nc + 1) for _ in range(m)]
    for i in range(m):
        M[i][:n], M[i][-1] = rows[i], rhs[i]
        if i in slack:
            M[i][slack[i]] = F(1) if sen[i] == S.LE else F(-1)
        if i in art:
            M[i][art[i]] = F(1)
    init = [art[i] if i in art else slack[i] for i in range(m)]
    return M, init, set(art.values()), n, nc, [_frac(v) for v in inst.c]


def _tableau(M, basis, m, nc):
    aug = [[M[i][basis[j]] for j in range(m)] + [F(int(i == j)) for j in range(m)] for i in range(m)]
    for c in range(m):
        p = next(i for i in range(c, m) if aug[i][c] != 0)
        aug[c], aug[p] = aug[p], aug[c]
        aug[c] = [v / aug[c][c] for v in aug[c]]
        for i in range(m):
            if i != c and aug[i][c] != 0:
                f = aug[i][c]
                aug[i] = [a - f * b for a, b in zip(aug[i], aug[c])]
    binv = [row[m:] for row in aug]
    return [[sum(binv[i][k] * M[k][j] for k in range(m)) for j in range(nc + 1)] for i in range(m)]


def _pivot_errors(inst, res, rules):
    """Fehlerliste je Pivot; `rules`: erlaubte Regeln für die Wahl der eintretenden Spalte (Dantzig und, nach der Notbremse, Bland)."""
    M, init, art, n, nc, cvec = _exact_form(inst)
    m = len(M)
    bases = [tuple(init)] + [p.basis for p in res.pivots]
    errs = []
    for k, p in enumerate(res.pivots):
        pre, post = list(bases[k]), list(bases[k + 1])
        T = _tableau(M, pre, m, nc)
        cph = [F(-1) if j in art else F(0) for j in range(nc)] if p.phase == 1 else [cvec[j] if j < n else F(0) for j in range(nc)]
        r = [sum(cph[pre[i]] * T[i][j] for i in range(m)) - cph[j] for j in range(nc)]
        allowed = range(nc) if p.phase == 1 else [j for j in range(nc) if j not in art]
        cand = [j for j in allowed if r[j] < 0]
        if p.phase == 1 and not cand:                                                  # Austausch einer künstlichen Basisvariablen nach Phase 1
            if not (pre[p.leave_row] in art and p.enter not in art and T[p.leave_row][p.enter] != 0):
                errs.append((k, "Austausch künstlicher Variablen"))
            continue
        e = p.enter
        if e not in cand:
            errs.append((k, "kein Kandidat"))
            continue
        ok_enter = ("bland" in rules and e == cand[0]) or ("dantzig" in rules and r[e] == min(r[j] for j in cand) and e == min(j for j in cand if r[j] == r[e]))
        if not ok_enter:
            errs.append((k, "Regelwahl", e))
        pos = [i for i in range(m) if T[i][e] > 0]
        ratios = {i: T[i][-1] / T[i][e] for i in pos}
        rmin = min(ratios.values())
        tied = [i for i in pos if ratios[i] == rmin]
        want = min(tied, key=lambda i: pre[i])
        if p.ties != len(tied) or p.leave_row != want or p.leave_var != pre[want] or abs(p.ratio - float(rmin)) > 1e-9 * max(1.0, float(rmin)):
            errs.append((k, "Quotiententest", p.leave_row, want))
        expected_post = pre[:]
        expected_post[want] = e
        T2 = _tableau(M, post, m, nc)
        before = sum(cph[pre[i]] * T[i][-1] for i in range(m))
        after = sum(cph[post[i]] * T2[i][-1] for i in range(m))
        if post != expected_post or after < before or abs(p.obj - float(after)) > 1e-9 * max(1.0, abs(p.obj)) or p.degenerate != (after == before):
            errs.append((k, "Basis/Zielwert/Entartung"))
        if p.zero_basics != sum(1 for i in range(m) if T2[i][-1] == 0):
            errs.append((k, "Nullbasisvariablen"))
    return errs


def _holds(lhs, sense, rhs):
    return lhs <= rhs if sense == S.LE else (lhs >= rhs if sense == S.GE else lhs == rhs)


def _random_lp(rng):
    m, n = rng.randint(1, 6), rng.randint(1, 5)
    rows = [[float(rng.choice([-2, -1, 0, 0, 1, 2, 3])) for _ in range(n)] for _ in range(m)]
    b = [float(rng.choice([0, 0, 1, 2, 4, 6, -1, -2])) for _ in range(m)]
    sen = [rng.choice([S.LE, S.LE, S.LE, S.GE, S.EQ]) for _ in range(m)]
    if m >= 2 and rng.random() < 0.3:                                                     # abhängige Zeile
        i, j = rng.sample(range(m), 2)
        rows.append([rows[i][k] + rows[j][k] for k in range(n)]), b.append(b[i] + b[j]), sen.append(S.EQ)
    return _custom(rows, b, [float(rng.choice([-2, -1, 0, 1, 2, 3])) for _ in range(n)], sen)


def _custom(rows, b, c, sen):
    n = len(c)
    return S.Instance(tuple(tuple(float(v) for v in r) for r in rows), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(sen), tuple(f"x{j}" for j in range(n)),
                      tuple(f"r{i}" for i in range(len(b))), "custom")


def _beale():
    return _custom([[0.25, -8, -1, 9], [0.5, -12, -0.5, 3], [0, 0, 1, 0]], [0, 0, 1], [0.75, -20, 0.5, -6], [S.LE] * 3)


def _instances():
    rng = random.Random(4321)
    out = [_random_lp(rng) for _ in range(60)] + [S.generate("degenerate", 8, 6, 0.5, s) for s in range(6)] + [S.generate("mixed", 7, 5, 0.5, s) for s in range(6)]
    return out + [_beale(), S.textbook_instance(), S.degenerate_instance(), S.infeasible_instance(), S.unbounded_instance()]


@pytest.mark.parametrize("rule", ["dantzig", "bland"])
def test_every_pivot_is_what_the_rule_prescribes_in_exact_arithmetic(rule):
    checked = 0
    for inst in _instances():
        res = A.tableau_simplex(inst, rule=rule, keep=True)
        errs = _pivot_errors(inst, res, ("dantzig", "bland") if res.switched_to_bland else (rule,))
        assert not errs, errs[:3]
        checked += len(res.pivots)
    assert checked > 150


def test_beale_cycles_under_dantzig_and_the_emergency_brake_ends_it():
    res = A.tableau_simplex(_beale(), keep=True)
    assert res.status == "optimal" and res.switched_to_bland and res.obj == pytest.approx(1.25) and res.x == pytest.approx((1.0, 0.0, 1.0, 0.0))
    assert res.total_pivots > 6 and sum(p.degenerate for p in res.pivots) >= 2 * (3 + 7)                  # mehr als ein voller Zyklus Nullschritte, dann Bland
    assert not _pivot_errors(_beale(), res, ("dantzig", "bland"))


def test_status_optimum_and_certificates_agree_with_highs_on_random_small_lps():
    pytest.importorskip("scipy")
    from tests.test_scenario import _highs, reference_status
    rng = random.Random(8)
    seen = set()
    for _ in range(150):
        inst = _random_lp(rng)
        ref = reference_status(inst)
        seen.add(ref)
        for rule in ("dantzig", "bland"):
            res = A.tableau_simplex(inst, rule=rule)
            assert res.status == ref
            if ref == "optimal":
                b = np.array(inst.b)
                assert res.obj == pytest.approx(-_highs(inst).fun, rel=1e-6, abs=1e-6)
                assert A.primal_violation(inst, res.x) < 1e-6 and A.dual_violation(inst, res.duals) < 1e-6
                assert float(np.dot(res.duals, b)) == pytest.approx(res.obj, rel=1e-6, abs=1e-6)
    assert seen == {"optimal", "infeasible", "unbounded"}


def test_redundant_rows_equal_the_rank_deficiency_of_the_constraint_matrix():
    rng = random.Random(31)
    positive = 0
    for _ in range(80):
        n = rng.randint(2, 5)
        x0 = [rng.choice([0, 1, 2, 3]) for _ in range(n)]
        rows = [[rng.choice([-1, 0, 1, 2]) for _ in range(n)] for _ in range(rng.randint(1, 3))]
        b = [sum(a * x for a, x in zip(r, x0)) for r in rows]
        for _ in range(rng.randint(1, 3)):                                                # Linearkombinationen bestehender Gleichungen
            i, j, f, g = rng.randrange(len(rows)), rng.randrange(len(rows)), rng.choice([1, 2, -1, 3]), rng.choice([0, 1, -2])
            rows.append([f * rows[i][t] + g * rows[j][t] for t in range(n)]), b.append(f * b[i] + g * b[j])
        inst = _custom(rows, b, [rng.choice([-1, 0, 1, 2]) for _ in range(n)], [S.EQ] * len(rows))
        res = A.tableau_simplex(inst)
        assert res.status != "infeasible"                                                 # b = A x0: zulässig
        assert res.redundant_rows == len(rows) - np.linalg.matrix_rank(np.array(rows, dtype=float))
        positive += res.redundant_rows > 0
    assert positive > 20


def test_vertices_of_two_service_instances_equal_exact_intersections():
    rng = random.Random(9)
    total = 0
    for _ in range(150):
        m = rng.randint(1, 5)
        rows = [[rng.choice([-2, -1, 0, 1, 2, 3]) for _ in range(2)] for _ in range(m)]
        b = [rng.choice([0, 1, 2, 4, 6, 8]) for _ in range(m)]
        sen = [rng.choice([S.LE, S.LE, S.GE, S.EQ]) for _ in range(m)]
        inst = _custom(rows, b, [1, 1], sen)
        lines = [([F(a) for a in rows[i]], F(b[i])) for i in range(m)] + [([F(1), F(0)], F(0)), ([F(0), F(1)], F(0))]
        exact = set()
        for (a1, r1), (a2, r2) in itertools.combinations(lines, 2):
            det = a1[0] * a2[1] - a1[1] * a2[0]
            if det == 0:
                continue
            x, y = (r1 * a2[1] - r2 * a1[1]) / det, (a1[0] * r2 - a2[0] * r1) / det
            feasible = x >= 0 and y >= 0 and all(_holds(rows[i][0] * x + rows[i][1] * y, sen[i], b[i]) for i in range(m))
            if feasible:
                exact.add((float(x), float(y)))
        got = A.vertices_2d(inst)
        assert len(got) == len(exact) and {(round(x, 9), round(y, 9)) for x, y in got} == {(round(x, 9), round(y, 9)) for x, y in exact}
        total += len(exact)
    assert total > 150
