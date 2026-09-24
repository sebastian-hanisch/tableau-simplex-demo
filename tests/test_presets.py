"""Presets: gültige Werte, Bänder (Median über die 5 festen Instanzen) und jede Zahl im Hilfetext gegen die echten Auswertungsfunktionen."""

import pytest

import tsx_constants as C
import tsx_evaluation as ev
from tsx_presets import PRESET_KEYS, SETTING_SPECS, STEP_SLIDERS


def _settings(p):
    return ev.Settings(p["kind"], p["m"], p["n"], p["density"], p["seed"])


def _analysis(name):
    return ev.analyse(_settings(C.PRESETS[name]))


def _has(name, *values):
    text = C.PRESET_HELP[name]
    for v in values:
        assert v in text, (name, v)


def test_every_preset_has_valid_values_and_a_help_text():
    assert list(C.PRESETS) == list(C.PRESET_HELP) and len(C.PRESETS) == 10
    for name, p in C.PRESETS.items():
        assert set(p) == set(PRESET_KEYS), name
        for key, state_key in PRESET_KEYS.items():
            if state_key in STEP_SLIDERS:
                assert isinstance(p[key], int) and p[key] >= 0
                continue
            spec = SETTING_SPECS[state_key]
            assert spec.caster(p[key]) == p[key], (name, key)
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
        assert C.PRESET_HELP[name].strip(), name
        a = _analysis(name)
        if p["pivot_i"]:
            assert p["step"] == 1 and p["pivot_i"] <= a.pivots
        if p["tab_i"]:
            assert p["step"] == 2 and p["tab_i"] < len(a.res.snapshots)


@pytest.mark.parametrize("name", list(C.PRESET_EXPECTED_BANDS))
def test_preset_bands_over_the_five_fixed_instances(name):
    metric, lo, hi = C.PRESET_EXPECTED_BANDS[name]
    assert lo <= ev.run_config(_settings(C.PRESETS[name]))[metric] <= hi


def test_help_standard():
    a = _analysis("Standardfall (Lehrbuchbeispiel)")
    assert a.res.obj == 36.0 and [p.x for p in a.res.pivots] == [(0.0, 6.0), (2.0, 6.0)] and a.res.flops == 84 and a.res.duals == (0.0, 1.5, 1.0)
    _has("Standardfall (Lehrbuchbeispiel)", "(0, 0) über (0, 6) nach (2, 6)", "nach 2 Pivots", "36", "84 Rechenoperationen", "0, 1.5 und 1")


def test_help_two_pivots_by_hand():
    a = _analysis("Zwei Pivots von Hand")
    p1, p2 = a.res.pivots
    T1 = a.res.snapshots[1]["T"]
    col = T1[:3, p2.enter]
    assert (p1.ratio, p1.obj, p2.obj) == (6.0, 30.0, 36.0) and sorted(T1[i, -1] / col[i] for i in range(3) if col[i] > 0) == [2.0, 4.0]
    _has("Zwei Pivots von Hand", "Quotienten 6", "von 0 auf 30", "Quotienten 4 und 2", "Zielwert 36")


def test_help_degenerate_corner():
    a = _analysis("Entartete Ecke")
    p1, p2 = a.res.pivots
    assert (p2.ties, p2.ratio, p2.zero_basics >= 1, a.res.degenerate_pivots, a.pivots) == (2, 2.0, True, 0, 2) and a.res.duals == (0.0, 1.5, 1.0, 0.0)
    _has("Entartete Ecke", "x1 + x2 ≤ 8", "Quotienten 2", "2 Pivots", "0, 1.5, 1 und 0")


def test_help_phase1_in_the_picture():
    a = _analysis("Phase 1 in der Zeichnung")
    r = a.res
    assert (a.m, a.n, r.total_pivots, r.phase1_pivots) == (5, 2, 5, 3) and r.pivots[0].degenerate and not r.pivots[0].feasible and r.pivots[2].feasible
    assert r.x[0] == pytest.approx(0.0, abs=1e-9) and r.x[1] == pytest.approx(23.1, abs=0.05) and r.obj == pytest.approx(190.35, abs=0.005)
    _has("Phase 1 in der Zeichnung", "5 Ressourcen", "3 Pivots", "der erste ist ein Nullschritt", "2 Pivots in Phase 2", "(0, 23.1)", "190.35")


def test_help_phase1_in_the_objective():
    a = _analysis("Phase 1 im Zielwert")
    r = a.res
    assert (r.total_pivots, r.phase1_pivots, r.total_pivots - r.phase1_pivots) == (31, 8, 23) and a.phase1_share * 100 == pytest.approx(25.8, abs=0.05) and r.obj == pytest.approx(922.97, abs=0.005)
    _has("Phase 1 im Zielwert", "8 von 31 Pivots (25.8 %)", "23 zur Phase 2", "922.97")


def test_help_infeasible_and_unbounded():
    a = _analysis("Unzulässig")
    assert a.status == "infeasible" and a.pivots == 1 and a.res.phase1_pivots == 1 and a.vertices == []
    _has("Unzulässig", "x1 ≤ 4", "x1 ≥ 6", "nach 1 Pivot", "unzulässig")
    b = _analysis("Unbeschränkt")
    assert b.status == "unbounded" and b.pivots == 1
    _has("Unbeschränkt", "x1 − x2 ≤ 2", "−x1 + x2 ≤ 3", "Nach 1 Pivot", "unbeschränkt")


def test_help_random_and_large():
    a = _analysis("Zufallsinstanz")
    assert (a.pivots, a.res.flops_per_pivot, a.res.flops, a.res.phase1_pivots) == (5, 441.0, 2205, 0) and a.res.obj == pytest.approx(314.21, abs=0.005)
    assert all(p.ties == 1 for p in a.res.pivots)
    _has("Zufallsinstanz", "5 Pivots (0.5 je Ressource)", "441 Rechenoperationen", "2205", "314.21")
    b = _analysis("Große Instanz")
    assert (b.pivots, b.res.flops_per_pivot, b.res.flops) == (52, 14641.0, 761332) and b.pivots / 60 == pytest.approx(0.87, abs=0.005) and ev.run_config(_settings(C.PRESETS["Große Instanz"]))["pivots"] == 46
    assert (b.res.m + 1, b.res.n_cols + 1) == (61, 121)
    _has("Große Instanz", "52 Pivots (0.87 je Ressource", "Median der fünf festen Instanzen 46", "14 641", "761 332", "61 x 121")


def test_help_ties_in_random():
    a = _analysis("Gleichstände im Zufall")
    ties = [p.ties for p in a.res.pivots]
    assert (a.pivots, sum(t > 1 for t in ties), max(ties), a.res.degenerate_pivots) == (9, 4, 4, 0) and a.res.obj == pytest.approx(477.66, abs=0.005)
    _has("Gleichstände im Zufall", "4 von 9 Pivots", "bis zu 4 Zeilen", "9 Pivots", "477.66")
