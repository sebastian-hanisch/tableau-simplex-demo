"""Auswertung: Analyse, Selbstprüfung, Kennzahlen über feste Instanzen, Sweeps, Pivotkurve."""

import math
import pytest

import tsx_constants as C
import tsx_evaluation as ev
import tsx_scenario as S


def test_analyse_of_the_textbook_is_consistent():
    a = ev.analyse(ev.Settings())
    assert a.status == "optimal" and (a.m, a.n, a.pivots) == (3, 2, 2) and a.res.obj == 36.0 and len(a.vertices) == 5 and a.phase1_share == 0.0
    cert = a.certificate
    assert cert == {"primal": 0.0, "dual": 0.0, "gap": 0.0}
    assert ev.analyse(ev.Settings()) is ev.analyse(ev.Settings())                         # zwischengespeichert


def test_certificate_is_none_without_an_optimum_and_tiny_otherwise():
    assert ev.analyse(ev.Settings("infeasible")).certificate is None and ev.analyse(ev.Settings("unbounded")).certificate is None
    for kind in ("random", "mixed", "degenerate"):
        c = ev.analyse(ev.Settings(kind, 12, 9, 0.5, 4)).certificate
        assert c["primal"] < 1e-8 and c["dual"] < 1e-8 and c["gap"] < 1e-8


def test_vertices_only_for_two_services():
    assert ev.analyse(ev.Settings("random", 6, 2, 0.7, 3)).vertices and ev.analyse(ev.Settings("random", 6, 3, 0.7, 3)).vertices == []


def test_instance_of_matches_generate():
    s = ev.Settings("mixed", 7, 5, 0.3, 9)
    assert ev.instance_of(s) == S.generate("mixed", 7, 5, 0.3, 9)


def test_metrics_fields_and_relations():
    r = ev.metrics(S.generate("mixed", 10, 8, 0.5, 1))
    assert r["status"] == "optimal" and r["pivots_per_m"] == pytest.approx(r["pivots"] / 10) and 0 <= r["phase1_share"] <= 1 and r["flops"] == pytest.approx(r["pivots"] * r["flops_per_pivot"])
    assert r["flops_per_pivot"] == (r["n_cols"] + 1) * (1 + 2 * 10)
    u = ev.metrics(S.unbounded_instance())
    assert u["status"] == "unbounded" and u["pivots"] == 1


def test_run_config_bands_are_ordered_and_deterministic():
    r = ev.run_config(ev.Settings("random", 10, 10, 0.5, 0))
    assert r["n_runs"] == 5 and r["solved"] == 5
    for key in ("pivots", "pivots_per_m", "flops_per_pivot", "phase1_share", "tie_share"):
        assert r[f"{key}_lo"] <= r[key] + 1e-12 and r[key] <= r[f"{key}_hi"] + 1e-12
    assert ev.run_config(ev.Settings("random", 10, 10, 0.5, 0)) == r
    assert r["bland"] == 0 and r["phase1_share"] == 0.0


def test_run_config_of_fixtures_ignores_the_seed_and_reports_unsolved_ones():
    r = ev.run_config(ev.Settings("textbook"))
    assert r["pivots"] == 2 and r["pivots_lo"] == r["pivots_hi"] == 2
    bad = ev.run_config(ev.Settings("infeasible"))
    assert bad["solved"] == 0 and math.isnan(bad["pivots"])


def test_sweeps_cover_every_parameter():
    base = ev.Settings("random", 8, 8, 0.5, 0)
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    for param, values in (("m", (4, 8)), ("n", (4, 8)), ("density", (0.3, 0.7)), ("kind", ("random", "mixed"))):
        rows = ev.sweep(param, base, values)
        assert [r["value"] for r in rows] == list(values) and all(r["solved"] == 5 for r in rows)
    assert len(ev.sweep("m", base)) == len(C.SIZE_SWEEP)


def test_pivot_curve_shape_and_determinism():
    rows = ev.pivot_curve("random", sizes=(5, 10), seeds=range(200000, 200008))
    assert [r["size"] for r in rows] == [5, 10] and rows[0]["pivots_lo"] <= rows[0]["pivots"] <= rows[0]["pivots_hi"]
    assert rows[1]["pivots"] > rows[0]["pivots"] and rows == ev.pivot_curve("random", sizes=(5, 10), seeds=range(200000, 200008))
    assert rows[1]["flops_per_pivot"] == (2 * 10 + 1) * (1 + 2 * 10)


def test_zero_step_pivots_exist_so_the_column_is_not_dead():
    a = ev.analyse(ev.Settings("mixed", 5, 2, 0.7, 0))
    assert a.res.degenerate_pivots >= 1
    assert any(ev.metrics(S.generate("mixed", 10, 8, 0.5, s))["degenerate_share"] > 0 for s in range(200000, 200060))

