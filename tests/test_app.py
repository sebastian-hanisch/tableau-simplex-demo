"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt und jede Position der Schritt-Regler, alle Instanztypen, Randwerte, Würfel, Permalink-Grenzen, Instanzwechsel, Experimente und Sweeps auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import tsx_constants as C
import tsx_scenario as S

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=1, **state):
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if step != 1:
        at.select_slider(key="tsx_step").set_value(step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m for m in at.metric if m.label.startswith(label))


def test_default_run_shows_the_textbook_numbers():
    at = _run()
    _ok(at)
    assert {"Pivots", "Pivots je Ressource", "Zielwert", "Operationen"} <= {m.label for m in at.metric}
    assert _metric(at, "Pivots").value == "2" and _metric(at, "Zielwert").value == "36.00" and _metric(at, "Operationen").value == "84" and _metric(at, "Pivots je Ressource").value == "0.67"


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    ss = at.session_state
    assert (ss["kind_select"], ss["tsx_step"]) == (p["kind"], p["step"])
    if p["kind"] not in S.FIXTURE_KINDS:
        assert (ss["m_slider"], ss["n_slider"], ss["density_select"], ss["seed_input"]) == (p["m"], p["n"], p["density"], p["seed"])
    if p["pivot_i"]:
        assert ss["pivot_i"] == p["pivot_i"]
    if p["tab_i"]:
        assert ss["tab_i"] == p["tab_i"]
    assert at.metric and at.get("plotly_chart") or p["step"] == 2


@pytest.mark.parametrize("step", [1, 2, 3, 4])
@pytest.mark.parametrize("kind", list(S.KINDS))
def test_every_step_runs_for_every_kind(step, kind):
    at = _run(kind_select=kind, m_slider=8, n_slider=5, tsx_step=step)
    _ok(at)
    assert at.session_state["tsx_step"] == step


def test_pivot_slider_walks_through_the_path_in_two_and_more_dimensions():
    at = _run(kind_select="mixed", m_slider=5, n_slider=2, density_select=0.7, seed_input=0)
    _ok(at)
    pmax = int(at.slider(key="pivot_i").max)
    assert pmax == 5
    for i in (0, 1, 3, pmax):
        at.slider(key="pivot_i").set_value(i).run()
        _ok(at)
        assert any(x.value.startswith("**Start:**" if i == 0 else f"**Pivot {i} von {pmax}**") for x in at.markdown)
    at2 = _run(kind_select="random", m_slider=8, n_slider=6)
    _ok(at2)
    assert any("braucht genau zwei Dienste" in i.value for i in at2.info)
    at2.slider(key="pivot_i").set_value(int(at2.slider(key="pivot_i").max)).run()
    _ok(at2)


def test_tableau_slider_walks_through_every_tableau():
    at = _run(step=2, kind_select="mixed", m_slider=6, n_slider=4, seed_input=3)
    _ok(at)
    tmax = int(at.slider(key="tab_i").max)
    assert tmax >= 4
    for j in range(tmax + 1):
        at.slider(key="tab_i").set_value(j).run()
        _ok(at)
        assert any(x.value.startswith(f"**Tableau {j} von {tmax}**") for x in at.markdown)
    assert any("**optimal**" in x.value for x in at.markdown)
    at.session_state["kind_select"] = "textbook"
    at.run()
    _ok(at)
    assert at.session_state["tab_i"] <= int(at.slider(key="tab_i").max) == 2


def test_status_messages_for_infeasible_and_unbounded():
    at = _run(kind_select="infeasible")
    _ok(at)
    assert any("Unzulässig" in e.value for e in at.error)
    at2 = _run(kind_select="unbounded", step=2)
    _ok(at2)
    assert any("Unbeschränkt" in w.value for w in at2.warning) and any("**unbeschränkt**" in x.value for x in at2.markdown)
    at3 = _run(kind_select="infeasible", step=2)
    _ok(at3)
    assert any("**unzulässig**" in x.value for x in at3.markdown)


def test_phase_three_explains_when_phase_one_is_not_needed():
    at = _run(step=3)
    _ok(at)
    assert any("Phase 1 ist hier nicht nötig" in i.value for i in at.info)
    at2 = _run(step=3, kind_select="mixed", m_slider=10, n_slider=6, seed_input=2)
    _ok(at2)
    assert any(x.value.startswith("**Phase 1:**") for x in at2.markdown)


@pytest.mark.parametrize("kw", [
    dict(kind_select="random", m_slider=C.M_MIN, n_slider=C.N_MIN), dict(kind_select="random", m_slider=C.M_MAX, n_slider=C.N_MAX), dict(kind_select="random", m_slider=C.M_MAX, n_slider=C.N_MIN),
    dict(kind_select="random", m_slider=C.M_MIN, n_slider=C.N_MAX), dict(kind_select="mixed", m_slider=C.M_MAX, n_slider=30), dict(kind_select="degenerate", m_slider=C.M_MAX, n_slider=C.N_MAX),
    dict(kind_select="random", m_slider=10, n_slider=10, density_select=C.DENSITY_OPTIONS[0]), dict(kind_select="random", m_slider=10, n_slider=10, density_select=C.DENSITY_OPTIONS[-1]),
])
def test_extreme_settings_run(kw):
    for step in (1, 2, 3, 4):
        _ok(_run(step=step, **kw))


def test_dice_button_changes_the_seed():
    at = _run(kind_select="random")
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(m="999", n="1", density="0.42", step="9", kind="nope", seed="-4").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["m_slider"], ss["n_slider"], ss["density_select"], ss["tsx_step"], ss["kind_select"], ss["seed_input"]) == (C.M_MAX, C.N_MIN, C.DEFAULT_DENSITY, 1, "textbook", 0)


def test_permalink_accepts_valid_values():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(kind="mixed", m="12", n="7", density="0.3", seed="7", step="3").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["kind_select"], ss["m_slider"], ss["n_slider"], ss["density_select"], ss["seed_input"], ss["tsx_step"]) == ("mixed", 12, 7, 0.3, 7, 3)


def test_sidebar_shows_the_random_controls_only_for_random_kinds():
    fixed = _run()
    assert not any(w.key == "m_widget" for w in fixed.slider) and not any(w.key == "density_widget" for w in fixed.select_slider) and not any(n.key == "seed_widget" for n in fixed.number_input)
    rnd = _run(kind_select="random")
    assert any(w.key == "m_widget" for w in rnd.slider) and any(w.key == "n_widget" for w in rnd.slider) and any(w.key == "density_widget" for w in rnd.select_slider)
    assert any(n.key == "seed_widget" for n in rnd.number_input)


def test_changing_the_instance_while_on_later_steps_does_not_crash():
    for step in (2, 3, 4):
        at = _run(step=step, kind_select="random")
        _ok(at)
        for kw in (dict(kind_select="textbook"), dict(kind_select="mixed", m_slider=6, n_slider=2), dict(kind_select="infeasible"), dict(kind_select="degenerate", m_slider=15, n_slider=15), dict(kind_select="unbounded")):
            for k, v in kw.items():
                at.session_state[k] = v
            at.run()
            _ok(at)


def test_pivot_curve_runs_on_demand():
    at = _run(kind_select="random")
    next(b for b in at.button if b.key == "curve_start").click().run()
    _ok(at)
    assert any("Median" in c.value and "Pivots" in c.value for c in at.caption)


@pytest.mark.parametrize("param", ["m", "n", "density", "kind"])
@pytest.mark.parametrize("metric", ["pivots", "per_m", "phase1", "ties", "flops"])
def test_sweeps_run_on_demand_for_every_metric(param, metric):
    at = _run(kind_select="random", m_slider=8, n_slider=8, sweep_metric=metric)
    at.selectbox(key="sweep_select").set_value(param).run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_fixed_instances_have_no_experiments():
    for kind in S.FIXTURE_KINDS:
        at = _run(kind_select=kind)
        assert not any(b.key in ("curve_start", "sweep_start") for b in at.button)


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Dantzig, G. B. (1951)" in m.value and "Klee, V., & Minty, G. J. (1972)" in m.value for m in at.markdown)
