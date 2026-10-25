"""Tableau-Simplex – von Ecke zu Ecke, Pivot für Pivot - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Erstes Stück (Wurzel) der Lineare-Programmierung-Reihe der "Konzepte"-Reihe: ein Distributionszentrum plant seine Auslastung als Lineares Programm. Der Simplex-Algorithmus läuft mit einem dichten Tableau von Ecke zu Ecke der
Zulässigkeitsmenge: Geometrie bei zwei Diensten, das Tableau Pivot für Pivot, der Zwei-Phasen-Start und der Aufwand.

Lauffähig mit: streamlit run app.py
"""

from dataclasses import replace

import numpy as np
import streamlit as st

import tsx_algorithm as A
import tsx_constants as C
import tsx_evaluation as ev
import tsx_scenario as S
from tsx_evaluation import SWEEP_LABELS, Settings, analyse
from tsx_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    store_from_widget,
    sync_query_params,
)
from tsx_visualization import (
    build_basis_bars,
    build_geometry,
    build_pivot_curve,
    build_progress,
    build_step_sizes,
    build_sweep,
    style_tableau,
    tableau_frame,
)

st.set_page_config(page_title="Tableau-Simplex – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _curve(kind):
    return ev.pivot_curve(kind)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return ev.sweep(param, base)


def num(x, digits=2):
    return f"{x:,.{digits}f}".replace(",", " ") if abs(x) >= 10000 else f"{x:.{digits}f}"


def thousands(x):
    return f"{int(round(x)):,}".replace(",", " ")


st.title("📐 Tableau-Simplex – von Ecke zu Ecke")
st.markdown(
    """
**Erstes Stück der Lineare-Programmierung-Reihe.** Ein Distributionszentrum plant seine Auslastung: welche Dienste (Express-Pakete, Palettenversand, ...) in welcher Menge, damit der Deckungsbeitrag maximal wird und keine knappe
Ressource (Rampenzeit, Kommissionierstunden, Lagerfläche) überschritten wird. Das ist ein **Lineares Programm**. Der **Simplex-Algorithmus** löst es, indem er von Ecke zu Ecke der Zulässigkeitsmenge läuft: jede Ecke ist eine
*Basis*, jeder **Pivot** tauscht eine Variable in ihr aus und verbessert (oder hält) den Zielwert. Vier Fragen, alle gemessen: **(1) Geometrie** - wo läuft er entlang? **(2) Tableau** - was passiert bei einem Pivot? **(3) Zwei Phasen** - was, wenn der
Ursprung nicht zulässig ist? **(4) Aufwand** - wie viele Pivots braucht er, und was kostet jeder?
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/lp-simplex-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Stellen Sie sich ein Distributionszentrum vor, das mehrere Dienste anbietet. Jeder Dienst bringt Gewinn, braucht aber knappe Mittel wie Rampenzeit, Kommissionierstunden und Lagerfläche. Gesucht ist die Mischung, die den Gewinn maximiert, ohne einen Vorrat zu überschreiten. Das ist ein **Lineares Programm**. Zeichnet man bei zwei Diensten alle erlaubten Mengen ein, ergibt sich eine Fläche mit Ecken: der **zulässige Bereich**.

Weil der Gewinn gleichmäßig mit den Mengen wächst, liegt der beste Punkt in einer Ecke dieser Fläche. Der **Simplex-Algorithmus** startet in einer Ecke, geht zu einer Nachbarecke mit mehr Gewinn und hört auf, wenn keine Nachbarecke besser ist. Jeder dieser Schritte heißt **Pivot**. Die Demo zeigt den Weg in der Zeichnung und die Rechentabelle (das **Tableau**), die dabei bei jedem Pivot neu entsteht.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/lp-simplex-erklaert.html)
        """
    )
# erklaerseite:einfach:end
st.caption(
    "Wurzel der Reihe. Die Verfahren der Folgestücke (Pivotregeln, Revised Simplex, Dualität, Innere Punkte, PDLP) sind inzwischen als eigene Demos gebaut. Den Netzwerksimplex, eine Spezialform mit Spannbaum als Basis, zeigt "
    "[network-flow-demo](https://github.com/sebastian-hanisch/network-flow-demo)."
)

with st.expander("So funktioniert der Simplex", expanded=True):
    st.markdown(
        """
1. **Standardform:** Jede Bedingung bekommt eine **Schlupfvariable** (bei ≤) bzw. eine **Überschussvariable** (bei ≥), aus Ungleichungen werden Gleichungen. Ist der Ursprung zulässig (alle ≤ mit b ≥ 0), bilden die Schlupfvariablen die Start-Basis.
2. **Tableau:** eine Zeile je Bedingung, dazu die Zielzeile mit den **reduzierten Kosten** r = z − c: eine Spalte mit r < 0 würde den Zielwert erhöhen. Die Basisvariablen stehen als Einheitsvektoren im Tableau.
3. **Pivot:** Die Spalte mit dem kleinsten r **tritt ein** (Dantzig-Regel). Der **Quotiententest** rhs / Spaltenwert (nur positive Einträge) bestimmt, welche Basisvariable **austritt**, bevor eine Bedingung verletzt wird. Dann wird das Tableau mit dem
   Pivotelement eliminiert. Alle r ≥ 0: **optimal**. Kein positiver Eintrag in der Spalte: **unbeschränkt**.
4. **Zwei Phasen:** Ist der Ursprung nicht zulässig (Bedingungen mit ≥ oder =), löst **Phase 1** ein Hilfsproblem mit **künstlichen Variablen** (Summe minimieren); endet sie bei 0, ist eine zulässige Ecke gefunden, sonst ist die Instanz **unzulässig**. Phase 2 optimiert dann den Zielwert.
5. **Entartung:** Ein Gleichstand im Quotiententest heißt, dass zwei Bedingungen an derselben Ecke bindend werden; danach ist eine Basisvariable 0 (entartete Ecke).
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:4], preset_names[4:7], preset_names[7:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

ss = st.session_state
with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.selectbox("Instanz", options=list(S.KINDS), format_func=lambda v: S.KIND_LABELS[v], key="kind_select",
                        help="Lehrbuchbeispiel und die drei Sonderfälle sind fest (2 Dienste, von Hand nachrechenbar); die drei Zufallstypen sind regelbar.")
    random_kind = kind not in S.FIXTURE_KINDS
    if random_kind:
        m = st.slider("Ressourcen m", *bounds("m_slider"), value=int(ss["m_slider"]), key="m_widget", on_change=store_from_widget, args=("m_slider",), help="Zahl der Bedingungen (knappe Ressourcen).")
        n = st.slider("Dienste n", *bounds("n_slider"), value=int(ss["n_slider"]), key="n_widget", on_change=store_from_widget, args=("n_slider",),
                      help="Zahl der Entscheidungsvariablen. Bei n = 2 gibt es die Zeichnung der Zulässigkeitsmenge.")
        density = st.select_slider("Dichte der Matrix", options=list(C.DENSITY_OPTIONS), value=float(ss["density_select"]), key="density_widget", on_change=store_from_widget, args=("density_select",),
                                   help="Anteil der Dienste, die eine Ressource verbrauchen.")
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), value=int(ss["seed_input"]), key="seed_widget", step=1, on_change=store_from_widget, args=("seed_input",))
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        m, n, density, seed = C.DEFAULT_M, C.DEFAULT_N, C.DEFAULT_DENSITY, C.DEFAULT_SEED

sync_query_params({"kind_select": kind, "m_slider": int(ss["m_slider"]), "n_slider": int(ss["n_slider"]), "density_select": float(ss["density_select"]), "seed_input": int(ss["seed_input"]),
                   "tsx_step": int(ss["tsx_step"])})

settings = Settings(kind, int(m), int(n), float(density), int(seed))
with st.spinner("Rechne..."):
    a = analyse(settings)
inst, res = a.inst, a.res
names = list(res.col_names)

# --- In Aktion ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ecke für Ecke")
step = st.select_slider("Schritt", options=list(C.STEPS), key="tsx_step", format_func=lambda s: C.STEPS[s])

if a.status == "infeasible":
    st.error("**Unzulässig:** Phase 1 endet mit positiver Summe der künstlichen Variablen - die Bedingungen widersprechen sich, es gibt keine zulässige Ecke.")
elif a.status == "unbounded":
    st.warning("**Unbeschränkt:** Die eintretende Spalte hat keinen positiven Eintrag, es gibt keinen Quotienten - der Zielwert wächst ohne Grenze.")


def var_label(j):
    return names[j]


if step == 1:
    total = len(res.pivots)
    if "pivot_i" in ss:
        ss["pivot_i"] = min(max(0, int(ss["pivot_i"])), total)
    k = st.slider("Pivot", 0, total, key="pivot_i", help="0 = Start; jeder weitere Schritt ist ein Pivot.") if total > 0 else 0
    if k == 0:
        line = "**Start:** Basislösung im Ursprung" + (" (zulässig: alle Bedingungen sind mit Schlupf erfüllt)." if res.phase1_pivots == 0 else " - er ist **nicht zulässig**, Phase 1 sucht zuerst eine zulässige Ecke.")
    else:
        p = res.pivots[k - 1]
        z = f"Zielwert {num(p.obj)}" if p.phase == 2 else f"Summe der künstlichen Variablen {num(-p.obj)}"
        line = (f"**Pivot {k} von {total}** (Phase {p.phase}): {var_label(p.enter)} tritt ein, {var_label(p.leave_var)} verlässt die Basis, Schrittweite {num(p.ratio)}" + (" (Nullschritt: die Ecke ändert sich nicht)" if p.degenerate else "") +
                f"; {z}.")
    st.markdown(line)
    if inst.n == 2:
        st.plotly_chart(build_geometry(inst, res, a.vertices, k), width="stretch", key=f"s1_geo_{k}")
        st.caption("Grüne Fläche = zulässig, weiße Kreise = Ecken, orange = Simplex-Pfad in zulässigen Ecken, grau gepunktet = Phase 1 (durch unzulässige Basislösungen), rot gepunktet = Höhenlinie des Zielwerts durch die aktuelle Basislösung. "
                   "Jede Basis ist eine Ecke, jeder Pivot läuft eine Kante entlang.")
    else:
        st.info(f"Die Zeichnung der Zulässigkeitsmenge braucht genau zwei Dienste (hier {inst.n}); für höhere Dimensionen zeigt der Schritt die Basislösung nach jedem Pivot. Stellen Sie n = 2 ein, um die Ecken zu sehen.")
        st.plotly_chart(build_basis_bars(inst, res, k), width="stretch", key=f"s1_bars_{k}")
elif step == 2:
    snaps = res.snapshots
    if "tab_i" in ss:
        ss["tab_i"] = min(max(0, int(ss["tab_i"])), len(snaps) - 1)
    j = st.slider("Tableau", 0, len(snaps) - 1, key="tab_i", help="0 = Start-Tableau; danach eines je Pivot (bei Zwei-Phasen-Instanzen zusätzlich eines am Beginn von Phase 2).") if len(snaps) > 1 else 0
    snap = snaps[j]
    nxt = snaps[j + 1] if j + 1 < len(snaps) else None
    piv = res.pivots[nxt["pivot"] - 1] if nxt is not None and nxt["pivot"] is not None else None
    T = snap["T"]
    m_rows = T.shape[0] - 1
    df, cell = tableau_frame(snap, res, piv)
    head = f"**Tableau {j} von {len(snaps) - 1}** (Phase {snap['phase']})."
    if piv is not None:
        col = T[:m_rows, piv.enter]
        quotients = [(i, T[i, -1] / col[i]) for i in range(m_rows) if col[i] > A.TOL]
        qtext = ", ".join(f"{df.index[i]}: {num(T[i, -1], 3)} / {num(col[i], 3)} = {num(q, 3)}" for i, q in quotients)
        head += (f" Nächster Pivot: **{var_label(piv.enter)}** tritt ein (reduzierte Kosten {num(T[-1, piv.enter], 3)}, die kleinsten), Quotienten {qtext}; **{var_label(piv.leave_var)}** verlässt die Basis" +
                 (f" (Gleichstand zwischen {piv.ties} Zeilen, kleinster Index gewinnt)" if piv.ties > 1 else "") + ".")
    elif j == len(snaps) - 1:
        head += {"optimal": " Alle reduzierten Kosten sind ≥ 0: **optimal**.", "infeasible": " Phase 1 endet mit positiver Summe der künstlichen Variablen: **unzulässig**.",
                 "unbounded": " Die eintretende Spalte hat keinen positiven Eintrag: **unbeschränkt**."}[res.status]
    st.markdown(head)
    st.dataframe(style_tableau(df, cell), width="stretch", height=min(700, 40 + 35 * len(df)))
    st.caption("Orange = Pivotelement des nächsten Pivots (Spalte: kleinste reduzierte Kosten, Zeile: kleinster Quotient), letzte Zeile = reduzierte Kosten und Zielwert (rechts), erste Spalten = Dienste, dann Schlupf- (s), Überschuss- (e) und künstliche Variablen (a)."
               + (f" Es sind die ersten {len(snaps)} Tableaus gespeichert; die Instanz braucht {res.total_pivots} Pivots." if len(res.pivots) + 2 > len(snaps) + 1 and len(snaps) >= 80 else ""))
elif step == 3:
    if res.phase1_pivots == 0 and a.status != "infeasible":
        st.info("**Phase 1 ist hier nicht nötig:** alle Bedingungen sind ≤ mit b ≥ 0, der Ursprung ist zulässig und die Schlupfvariablen bilden die Start-Basis. Die Mischinstanz und der Sonderfall \"Unzulässig\" zeigen Phase 1.")
    else:
        art_rows = [i + 1 for i in range(inst.m) if inst.senses[i] in (S.GE, S.EQ)]
        st.markdown(f"**Phase 1:** {len(art_rows)} Bedingungen sind ≥ oder = und bekommen je eine künstliche Variable (Zeilen {', '.join(map(str, art_rows[:12]))}{' ...' if len(art_rows) > 12 else ''}); der Simplex minimiert ihre Summe. "
                    f"Das dauert **{res.phase1_pivots}** von {res.total_pivots} Pivots ({a.phase1_share * 100:.1f} %)" + (" und endet bei 0: eine zulässige Ecke ist gefunden, Phase 2 optimiert den Zielwert." if a.status == "optimal" else
                                                                                                                            ", endet aber über 0: die Bedingungen widersprechen sich."))
    st.plotly_chart(build_progress(res), width="stretch", key="s3_progress")
    st.caption("Rot (rechte Achse) = Summe der künstlichen Variablen in Phase 1, fällt auf 0; grün = Zielwert in Phase 2, steigt (bei Entartung bleibt er stehen). Jeder Punkt ist ein Tableau.")
else:
    if res.status == "optimal":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{res.total_pivots} Pivots** ({res.phase1_pivots} in Phase 1, {res.total_pivots - res.phase1_pivots} in Phase 2), **{thousands(res.flops)} Rechenoperationen** im dichten Tableau: je Pivot (Spalten + 1) · (1 + 2 · m) = "
                        f"{res.n_cols + 1} · {1 + 2 * res.m} = **{thousands(res.flops_per_pivot)}**. Das Tableau hat {res.m + 1} Zeilen und {res.n_cols + 1} Spalten, und jeder Pivot rechnet alle Einträge neu - auch die, die sich nicht ändern.")
        with c2:
            cert = a.certificate
            st.markdown(f"**Selbstprüfung ohne fremden Löser:** größte Verletzung einer Bedingung {cert['primal']:.1e}, der Dual-Zulässigkeit {cert['dual']:.1e}; Lücke zwischen Deckungsbeitrag c·x = {num(res.obj)} und b·y = "
                        f"{num(float(np.dot(res.duals, np.array(inst.b))))} ist {cert['gap']:.1e} (starke Dualität: die Schattenpreise y beweisen das Optimum).")
        st.plotly_chart(build_step_sizes(res), width="stretch", key="s4_steps")
        ties = sum(p.ties > 1 for p in res.pivots)
        st.caption(f"Schrittweite je Pivot; rot = Nullschritt (die Ecke ändert sich nicht). Gleichstände im Quotiententest: {ties} von {res.total_pivots} Pivots, Nullschritte: {res.degenerate_pivots}.")
    else:
        st.info("Bei einer unzulässigen oder unbeschränkten Instanz gibt es kein Optimum; der Aufwand bis zur Erkennung: "
                f"{res.total_pivots} Pivot(s), {thousands(res.flops)} Rechenoperationen.")

st.markdown("---")

# --- Kennzahlen --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ⚙️ Wie viele Pivots, wie viel Rechnung?")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Pivots", str(res.total_pivots), delta=f"Phase 1: {res.phase1_pivots}", delta_color="off")
m2.metric("Pivots je Ressource", f"{res.total_pivots / inst.m:.2f}", delta=f"m = {inst.m}", delta_color="off")
m3.metric("Zielwert", num(res.obj) if res.status == "optimal" else "-", delta=res.status, delta_color="off")
m4.metric("Operationen", thousands(res.flops), delta=f"{thousands(res.flops_per_pivot)} je Pivot", delta_color="off")

st.markdown("---")

# --- Experimente auf Abruf ---------------------------------------------------------------------------------------------------------------------

base = replace(settings, seed=0)
if random_kind:
    st.subheader("🔬 Wachsen die Pivots mit der Größe?")
    st.caption("30 Instanzen je Größe (m = n = 5 bis 60) im gewählten Zufallstyp, dichte Matrix wie in der Seitenleiste eingestellt: Median und Band (10. bis 90. Perzentil) der Pivots.")
    if st.button("Pivotkurve berechnen (dauert einige Sekunden)", key="curve_start"):
        ss["curve_done"] = ss.get("curve_done", set()) | {kind}
    if kind in ss.get("curve_done", set()):
        with st.spinner("Rechne..."):
            rows = _curve(kind)
        st.plotly_chart(build_pivot_curve(rows), width="stretch", key="curve_chart")
        last, first = rows[-1], rows[0]
        st.caption(f"Bei m = n = {last['size']}: Median {num(last['pivots'], 1)} Pivots ({last['per_m']:.2f} je Ressource), {thousands(last['flops_per_pivot'])} Operationen je Pivot; bei m = n = {first['size']}: {num(first['pivots'], 1)} Pivots. "
                   f"Anteil Phase 1 (Median): {last['phase1_share'] * 100:.0f} %.")
    st.markdown("---")

    st.subheader("📐 Sweeps")
    sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda v: SWEEP_LABELS[v], key="sweep_select")
    metric_opts = {"pivots": "Pivots", "per_m": "Pivots je Ressource", "phase1": "Anteil Phase 1", "ties": "Gleichstände im Quotiententest", "flops": "Operationen je Pivot"}
    if ss.get("sweep_metric") not in metric_opts:
        ss.pop("sweep_metric", None)
    metric = st.radio("Kennzahl", options=list(metric_opts), format_func=lambda v: metric_opts[v], key="sweep_metric", horizontal=True)
    if st.button("Sweep über 5 feste Instanzen berechnen (kann einige Sekunden dauern)", key="sweep_start"):
        ss["sweep_done"] = ss.get("sweep_done", set()) | {(sweep_param, base)}
    if (sweep_param, base) in ss.get("sweep_done", set()):
        with st.spinner("Rechne den Sweep über 5 feste Instanzen..."):
            rows_w = _sweep(sweep_param, base)
        series = {
            "pivots": ([("pivots", "Pivots", "#2F6B65")], "Pivots"),
            "per_m": ([("pivots_per_m", "Pivots je Ressource", "#2F6B65")], "Pivots je Ressource"),
            "phase1": ([("phase1_share", "Anteil Phase 1", "#d62728")], "Anteil der Pivots"),
            "ties": ([("tie_share", "Gleichstände", "#e8a13a"), ("degenerate_share", "Nullschritte", "#d62728")], "Anteil der Pivots"),
            "flops": ([("flops_per_pivot", "Operationen je Pivot", "#1f4e9c")], "Operationen"),
        }[metric]
        tick = (lambda v: S.KIND_LABELS[v].split(" (")[0]) if sweep_param == "kind" else None
        st.plotly_chart(build_sweep(rows_w, SWEEP_LABELS[sweep_param], series[0], series[1], tick=tick), width="stretch", key="sweep_chart")
        st.caption("Median über 5 feste Instanzen (Seeds 100000–100004), Band = 10. bis 90. Perzentil. Die übrigen Regler stehen wie in der Seitenleiste.")
    st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Simplex braucht viele Pivots** | Auf zufälligen Instanzen (alle Bedingungen ≤) nicht: bei m = n = 5/10/20/40/60 im Median 3/5.5/10.5/20/32.5 Pivots, etwa 0.5 bis 0.6 je Ressource (30 Instanzen je Größe). Mit Mindest- und Gleichungsbedingungen sind es 5/13/38/105.5/199, also 1.0 bis 3.3 je Ressource und je Größe mehr. Der exponentielle schlimmste Fall (Klee-Minty) kommt nicht vor: die Instanzen sind zufällig, nicht gebaut. | Folgestück: schlimmster und typischer Fall |
| **Phase 1 ist ein Randfall** | Nur bei ≥ und = : dann gehören in den Mischinstanzen 25 bis 39 % der Pivots (Median über 30 Instanzen je Größe) zur Phase 1, bei m = n = 60 sind es 25 % von 199 Pivots. Ein Zufall mit ausschließlich ≤ und b ≥ 0 braucht sie nie. | - |
| **Gleichstände führen in den Stillstand** | Nicht in diesen Instanzen: bei Ressourcen, die Summen anderer sind, gibt es im Quotiententest in 16.7/36.4/15.8 % der Pivots (m = n = 10/20/40) einen Gleichstand, aber keinen einzigen Nullschritt. Zyklen sind nicht aufgetreten (die Bland-Regel als Notbremse wurde nie gebraucht). | Folgestück: Pivotregeln und Entartung |
| **Ein Pivot ist billig** | Im dichten Tableau nicht: ein Pivot kostet (Spalten + 1) · (1 + 2 m) Rechenoperationen, bei m = n = 60 sind das 14 641, und jeder Pivot rechnet alle Einträge neu, auch die, die sich nicht ändern. Die Zahl der Operationen wächst mit dem Produkt aus Zeilen und Spalten. | Folgestück: Revised Simplex |
| **Die Zahlen sind exakt** | Nein: Gleitkomma mit fester Toleranz 1e-9, ohne Skalierung oder Präsolve. Auf den erzeugten, gut skalierten Instanzen stimmt jedes Ergebnis mit HiGHS überein (Tests: über 1000 Instanzen); auf schlecht skalierten Daten würde das nicht reichen. | Folgestück: Präsolve und Numerik |
| **Die Instanzen sind realistisch** | Synthetische Auslastungsplanung: nichtnegative Verbrauchskoeffizienten, eine dichte Gesamtkapazität, Zufallswerte. Echte LPs sind dünner, größer und stärker strukturiert (etwa Netzwerke: dort ist der Netzwerksimplex mit Spannbaum-Basis die richtige Form). | [network-flow-demo](https://github.com/sebastian-hanisch/network-flow-demo) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/lp-simplex-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind der Gewinn je Einheit $c$ der Dienste, der Verbrauch $A$ der knappen Ressourcen je Einheit und die Vorräte $b$. Gesucht sind die Mengen $x$ der Dienste, die den Gewinn $c^\top x$ maximieren, ohne einen Vorrat zu überschreiten.

| Symbol | Bedeutung |
|---|---|
| $x$, $c$ | Mengen der Dienste und Gewinn (Deckungsbeitrag) je Einheit |
| $A$, $b$ | Verbrauch je Einheit und Vorrat der knappen Ressourcen |
| $s$ | Schlupfvariable: ungenutzter Rest einer Ressource |
| $B$ | Basis: ausgewählte Spalten, die die aktuelle Ecke festlegen |
| $x_B$, $x_N$ | Werte der Variablen in der Basis bzw. außerhalb (diese sind null) |
| $r_j$ | reduzierte Kosten der Spalte $j$: ist $r_j < 0$, erhöht der Eintritt dieser Spalte den Zielwert |
| $t_{ij}$ | Eintrag des Tableaus in Zeile $i$ und Spalte $j$ |
| $y$ | Schattenpreise der Ressourcen (Dualvariablen) |
| $m$, $N$ | Zahl der Zeilen (Ressourcen) bzw. Spalten des Tableaus |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/lp-simplex-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Lineares Programm (Standardform mit Schlupf).** $\max\; c^\top x$ unter $A x + s = b,\; x, s \ge 0$. Eine **Basis** $B$ ($m$ linear unabhängige Spalten) liefert die **Basislösung** $x_B = B^{-1} b$, $x_N = 0$; sie ist zulässig, wenn $x_B \ge 0$, und
entspricht dann einer **Ecke** des Polyeders $\{x \ge 0 : A x \le b\}$.

**Tableau.** Mit $B^{-1}[A\;|\;b]$ und der Zielzeile $r_j = c_B^\top B^{-1} a_j - c_j$ ist die Basislösung **optimal**, wenn alle $r_j \ge 0$. Ein Pivot mit eintretender Spalte $j$ ($r_j < 0$) und austretender Zeile $i^\* = \arg\min_{i:\,t_{ij} > 0} t_{i,\mathrm{rhs}} / t_{ij}$
(Quotiententest) führt zur Nachbarecke; ist kein $t_{ij} > 0$, ist das Problem **unbeschränkt**.

**Zwei Phasen.** Für jede Bedingung mit $\ge$ oder $=$ eine künstliche Variable $a_i \ge 0$; Phase 1 maximiert $-\sum_i a_i$. Ist das Optimum $< 0$, ist die Instanz **unzulässig**; sonst liefert die Endbasis eine zulässige Ecke für Phase 2.

**Schattenpreise.** Die Zielzeile unter den Schlupfspalten enthält die Dualvariablen $y = c_B^\top B^{-1}$; im Optimum gilt $c^\top x = b^\top y$ (starke Dualität), $y \ge 0$ für Bedingungen mit $\le$.

**Aufwand.** Ein Pivot des dichten Tableaus mit $m$ Zeilen und $N$ Spalten kostet $(N + 1) + 2 m (N + 1)$ Gleitkomma-Operationen (Zeile normieren, $m$ Zeilen inklusive Zielzeile eliminieren); gezählt wird dieses Modell, keine Laufzeit.

**Literatur.** Dantzig, G. B. (1951). *Maximization of a linear function of variables subject to linear inequalities.* In T. C. Koopmans (Hrsg.), Activity Analysis of Production and Allocation, 339-347. Wiley. Klee, V., & Minty, G. J. (1972). *How good is the simplex algorithm?*
In O. Shisha (Hrsg.), Inequalities III, 159-175. Academic Press (Ausblick: exponentieller schlimmster Fall, Folgestück).

Implementiert in `tsx_algorithm.py` (Standardform, Zwei-Phasen-Simplex, Prüfgrößen), `tsx_scenario.py` (Instanzen), `tsx_evaluation.py` (Kennzahlen, Sweeps, Pivotkurve).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Lineare Programmierung: vom Tableau zum Crossover](https://sebastianhanisch.net/konzepte-lineare-programmierung.html)."
)
