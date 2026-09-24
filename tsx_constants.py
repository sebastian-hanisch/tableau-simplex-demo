"""Konstanten der Demo Tableau-Simplex: Regler-Bereiche, feste Instanzen, Presets (Werte aus der Vormessung)."""
M_MIN, M_MAX, DEFAULT_M = 2, 100, 10
N_MIN, N_MAX, DEFAULT_N = 2, 100, 10
DENSITY_OPTIONS = (0.1, 0.3, 0.5, 0.7, 1.0)
DEFAULT_DENSITY = 0.5
SEED_MAX = 999999
DEFAULT_SEED = 35
STEPS = {1: "1 · Geometrie", 2: "2 · Tableau", 3: "3 · Zwei Phasen", 4: "4 · Aufwand"}
SWEEP_SEEDS = tuple(range(100000, 100005))
FEAS_SEEDS = tuple(range(200000, 200030))
SIZE_SWEEP = (5, 10, 20, 40, 60)
_BASE = {"kind": "textbook", "m": 10, "n": 10, "density": 0.5, "seed": 35, "step": 1, "pivot_i": 0, "tab_i": 0}
PRESETS = {
    "Standardfall (Lehrbuchbeispiel)": dict(_BASE),
    "Zwei Pivots von Hand": {**_BASE, "step": 2, "tab_i": 1},
    "Entartete Ecke": {**_BASE, "kind": "fix_degenerate", "step": 2, "tab_i": 1},
    "Phase 1 in der Zeichnung": {**_BASE, "kind": "mixed", "m": 5, "n": 2, "density": 0.7, "seed": 0, "step": 1, "pivot_i": 5},
    "Phase 1 im Zielwert": {**_BASE, "kind": "mixed", "m": 20, "n": 20, "step": 3},
    "Unzulässig": {**_BASE, "kind": "infeasible"},
    "Unbeschränkt": {**_BASE, "kind": "unbounded"},
    "Zufallsinstanz": {**_BASE, "kind": "random", "step": 4},
    "Große Instanz": {**_BASE, "kind": "random", "m": 60, "n": 60, "step": 4},
    "Gleichstände im Zufall": {**_BASE, "kind": "degenerate", "m": 20, "n": 20, "step": 4},
}
PRESET_HELP = {
    "Standardfall (Lehrbuchbeispiel)": "2 Dienste (Express-Pakete, Palettenversand) mit Deckungsbeitrag 3 und 5, 3 Ressourcen. Der Simplex läuft von Ecke (0, 0) über (0, 6) nach (2, 6) und ist nach 2 Pivots fertig: Optimum 36, 84 Rechenoperationen. Die Schattenpreise sind 0, 1.5 und 1: nur Kommissionierstunden und Lagerfläche sind knapp.",
    "Zwei Pivots von Hand": "Nach Pivot 1 ist Palettenversand (x2) in der Basis: die Zeile Kommissionierstunden hatte den kleinsten Quotienten 6, der Zielwert steigt von 0 auf 30. Im zweiten Pivot tritt x1 ein (Quotienten 4 und 2, die Zeile Lagerfläche gewinnt), Zielwert 36.",
    "Entartete Ecke": "Eine vierte Ressource x1 + x2 ≤ 8 läuft durch das Optimum (2, 6): im zweiten Pivot haben zwei Zeilen denselben kleinsten Quotienten 2 (Gleichstand), danach ist eine Basisvariable null, die Ecke ist entartet (drei Bedingungen bindend). Hier bleibt es folgenlos: 2 Pivots, kein Nullschritt, Schattenpreise 0, 1.5, 1 und 0.",
    "Phase 1 in der Zeichnung": "5 Ressourcen mit Mindest- und Gleichungsbedingungen, 2 Dienste: der Ursprung ist unzulässig, Phase 1 braucht 3 Pivots (grau gepunktet, durch unzulässige Basislösungen, der erste ist ein Nullschritt) bis zur ersten zulässigen Ecke, dann 2 Pivots in Phase 2 zum Optimum (0, 23.1) mit Wert 190.35.",
    "Phase 1 im Zielwert": "20 Ressourcen, 20 Dienste, mit Mindest- und Gleichungsbedingungen: 8 von 31 Pivots (25.8 %) gehören zur Phase 1 (die Summe der künstlichen Variablen fällt auf 0), 23 zur Phase 2; Zielwert 922.97.",
    "Unzulässig": "Rampenzeit x1 ≤ 4 und Mindestmenge x1 ≥ 6 widersprechen sich: Phase 1 endet nach 1 Pivot mit positiver Summe der künstlichen Variablen, der Simplex meldet \"unzulässig\" statt eines Optimums; die Zeichnung zeigt keine zulässige Fläche.",
    "Unbeschränkt": "max x1 + x2 unter x1 − x2 ≤ 2 und −x1 + x2 ≤ 3: entlang (1, 1) wächst der Zielwert ohne Grenze. Nach 1 Pivot hat die eintretende Spalte keinen positiven Eintrag, es gibt keinen Quotienten: der Simplex meldet \"unbeschränkt\".",
    "Zufallsinstanz": "10 Dienste, 10 Ressourcen, Dichte 0.5: 5 Pivots (0.5 je Ressource), 441 Rechenoperationen je Pivot und 2205 insgesamt; keine Phase 1, keine Gleichstände; Optimum 314.21.",
    "Große Instanz": "60 Dienste, 60 Ressourcen: 52 Pivots (0.87 je Ressource; Median der fünf festen Instanzen 46), 14 641 Rechenoperationen je Pivot und 761 332 insgesamt: das dichte Tableau (61 x 121) wird bei jedem Pivot komplett neu gerechnet, die Schwäche, die der Revised Simplex behebt.",
    "Gleichstände im Zufall": "Ressourcen, die Summen anderer Ressourcen sind (20 x 20): in 4 von 9 Pivots gibt es Gleichstände im Quotiententest (bis zu 4 Zeilen), aber keinen Nullschritt; 9 Pivots, Optimum 477.66.",
}
# Beobachtete Spannweite der Kennzahl (MEDIAN über die 5 festen Instanzen Seeds 100000-100004) je Preset, mit Sicherheitsabstand: (Kennzahl, untere, obere Grenze).
PRESET_EXPECTED_BANDS = {
    "Standardfall (Lehrbuchbeispiel)": ("pivots", 2, 2),
    "Entartete Ecke": ("pivots", 2, 2),
    "Phase 1 im Zielwert": ("pivots", 25, 42),
    "Zufallsinstanz": ("pivots", 3, 8),
    "Große Instanz": ("pivots", 35, 60),
    "Gleichstände im Zufall": ("tie_share", 0.2, 0.5),
}
