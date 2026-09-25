# Tableau-Simplex – von Ecke zu Ecke, Pivot für Pivot – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-tableau-simplex-demo.streamlit.app/)**

Erstes Stück (Wurzel) der **Lineare-Programmierung-Reihe** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning". Ein Distributionszentrum plant seine Auslastung: welche Dienste (Express-Pakete, Palettenversand, ...) in welcher Menge, damit der Deckungsbeitrag maximal wird und keine knappe Ressource (Rampenzeit, Kommissionierstunden, Lagerfläche) überschritten wird. Das ist ein **Lineares Programm**. Der **Simplex-Algorithmus** löst es mit einem dichten **Tableau**, indem er von Ecke zu Ecke der Zulässigkeitsmenge läuft: jede Ecke ist eine Basis, jeder **Pivot** tauscht eine Variable in ihr aus. Die Demo stellt vier Fragen, alle gemessen: **(1) Geometrie** – wo läuft der Simplex entlang (bei zwei Diensten gezeichnet)? **(2) Tableau** – was passiert bei einem Pivot? **(3) Zwei Phasen** – was, wenn der Ursprung nicht zulässig ist? **(4) Aufwand** – wie viele Pivots braucht er, und was kostet jeder?

**Einordnung in die Reihe:** geplant sind zwölf Stücke, dies ist die Wurzel (Details in der Scoping-Notiz `lp-planung/PLAN.md` des Portfolio-Ordners):

```
Tableau-Simplex (Wurzel: Ecken, Basis, Pivot, Zwei-Phasen-Start)                         [DIESES STÜCK]
 ├─ Pivotregeln & Entartung  ─ Simplex im schlimmsten und im typischen Fall (Klee-Minty)  [nicht gebaut]
 ├─ Revised Simplex ─ Präsolve, Skalierung & Numerik                                     [nicht gebaut]
 ├─ Dualität & Sensitivität ─ Dualer Simplex & Neuoptimierung                            [nicht gebaut]
 ├─ Ellipsoid-Methode (Kontrast: polynomial in der Theorie)                              [nicht gebaut]
 └─ Innere Punkte ─ PDLP (Verfahren erster Ordnung) ─ Crossover & Simplex gegen Innere Punkte gegen PDLP  [nicht gebaut]
```

Ergebnis in Kürze: **Der Simplex braucht auf diesen Instanzen wenige Pivots – aber wie viele, hängt davon ab, ob der Ursprung zulässig ist.** Auf zufälligen Instanzen mit lauter ≤-Bedingungen sind es im Median **3/5.5/10.5/20/32.5** Pivots bei m = n = 5/10/20/40/60, also nur 0.5 bis 0.6 je Ressource (meine Vorab-Hypothese "1 bis 3 je Ressource" ist dort **widerlegt**). Mit Mindest- und Gleichungsbedingungen sind es **5/13/38/105.5/199**, 1.0 bis 3.3 je Ressource und je Größe mehr, und **25 bis 39 %** der Pivots gehören zur Phase 1. Gleichstände im Quotiententest treten bei Ressourcen, die Summen anderer sind, in bis zu **36 %** der Pivots auf, ein Nullschritt (Stillstand) aber nie. Ein Pivot im dichten Tableau kostet (Spalten + 1) · (1 + 2 m) Rechenoperationen, bei m = n = 60 sind das **14 641**: die Schwäche, die der Revised Simplex behebt.

| Frage | Ergebnis (Distributionszentrum, Standard-LP max c·x, A x ≤/≥/= b, x ≥ 0; **Median** über 30 Instanzen, Seeds 200000–200029, bzw. über 5 feste Instanzen, Seeds 100000–100004; vollständig deterministisch, `random.Random` mit Zeichenketten-Seeds) |
|---|---|
| **Stimmt der Löser?** | ✅ Optimum == HiGHS (`scipy.optimize.linprog`, nur als Gegenprobe in den Tests) auf **über 1000 Instanzen**: 720 erzeugte (Zufall, Mischung, Gleichstände), 300 kleine Zufalls-LPs mit gemischten Vorzeichen (dabei alle drei Status), 200 konstruierte unzulässige und unbeschränkte; jedes Ergebnis primal und dual zulässig, starke Dualität c·x = b·y, komplementärer Schlupf. Unzulässig/unbeschränkt werden nie als Optimum ausgegeben (HiGHS meldet bei zulässigen, unbeschränkten LPs gelegentlich "unzulässig", der Test sichert den Status darum über ein Zulässigkeitsproblem ab) |
| **Lehrbuchbeispiel von Hand** | 2 Dienste (Deckungsbeitrag 3 und 5), 3 Ressourcen: Ecken (0, 0) → (0, 6) → (2, 6), **2 Pivots**, Optimum 36, alle drei Tableaus gegen eine von Hand gerechnete Tabelle geprüft; Schattenpreise 0, 1.5 und 1 |
| **Zufall mit lauter ≤** | Pivots bei m = n = 5/10/20/40/60: **3/5.5/10.5/20/32.5** (0.6/0.55/0.53/0.5/0.54 je Ressource); keine Phase 1, keine Gleichstände. Bei festem m = 20 wachsen sie langsamer als n (n = 5/20/60: 4/7/15), bei festem n = 20 langsamer als m (m = 5/20/60: 5/7/13) |
| **Mischung mit ≥ und =** | Pivots bei m = n = 5/10/20/40/60: **5/13/38/105.5/199** (1.0/1.3/1.9/2.6/3.3 je Ressource, wächst also mit der Größe); **25 bis 39 %** der Pivots in Phase 1 (bei m = n = 60: 25 % von 199) |
| **Entartung** | Bei Ressourcen, die Summen anderer Ressourcen sind: Gleichstände im Quotiententest in **16.7/36.4/15.8 %** der Pivots (m = n = 10/20/40), aber **kein einziger Nullschritt**. Nullschritte gibt es überhaupt nur in den Mischinstanzen und selten: 22 in gut 11 000 Pivots auf 720 Instanzen (240 je Typ), 19 davon in Phase 1 (Beispiel im Preset "Phase 1 in der Zeichnung": der erste Pivot ändert die Ecke nicht) |
| **Notbremse** | Die Bland-Regel als Sicherheitsnetz gegen Zyklen (nach 2 · (m + Spalten) Nullschritten in Folge) wurde auf keiner der 360 geprüften erzeugten Instanzen gebraucht |
| **Aufwand je Pivot** | (Spalten + 1) · (1 + 2 m) Rechenoperationen, exakt nachgezählt: 441 bei m = n = 10, 1681 bei 20, **14 641** bei 60; das Tableau rechnet bei jedem Pivot alle Einträge neu |

## Was die Demo zeigt

1. **Vier Schritte** (Schritt-Slider): **Geometrie** (bei zwei Diensten: Zulässigkeitsmenge, Randgeraden, Ecken, Simplex-Pfad Pivot für Pivot in Orange, Phase 1 grau gepunktet durch unzulässige Basislösungen, Höhenlinie des Zielwerts; bei mehr Diensten die Basislösung als Balken) → **Tableau** (Slider über alle Tableaus: Tabelle mit dem Pivotelement in Orange, eintretende Spalte, Quotienten, austretende Variable) → **Zwei Phasen** (Zielwert je Tableau: Summe der künstlichen Variablen fällt auf 0, dann steigt der Zielwert) → **Aufwand** (Pivots, Phasen, Operationen, Schrittweite je Pivot, Selbstprüfung ohne fremden Löser).
2. **Kennzahlen:** Pivots, Pivots je Ressource, Zielwert, Operationen.
3. **🔬 Auf Abruf:** Pivotkurve über die Größe (30 Instanzen je Größe, Median und Band), Sweeps über m / n / Dichte / Instanztyp für Pivots, Pivots je Ressource, Phase-1-Anteil, Gleichstände und Operationen je Pivot (5 feste Instanzen).

Presets (10): Standardfall (Lehrbuchbeispiel), Zwei Pivots von Hand, Entartete Ecke, Phase 1 in der Zeichnung, Phase 1 im Zielwert, Unzulässig, Unbeschränkt, Zufallsinstanz, Große Instanz, Gleichstände im Zufall.

## Messwerte der Presets

| Preset | Einstellungen | Ergebnis |
|---|---|---|
| **Standardfall** | Lehrbuchbeispiel | Ecken (0, 0) → (0, 6) → (2, 6), 2 Pivots, Optimum 36, 84 Rechenoperationen; Schattenpreise 0, 1.5, 1 |
| **Zwei Pivots von Hand** | Tableau 1 | Quotienten 6 (Kommissionierstunden), Zielwert 0 → 30; zweiter Pivot: Quotienten 4 und 2, Zielwert 36 |
| **Entartete Ecke** | vierte Ressource x1 + x2 ≤ 8 | zweiter Pivot mit Gleichstand (2 Zeilen, Quotient 2), danach eine Basisvariable null; 2 Pivots, kein Nullschritt, Schattenpreise 0, 1.5, 1, 0 |
| **Phase 1 in der Zeichnung** | Mischung 5 x 2, Dichte 0.7, Seed 0 | 5 Pivots, davon 3 in Phase 1 (der erste ein Nullschritt) bis zur ersten zulässigen Ecke, dann 2 in Phase 2; Optimum (0, 23.1), Wert 190.35 |
| **Phase 1 im Zielwert** | Mischung 20 x 20, Seed 35 | 31 Pivots, 8 in Phase 1 (25.8 %), 23 in Phase 2; Zielwert 922.97 |
| **Unzulässig** | x1 ≤ 4 und x1 ≥ 6 | 1 Pivot in Phase 1, Summe der künstlichen Variablen bleibt positiv → "unzulässig" |
| **Unbeschränkt** | x1 − x2 ≤ 2, −x1 + x2 ≤ 3 | 1 Pivot, dann eintretende Spalte ohne positiven Eintrag → "unbeschränkt" |
| **Zufallsinstanz** | 10 x 10, Seed 35 | 5 Pivots (0.5 je Ressource), 441 Operationen je Pivot, 2205 insgesamt, Optimum 314.21 |
| **Große Instanz** | 60 x 60, Seed 35 | 52 Pivots (Median der fünf festen Instanzen 46), 14 641 Operationen je Pivot, 761 332 insgesamt |
| **Gleichstände im Zufall** | 20 x 20, Seed 35 | 9 Pivots, in 4 davon Gleichstände (bis zu 4 Zeilen), kein Nullschritt, Optimum 477.66 |

## Modell und Verfahren

- **Instanz** (`tsx_scenario.py`): Auslastungsplanung als LP `max c·x`, `A x (≤|≥|=) b`, `x ≥ 0`. Lehrbuchbeispiel und drei Fixtures (entartet, unzulässig, unbeschränkt) fest, drei Zufallstypen regelbar: *Zufall* (nichtnegative Verbrauchskoeffizienten, alle Bedingungen ≤, Gesamtkapazität dicht, damit alles beschränkt bleibt), *Mischung* (aus einem zulässigen Punkt gebaut, Mindest- und Gleichungsbedingungen, höchstens n/2 Gleichungen) und *Zufall mit Gleichständen* (Ressourcen, die Summen anderer Ressourcen sind).
- **Simplex** (`tsx_algorithm.py`): Standardform mit Schlupf-, Überschuss- und künstlichen Variablen (Zeilen mit negativem b werden mit −1 multipliziert), dichtes Tableau mit Zielzeile r = z − c, Dantzig-Regel (kleinstes r, Gleichstand: kleinster Index), Quotiententest mit Toleranz (Gleichstand: kleinster Index der Basisvariable), Zwei-Phasen-Start mit Herausdrehen künstlicher Variablen aus der Basis und Erkennung redundanter Zeilen, Schattenpreise aus der Zielzeile, Bland-Regel als Notbremse gegen Zyklen.
- **Ecken** (`vertices_2d`): bei zwei Diensten alle Schnittpunkte der Randgeraden, geprüft gegen alle Bedingungen; jede Basislösung des Simplex ist Ecke (Test).
- **Aufwand:** Gleitkomma-Operationen des dichten Modells: ein Pivot kostet (Spalten + 1) + 2 · m · (Spalten + 1), gezählt statt gemessen. Ein Näherungsmaß, keine Laufzeit.
- **Selbstprüfung:** die App prüft ihr Ergebnis ohne fremden Löser (Primal- und Dual-Zulässigkeit, Lücke c·x − b·y).

## Was nicht funktioniert hat / Grenzen

- **Vorab-Hypothese "1 bis 3 Pivots je Ressource" – auf Zufallsinstanzen mit lauter ≤ widerlegt** (0.5 bis 0.6). Nur die Mischinstanzen mit Phase 1 liegen darüber (1.0 bis 3.3, mit der Größe steigend).
- **Kein Stillstand trotz Gleichständen.** Gleichstände im Quotiententest gibt es oft (bis 36 %), Nullschritte in den Gleichstands-Instanzen nie; Zyklen sind nicht aufgetreten. Das heißt nicht, dass es sie nicht gibt (Beales Beispiel gehört ins Folgestück "Pivotregeln und Entartung"), sondern dass der Zufall sie nicht findet.
- **Kein schlimmster Fall.** Der exponentielle Fall (Klee-Minty) kommt nicht vor, weil die Instanzen zufällig sind; er ist das Thema eines Folgestücks.
- **Nur Dantzig.** Regelvergleiche, Steepest Edge, Bland und Zyklen gehören zu "Pivotregeln und Entartung"; die Regel ist hier kein Regler.
- **Dichtes Tableau.** Jeder Pivot rechnet alle Einträge neu; das ist der Grund für den Revised Simplex.
- **Numerik.** Feste Toleranz 1e-9, keine Skalierung, kein Präsolve: auf den erzeugten, gut skalierten Instanzen stimmt jedes Ergebnis mit HiGHS überein, auf echten, schlecht skalierten Daten würde das nicht genügen ("Präsolve, Skalierung & Numerik").
- **Synthetische Instanzen.** Nichtnegative Verbrauchskoeffizienten, dichte Gesamtkapazität; echte LPs sind dünner, größer und strukturierter (Netzwerke: der Netzwerksimplex in [network-flow-demo](https://github.com/sebastian-hanisch/network-flow-demo) mit einem Spannbaum als Basis).
- **Keine ganzzahligen Programme.** Ganzzahligkeit (Branch and Bound, Schnittebenen, Column Generation) gehört in die Exakte-Suche-Linie; die Dualvariablen sind hier nur eine Vorschau auf das Dualitätsstück.

## Verifikation

- `tests/test_algorithm.py` (14 Tests): Optimum == HiGHS auf 720 erzeugten Instanzen, primal/dual zulässig, starke Dualität, komplementärer Schlupf; 300 kleine Zufalls-LPs mit gemischten Vorzeichen (alle drei Status gegen HiGHS); 200 konstruierte unzulässige und unbeschränkte Instanzen; Lehrbuchbeispiel mit allen drei Tableaus von Hand; jede Basislösung ist Ecke, der Zielwert fällt nie; Entartungs-Fixture (Gleichstand, Basisvariable null); Bland gleicher Optimalwert; Buchführung (Pivotzahl, Operationen, Phasen, Schnappschüsse mit Einheitsvektoren in der Basis); Sonderfälle (m = 1, c = 0, b = 0, doppelte Zeilen, negatives b).
- `tests/test_scenario.py`, `test_evaluation.py`, `test_presets.py` (Bänder + jede Zahl der Hilfetexte), `test_claims.py` (jede Zahl aus README und App über die echten `ev.*`-Funktionen), `test_app.py` (Streamlit-AppTest: Voreinstellung, jedes Preset, jeder Schritt für jeden Instanztyp, jede Position beider Schritt-Regler, Randwerte, Würfel, Permalink-Grenzen, Instanzwechsel, Experimente und Sweeps auf Abruf, Footer).
- Für die Prüfung genügt **pytest**; `scipy` dient nur als Gegenprobe (`requirements-dev.txt`), die App braucht nur numpy.

## Lokal starten

```bash
python -m venv venv && venv/Scripts/activate  # Windows; Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -W error::SyntaxWarning`.

## Literatur

- Dantzig, G. B. (1951). *Maximization of a linear function of variables subject to linear inequalities.* In T. C. Koopmans (Hrsg.), Activity Analysis of Production and Allocation, 339–347. Wiley.
- Klee, V., & Minty, G. J. (1972). *How good is the simplex algorithm?* In O. Shisha (Hrsg.), Inequalities III, 159–175. Academic Press (nur als Ausblick, Folgestück).

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
