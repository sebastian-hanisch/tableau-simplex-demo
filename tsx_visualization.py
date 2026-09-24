"""Plotly-Abbildungen: Zulässigkeitsmenge mit Simplex-Pfad (zwei Dienste), Basislösung, Zielwert je Pivot, Tableau-Tabelle, Pivotkurven und Sweeps.
Die Geometrie hat feste Achsenbereiche aus der Instanz (ohne scaleanchor, damit nichts einfriert); Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import tsx_scenario as S

TEAL, ORANGE, RED, BLUE, GREY = "#2F6B65", "#e8a13a", "#d62728", "#1f4e9c", "#8a8f98"
LINE_COLORS = ["#4c78a8", "#7b3fbf", "#d95f9b", "#2ca02c", "#8c564b", "#17becf", "#bcbd22", "#e377c2"]


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _clip(poly, a, b_, c):
    """Halbebene a x + b y <= c auf ein konvexes Polygon anwenden (Sutherland-Hodgman)."""
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        fp, fq = a * p[0] + b_ * p[1] - c, a * q[0] + b_ * q[1] - c
        if fp <= 1e-12:
            out.append(p)
        if (fp < -1e-12 and fq > 1e-12) or (fp > 1e-12 and fq < -1e-12):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def plot_box(inst, vertices, status):
    """Sichtbarer Bereich [0, X] x [0, Y]: 1,35-fache größte Ecke, bei offenem Gebiet größer."""
    if not vertices:
        return 10.0, 10.0
    scale = 1.35 if status != "unbounded" else 2.4
    return max(1.0, scale * max(v[0] for v in vertices)), max(1.0, scale * max(v[1] for v in vertices))


def _line_in_box(a, b_, c, X, Y):
    pts = []
    if abs(b_) > 1e-12:
        for x in (0.0, X):
            y = (c - a * x) / b_
            if -1e-9 <= y <= Y + 1e-9:
                pts.append((x, y))
    if abs(a) > 1e-12:
        for y in (0.0, Y):
            x = (c - b_ * y) / a
            if -1e-9 <= x <= X + 1e-9:
                pts.append((x, y))
    uniq = []
    for p in pts:
        if not any(abs(p[0] - q[0]) < 1e-9 and abs(p[1] - q[1]) < 1e-9 for q in uniq):
            uniq.append(p)
    return uniq[:2]


def path_points(inst, res):
    """Punkte des Simplex-Pfads (Start = Ursprung, dann die Basislösung nach jedem Pivot) mit Zulässigkeit (Phase 1 läuft durch unzulässige Basislösungen)."""
    origin_ok = res.phase1_pivots == 0
    pts = [(0.0, 0.0)] + [p.x for p in res.pivots]
    ok = [origin_ok] + [p.feasible for p in res.pivots]
    return pts, ok


def build_geometry(inst, res, vertices, k):
    """Zwei Dienste: Zulässigkeitsmenge, Randgeraden, Ecken, Pfad bis Pivot k, Höhenlinie der Zielfunktion durch die aktuelle Basislösung."""
    A, b, c = inst.arrays()
    X, Y = plot_box(inst, vertices, res.status)
    poly = [(0.0, 0.0), (X, 0.0), (X, Y), (0.0, Y)]
    for i in range(inst.m):
        if inst.senses[i] == S.LE:
            poly = _clip(poly, A[i, 0], A[i, 1], b[i])
        elif inst.senses[i] == S.GE:
            poly = _clip(poly, -A[i, 0], -A[i, 1], -b[i])
    fig = go.Figure()
    if len(poly) >= 3:
        xs, ys = [p[0] for p in poly] + [poly[0][0]], [p[1] for p in poly] + [poly[0][1]]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor="rgba(47,107,101,0.13)", line=dict(width=0), name="zulässig", hoverinfo="skip"))
    for i in range(inst.m):
        seg = _line_in_box(A[i, 0], A[i, 1], b[i], X, Y)
        if len(seg) == 2:
            sense = {S.LE: "≤", S.GE: "≥", S.EQ: "="}[inst.senses[i]]
            fig.add_trace(go.Scatter(x=[seg[0][0], seg[1][0]], y=[seg[0][1], seg[1][1]], mode="lines", line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=2, dash="dash" if inst.senses[i] != S.LE else "solid"),
                                     name=f"{inst.row_names[i]} {sense} {b[i]:g}", hoverinfo="skip"))
    if vertices:
        fig.add_trace(go.Scatter(x=[v[0] for v in vertices], y=[v[1] for v in vertices], mode="markers", marker=dict(size=8, color="white", line=dict(width=1.5, color=GREY)), name="Ecken", hoverinfo="skip"))
    pts, ok = path_points(inst, res)
    k = max(0, min(k, len(pts) - 1))
    for a in range(k):
        good = ok[a] and ok[a + 1]
        fig.add_trace(go.Scatter(x=[pts[a][0], pts[a + 1][0]], y=[pts[a][1], pts[a + 1][1]], mode="lines", line=dict(color=ORANGE if good else GREY, width=4 if good else 2, dash="solid" if good else "dot"),
                                 name="Simplex-Pfad" if a == 0 else None, showlegend=a == 0, hoverinfo="skip"))
    cur = pts[k]
    z = float(np.dot(c, cur))
    if abs(c[0]) + abs(c[1]) > 0 and res.status != "infeasible":
        seg = _line_in_box(c[0], c[1], z, X, Y)
        if len(seg) == 2:
            fig.add_trace(go.Scatter(x=[seg[0][0], seg[1][0]], y=[seg[0][1], seg[1][1]], mode="lines", line=dict(color=RED, width=1.6, dash="dot"), name="Zielwert-Höhenlinie", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[cur[0]], y=[cur[1]], mode="markers", marker=dict(size=15, color=ORANGE if ok[k] else GREY, symbol="circle", line=dict(width=2, color="white")), name="aktuelle Basislösung", hoverinfo="skip"))
    fig.update_xaxes(title_text=inst.names[0], range=[-0.03 * X, X], zeroline=True)
    fig.update_yaxes(title_text=inst.names[1], range=[-0.03 * Y, Y], zeroline=True)
    return _base(fig, 470, legend_y=-0.32)


def build_basis_bars(inst, res, k):
    """Basislösung (Entscheidungsvariablen) nach Pivot k als Balken; k = 0 ist der Start."""
    x = (0.0,) * inst.n if k == 0 else res.pivots[k - 1].x
    fig = go.Figure(go.Bar(x=list(inst.names), y=list(x), marker_color=TEAL))
    fig.update_yaxes(title_text="Menge in der Basislösung")
    fig.update_xaxes(tickangle=-40 if inst.n > 6 else 0)
    return _base(fig, 320, legend_y=-0.4)


def build_progress(res):
    """Zielwert je Tableau: Phase 1 (Summe der künstlichen Variablen, rechte Achse, bis 0) und Phase 2 (Zielwert, steigt)."""
    idx1 = [i for i, s in enumerate(res.snapshots) if s["phase"] == 1]
    idx2 = [i for i, s in enumerate(res.snapshots) if s["phase"] == 2]
    fig = go.Figure()
    w = [max(0.0, -float(res.snapshots[i]["T"][-1, -1])) for i in idx1]
    if idx1:
        fig.add_trace(go.Scatter(x=idx1, y=w, mode="lines+markers", line=dict(color=RED, width=2.5), name="Phase 1: Summe der künstlichen Variablen", yaxis="y2"))
    if idx2:
        fig.add_trace(go.Scatter(x=idx2, y=[res.snapshots[i]["T"][-1, -1] for i in idx2], mode="lines+markers", line=dict(color=TEAL, width=2.5), name="Phase 2: Zielwert"))
    fig.update_xaxes(title_text="Tableau (0 = Start)", dtick=1 if len(res.snapshots) <= 20 else None)
    fig.update_yaxes(title_text="Zielwert", rangemode="tozero")
    fig.update_layout(yaxis2=dict(title="Künstliche Variablen (Phase 1)", overlaying="y", side="right", range=[0, 1.05 * max(w + [1e-9])], showgrid=False))
    return _base(fig, 340, legend_y=-0.35)


def build_step_sizes(res):
    """Schrittweite (kleinster Quotient) je Pivot; rot = Nullschritt (entartet)."""
    fig = go.Figure(go.Bar(x=list(range(1, res.total_pivots + 1)), y=[p.ratio for p in res.pivots], marker_color=[RED if p.degenerate else TEAL for p in res.pivots]))
    fig.update_xaxes(title_text="Pivot", dtick=1 if res.total_pivots <= 20 else None)
    fig.update_yaxes(title_text="Schrittweite (kleinster Quotient)")
    return _base(fig, 300, legend_y=-0.3)


def tableau_frame(snapshot, res, next_pivot):
    """Tableau als Tabelle (Zeilen = Basisvariablen, letzte Zeile = reduzierte Kosten) und Koordinaten des nächsten Pivotelements (oder None)."""
    T, basis = snapshot["T"], snapshot["basis"]
    names = list(res.col_names)
    df = pd.DataFrame(T, index=[names[j] for j in basis] + ["z"], columns=names + ["rhs"])
    cell = None if next_pivot is None else (next_pivot.leave_row, next_pivot.enter)
    return df, cell


def style_tableau(df, cell):
    def hi(_):
        css = pd.DataFrame("", index=df.index, columns=df.columns)
        if cell is not None:
            css.iloc[cell[0], cell[1]] = "background-color: #e8a13a; color: #14233B; font-weight: bold"
            css.iloc[len(df) - 1, cell[1]] = "background-color: rgba(232,161,58,0.35)"
        css.iloc[len(df) - 1, :] = css.iloc[len(df) - 1, :].where(css.iloc[len(df) - 1, :] != "", "background-color: rgba(47,107,101,0.10)")
        return css
    return df.style.apply(hi, axis=None).format("{:.3f}")


def build_pivot_curve(rows, label="m = n"):
    """Pivots über die Größe (Median, Band 10. bis 90. Perzentil) gegen die Linie 'ein Pivot je Ressource'."""
    xs = [r["size"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[r["pivots_hi"] for r in rows] + [r["pivots_lo"] for r in rows][::-1], mode="lines", fill="toself", fillcolor="rgba(47,107,101,0.13)", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["pivots"] for r in rows], mode="lines+markers", line=dict(color=TEAL, width=2.5), name="Pivots (Median)"))
    fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", line=dict(color=GREY, width=1.5, dash="dot"), name="ein Pivot je Ressource", hoverinfo="skip"))
    fig.update_xaxes(title_text=label)
    fig.update_yaxes(title_text="Pivots")
    return _base(fig, 340, legend_y=-0.35)


def build_sweep(rows, param_label, series, y_label, tick=None):
    """`series` = [(key, Name, Farbe)]: Median als Linie, 10. bis 90. Perzentil als Band (`<key>_lo`/`<key>_hi`)."""
    xs = [tick(r["value"]) if tick else str(r["value"]) for r in rows]
    fig = go.Figure()
    for key, name, color in series:
        ys = [None if r[key] != r[key] else r[key] for r in rows]
        lo = [None if r.get(f"{key}_lo", r[key]) != r.get(f"{key}_lo", r[key]) else r.get(f"{key}_lo", r[key]) for r in rows]
        hi = [None if r.get(f"{key}_hi", r[key]) != r.get(f"{key}_hi", r[key]) else r.get(f"{key}_hi", r[key]) for r in rows]
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        if all(v is not None for v in lo + hi):
            fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], mode="lines", fill="toself", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.13)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=color, width=2.5), name=name, connectgaps=False))
    fig.update_xaxes(title_text=param_label, type="category")
    fig.update_yaxes(title_text=y_label)
    return _base(fig, 340, legend_y=-0.35)
