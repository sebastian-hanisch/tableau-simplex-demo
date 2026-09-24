"""Auswertung des Tableau-Simplex: Analyse einer Instanz, Kennzahlen über feste Instanzen, Sweeps und die Pivotkurve."""

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import tsx_algorithm as A
import tsx_constants as C
import tsx_scenario as S

INF = math.inf


@dataclass(frozen=True)
class Settings:
    kind: str = "textbook"
    m: int = C.DEFAULT_M
    n: int = C.DEFAULT_N
    density: float = C.DEFAULT_DENSITY
    seed: int = C.DEFAULT_SEED


@lru_cache(maxsize=256)
def instance_of(settings):
    return S.generate(settings.kind, settings.m, settings.n, settings.density, settings.seed)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    res: object
    vertices: list                     # zulässige Ecken (nur bei zwei Diensten)

    @property
    def m(self):
        return self.inst.m

    @property
    def n(self):
        return self.inst.n

    @property
    def status(self):
        return self.res.status

    @property
    def pivots(self):
        return self.res.total_pivots

    @property
    def phase1_share(self):
        return self.res.phase1_pivots / self.res.total_pivots if self.res.total_pivots else 0.0

    @property
    def certificate(self):
        """Selbstprüfung ohne fremden Löser: Verletzungen von Primal- und Dual-Zulässigkeit und Lücke zwischen c·x und b·y (0 bei starker Dualität); nur bei Optimum."""
        if self.res.status != "optimal":
            return None
        b = np.array(self.inst.b)
        return {"primal": A.primal_violation(self.inst, self.res.x), "dual": A.dual_violation(self.inst, self.res.duals), "gap": abs(self.res.obj - float(np.dot(self.res.duals, b)))}


@lru_cache(maxsize=64)
def analyse(settings):
    inst = instance_of(settings)
    res = A.tableau_simplex(inst, keep=True)
    return Analysis(settings, inst, res, A.vertices_2d(inst) if inst.n == 2 else [])


def _median(values):
    values = [v for v in values if v is not None and v == v and v != INF]
    return float(np.median(values)) if values else float("nan")


def _stats(values):
    values = [v for v in values if v is not None and v == v and v != INF]
    if not values:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(values)), float(np.percentile(values, 10)), float(np.percentile(values, 90))


def metrics(inst):
    """Kennzahlen einer einzelnen Instanz ohne Schnappschüsse (schnell)."""
    r = A.tableau_simplex(inst)
    return {"status": r.status, "pivots": r.total_pivots, "pivots_per_m": r.total_pivots / inst.m, "phase1_share": (r.phase1_pivots / r.total_pivots) if r.total_pivots else 0.0,
            "degenerate_share": (r.degenerate_pivots / r.total_pivots) if r.total_pivots else 0.0, "tie_share": (sum(p.ties > 1 for p in r.pivots) / r.total_pivots) if r.total_pivots else 0.0,
            "flops_per_pivot": r.flops_per_pivot, "flops": r.flops, "n_cols": r.n_cols, "bland": r.switched_to_bland, "obj": r.obj}


def run_config(base, seeds=C.SWEEP_SEEDS, **changes):
    """Kennzahlen über feste Instanzen (Median, 10./90. Perzentil); die Instanztypen mit Zufall variieren nur den Seed."""
    s0 = replace(base, **changes)
    rows = [metrics(instance_of(replace(s0, seed=seed))) for seed in seeds]
    out = {"n_runs": len(rows), "solved": sum(r["status"] == "optimal" for r in rows)}
    for key in ("pivots", "pivots_per_m", "phase1_share", "degenerate_share", "tie_share", "flops_per_pivot", "flops", "n_cols"):
        out[key], out[f"{key}_lo"], out[f"{key}_hi"] = _stats([r[key] for r in rows if r["status"] == "optimal"])
    out["bland"] = sum(r["bland"] for r in rows)
    return out


SWEEP_VALUES = {"m": C.SIZE_SWEEP, "n": C.SIZE_SWEEP, "density": C.DENSITY_OPTIONS, "kind": ("random", "mixed", "degenerate")}
SWEEP_LABELS = {"m": "Ressourcen m", "n": "Dienste n", "density": "Dichte der Matrix", "kind": "Instanztyp"}


def sweep(param, base, values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def pivot_curve(kind, sizes=C.SIZE_SWEEP, seeds=C.FEAS_SEEDS, density=C.DEFAULT_DENSITY):
    """Pivots über die Größe m = n (Median, 10./90. Perzentil über `seeds` Instanzen)."""
    rows = []
    for k in sizes:
        ms = [metrics(S.generate(kind, k, k, density, s)) for s in seeds]
        piv = _stats([r["pivots"] for r in ms])
        rows.append({"size": k, "pivots": piv[0], "pivots_lo": piv[1], "pivots_hi": piv[2], "per_m": _median([r["pivots_per_m"] for r in ms]), "phase1_share": _median([r["phase1_share"] for r in ms]),
                     "flops_per_pivot": _median([r["flops_per_pivot"] for r in ms])})
    return rows
