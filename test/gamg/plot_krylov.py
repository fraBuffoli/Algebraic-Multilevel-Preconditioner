#!/usr/bin/env python3
"""Plots of a Krylov campaign run (Allrun_krylov.sh), written into the run directory itself.

    python3 plot_krylov.py run_krylov_np240_r1e-2_<jobid> [more runs] --gamg run_scaling_np240_<jobid>

For every run given, <run>/plots/ receives:
  cost_per_solve_np<NP>.png      cost per solve split by phase (setup + updates amortized, local solves,
                                 coarse correction, Krylov), GMRES and BiCGStab, and GAMG of --gamg
  time_per_iteration_np<NP>.png  time per application of the preconditioner, split into local solves,
                                 coarse correction, SpMV, orthogonalization / vector operations,
                                 stopping test, rest
  solves_np<NP>.png              every solve of the time loop: iterations (applications of the
                                 preconditioner; GAMG: V-cycles) and time (update + solve), rebuilds marked
--gamg (required): a run of the scaling campaign with the same processes, for the GAMG reference (same
nodes type, matrices and stopping test). The tables stay in report_krylov.py.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_krylov import LABEL, ORDER, load_run, plots  # noqa: E402
from report_scaling import INK2, MUTED, SURFACE, fmt, read_config, style_axes  # noqa: E402
from report_scaling import load_run as load_scaling_run  # noqa: E402

# Categorical slots 1-3 of the reference palette (validated all-pairs, light surface): the Schwarz
# reference keeps the blue and GAMG the orange of report_scaling.py; BiCGStab takes the aqua (below 3:1
# contrast: lines are labelled at their end, GAMG is also dashed).
COLOR = {"amg": "#2a78d6", "bicgstab": "#1baf7a", "gamg": "#eb6834"}
NAME = {"amg": "GMRES(30)", "bicgstab": "BiCGStab", "gamg": "GAMG (V-cycles)"}


def read_rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def solves_plot(run, recs, gamg_run, Ns, NP):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

    of_solver = read_config(gamg_run).get("OF_SOLVER", "GAMG_DIC") if gamg_run else None
    fig, axes = plt.subplots(2, len(Ns), figsize=(4.0 * len(Ns) + 0.6, 6.4), facecolor=SURFACE, squeeze=False,
                             sharex="col", gridspec_kw={"height_ratios": [1.3, 1]})
    for j, n in enumerate(Ns):
        ai, at = axes[0][j], axes[1][j]
        style_axes(ai)
        style_axes(at)
        series = []
        for v in ORDER:
            rows = read_rows(os.path.join(run, "csv", "bub%d_%s.csv" % (n, v)))
            if rows:
                its = [int(float(r["iterations"])) for r in rows]
                t = [float(r["t_update"]) + float(r["t_solve"]) for r in rows]
                rb = [k for k, r in enumerate(rows) if r["level"] == "rebuild" and k > 0]
                series.append((v, its, t, rb))
        if of_solver:
            rows = read_rows(os.path.join(gamg_run, "csv", "bub%d_%s.csv" % (n, of_solver)))
            if rows:
                series.append(("gamg", [int(float(r["iterations"])) for r in rows],
                               [float(r["t_setup"]) + float(r["t_cycles"]) for r in rows], []))
        for v, its, t, rb in series:
            x = list(range(1, len(its) + 1))
            ls = (0, (4, 2)) if v == "gamg" else "-"
            for ax, y in ((ai, its), (at, t)):
                ax.plot(x, y, color=COLOR[v], linewidth=2, linestyle=ls, zorder=3 if v != "gamg" else 2)
                if rb:  # rebuilds of the coarse space: markers with a surface ring
                    ax.plot([k + 1 for k in rb], [y[k] for k in rb], linestyle="none", marker="o", markersize=8,
                            color=COLOR[v], markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
            ai.annotate(NAME[v].split(" ")[0], (x[-1], its[-1]), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        ai.set_title("%d³ (%.1fM cells), %d processes" % (n, n ** 3 / 1e6, NP), loc="left")
        at.set_yscale("log")
        at.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        at.yaxis.set_minor_locator(NullLocator())
        at.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v, 2)))
        at.set_xlabel("solve (16 steps × 3 correctors)")
        if j == 0:
            ai.set_ylabel("iterations per solve\n(applications of M; GAMG: V-cycles)")
            at.set_ylabel("time per solve [s]\n(update + solve)")
        ai.set_ylim(bottom=0)
        ai.margins(x=0.08)
    handles = [Line2D([], [], color=COLOR[v], linewidth=2, linestyle=(0, (4, 2)) if v == "gamg" else "-",
                      label=NAME[v]) for v in ("amg", "bicgstab", "gamg")]
    handles.append(Line2D([], [], linestyle="none", marker="o", markersize=8, color=INK2, markeredgecolor=SURFACE,
                          markeredgewidth=2, label="rebuild of the coarse space"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.01, 0.045, "Schwarz: RAS + deflated coarse correction, AMG local solves; time per solve includes the "
             "update of the preconditioner (reuse / refactorization / rebuild); GAMG: construction of the coarse "
             "matrices + V-cycles", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = os.path.join(run, "plots", "solves_np%d.png" % NP)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run_krylov_* directories")
    ap.add_argument("--gamg", required=True, help="run_scaling_np<NP>_* directory with the same processes")
    a = ap.parse_args(argv)
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is needed")
        return 1
    gcfg, grecs = load_scaling_run(a.gamg)
    for run in a.runs:
        if not os.path.isdir(os.path.join(run, "csv")):
            print("%s: no csv/ directory, skipped" % run)
            continue
        cfg, recs = load_run(run)
        if gcfg.get("RTOL") and cfg.get("RTOL") and gcfg["RTOL"] != cfg["RTOL"]:
            print("%s: RTOL %s, --gamg run RTOL %s: GAMG left out" % (run, cfg["RTOL"], gcfg["RTOL"]))
            gamg, gamg_run = {}, None
        else:
            gamg = {(s["N"], s["NP"]): s for s in grecs if s["code"] == "of"}
            gamg_run = a.gamg
        data = {(s["var"], s["N"], s["NP"]): s for s in recs}
        Ns = sorted({k[1] for k in data})
        NPs = sorted({k[2] for k in data})
        if any((n, p) not in gamg for n in Ns for p in NPs) and gamg_run:
            print("%s: --gamg has no GAMG run for some (N, NP): those bars are missing" % run)
        vars_ = [v for v in ORDER if any(k[0] == v for k in data)]
        made = plots(run, data, gamg, vars_, Ns, NPs)
        for p in NPs:
            made.append(solves_plot(run, recs, gamg_run if p == int(gcfg.get("NP", p) or p) else None, Ns, p))
        print("%s: %s" % (run, ", ".join(os.path.relpath(m, run) for m in made)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
