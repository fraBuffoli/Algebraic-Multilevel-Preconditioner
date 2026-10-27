#!/usr/bin/env python3
"""Report of the Krylov campaign (Allrun_krylov.sh): GMRES vs BiCGStab with the RAS/deflated preconditioner.

    python3 report_krylov.py run_krylov_np240_r1e-2_<jobid> [more runs] [--gamg run_scaling_np240_<jobid>] [--out DIR]

Runs with the same RTOL are merged by (mesh size N, processes NP, variant); a triple present in several
runs is taken from the newest. --gamg adds GAMG (and the reference Schwarz run) of the scaling campaign
with the same NP. Writes DIR/report.md, DIR/summary.csv and, with matplotlib, DIR/plots/*.png.
Default DIR: krylov_report (next to the first run).

Times as in report_scaling.py (csv/<run>.csv: t_update, t_solve of every solve between two
synchronizations). The iterations are split with the timer table of log.<run> (averages over the
processes, inclusive timers): local solves (solve.preconditioner.local), coarse correction
(solve.preconditioner.coarse + .spmv: coarse solves and the products with A of the coarse
combination), SpMV of the Krylov method (solve.spmv), orthogonalization / vector updates
(GMRES: solve.orthogonalization; BiCGStab: solve.vector_ops), stopping test (solve.stopping_test;
BiCGStab: with the inner product (r^, r) fused in it) and the rest (true residual at the end of a
cycle, solution update, rotations). Iterations of both solvers = applications of the preconditioner
(BiCGStab: two per step), so the time per iteration compares directly.
"""
import csv
import glob
import os
import re
import statistics as st
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_scaling import (AXIS, GRID, INK, INK2, MUTED, SURFACE, fmt, job_key, read_config,  # noqa: E402
                            read_status, read_timers, style_axes, summarize_s2l)
from report_scaling import load_run as load_scaling_run  # noqa: E402

ORDER = ["amg", "bicgstab"]
LABEL = {"amg": "GMRES(30)\nRAS/deflated", "bicgstab": "BiCGStab\nRAS/deflated", "gamg": "GAMG\n(DIC)"}
SHORT = {k: v.replace("\n", " ") for k, v in LABEL.items()}
# Phases of a solve: ordinal steps of one hue (blue ramp of report_scaling.py), GAMG in its orange ramp.
COST_PHASES = [("setup", "setup + updates (amortized)", "#104281"), ("local", "local solves", "#256abf"),
               ("coarse", "coarse correction", "#3987e5"), ("krylov", "Krylov (SpMV, orth./vector ops, stop test)", "#9ec5f4")]
GAMG_PHASES = [("setup", "#a44924"), ("cycles", "#f3a485")]
IT_PHASES = [("local", "local solves", "#104281"), ("coarse", "coarse correction", "#256abf"),
             ("spmv", "SpMV (Krylov)", "#3987e5"), ("orth", "orthogonalization / vector ops", "#6da7ec"),
             ("stop", "stopping test", "#9ec5f4"), ("other", "rest", "#c9def8")]


def summarize(rows, timers):
    s, its, per = summarize_s2l(rows, timers)
    avg = lambda k: timers.get(k, (0, 0, 0))[1]
    tot = avg("solve")
    share = (lambda k: s["iter"] * avg(k) / tot) if tot > 0 else (lambda k: 0.0)
    s["spmv"] = share("solve.spmv")
    s["orth"] = share("solve.orthogonalization") + share("solve.vector_ops")
    s["stop"] = share("solve.stopping_test")
    s["other"] = max(0.0, s["krylov"] - s["spmv"] - s["orth"] - s["stop"])
    return s, its, per


def load_run(run):
    cfg = read_config(run)
    status = read_status(run)
    np_ = int(cfg.get("NP", "0") or 0)
    expected = int(cfg.get("STEPS", "0") or 0) * int(cfg.get("CORR", "0") or 0)
    recs = []
    for f in sorted(glob.glob(os.path.join(run, "csv", "bub*_*.csv"))):
        name = os.path.basename(f)[:-4]
        m = re.match(r"bub(\d+)_(.+)$", name)
        if not m or m.group(2) not in ORDER:
            continue
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        s, its, per = summarize(rows, read_timers(os.path.join(run, "log." + name)))
        st_ = status.get(name, ("missing", 0.0))
        s.update(var=m.group(2), N=int(m.group(1)), NP=np_ or int(rows[0]["np"]), run=os.path.basename(run.rstrip("/")),
                 status=st_[0], wall=st_[1], solves=len(rows), expected=expected or len(rows),
                 converged=sum(int(float(r["converged"])) for r in rows), its=its,
                 its_avg=st.mean(its), its_min=min(its), its_max=max(its))
        finish(s)
        recs.append(s)
    return cfg, recs


def finish(s):
    s["total"] = s["init"] + s["upd"] + s["iter"]
    s["cost"] = s["total"] / s["solves"]
    s["steady"] = (s["upd"] + s["iter"]) / s["solves"]
    s["t_it"] = s["iter"] / max(1, sum(s["its"]))
    s["ok"] = s["status"] == "ok" and s["solves"] == s["expected"] and s["converged"] == s["solves"]


def main(argv):
    runs, gamg_runs, outdir = [], [], None
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            outdir = argv[i + 1]
            i += 2
        elif argv[i] == "--gamg":
            gamg_runs.append(argv[i + 1])
            i += 2
        else:
            runs.append(argv[i])
            i += 1
    if not runs:
        print(__doc__)
        return 1
    runs = [r for r in runs if os.path.isdir(os.path.join(r, "csv"))]
    if not runs:
        print("no run directory with csv/ found")
        return 1
    runs.sort(key=job_key)
    outdir = outdir or os.path.join(os.path.dirname(os.path.abspath(runs[0].rstrip("/"))), "krylov_report")
    os.makedirs(outdir, exist_ok=True)

    data, rtols, cfg0 = OrderedDict(), set(), None
    for r in runs:
        cfg, recs = load_run(r)
        cfg0 = cfg0 or cfg
        rtols.add(cfg.get("RTOL", "?"))
        for s in recs:
            data[(s["var"], s["N"], s["NP"])] = s
    if len(rtols) > 1:
        print("runs with different RTOL (%s): report them separately" % ", ".join(sorted(rtols)))
        return 1
    rtol = rtols.pop()
    NPs = sorted({k[2] for k in data})
    gamg = {}
    for g in sorted(gamg_runs, key=job_key):
        gcfg, grecs = load_scaling_run(g)
        if gcfg.get("RTOL", rtol) != rtol:
            print("--gamg %s: RTOL %s differs from %s, ignored" % (g, gcfg.get("RTOL"), rtol))
            continue
        for s in grecs:
            if s["code"] == "of" and s["NP"] in NPs:
                gamg[(s["N"], s["NP"])] = s
            elif s["code"] == "s2l" and s["NP"] in NPs:
                data[("amg_scaling", s["N"], s["NP"])] = s
    Ns = sorted({k[1] for k in data if k[0] in ORDER})
    vars_ = [v for v in ORDER if any((v, n, p) in data for n in Ns for p in NPs)]
    get = lambda v, n, p: data.get((v, n, p))

    out = []
    w = out.append
    w("# Krylov campaign: GMRES vs BiCGStab, two-level Schwarz RAS/deflated\n")
    w("Runs: %s%s. Stopping test: OpenFOAM normalised residual, relTol %s; %s steps x %s correctors per run; "
      "AMG local solves (%s).\n" % (", ".join("`%s`" % os.path.basename(r.rstrip("/")) for r in runs),
                                     (" + GAMG of " + ", ".join("`%s`" % os.path.basename(g.rstrip("/")) for g in gamg_runs))
                                     if gamg_runs else "", rtol, cfg0.get("STEPS", "?"), cfg0.get("CORR", "?"),
                                     cfg0.get("BASE", "?")))
    w("Variants: " + "; ".join("**%s**" % SHORT[v] for v in vars_) + ".\n")
    bad = [s for k, s in data.items() if k[0] in ORDER and not s["ok"]]
    if bad:
        w("Incomplete or not converged: " + ", ".join("%s (%s, %d/%d solves, %d converged)" % (
            s["name"] if "name" in s else "bub%d_%s" % (s["N"], s["var"]), s["status"], s["solves"], s["expected"],
            s["converged"]) for s in bad) + ".\n")

    for p in NPs:
        w("\n## %d processes\n" % p)
        w("### Iterations per solve = applications of the preconditioner (average, min-max; BiCGStab: two per step)\n")
        w("| N | " + " | ".join(SHORT[v] for v in vars_) + (" | GAMG |" if gamg else " |"))
        w("|--:|" + "--:|" * (len(vars_) + (1 if gamg else 0)))
        for n in Ns:
            cells = []
            for v in vars_:
                s = get(v, n, p)
                cells.append("%.1f (%d-%d)" % (s["its_avg"], s["its_min"], s["its_max"]) if s else "-")
            if gamg:
                g = gamg.get((n, p))
                cells.append("%.1f (%d-%d)" % (g["its_avg"], g["its_min"], g["its_max"]) if g else "-")
            w("| %d³ | %s |" % (n, " | ".join(cells)))

        w("\n### Time per iteration [ms] and its split\n")
        w("| N | variant | per it. | local | coarse | SpMV | orth./vector | stop test | rest |")
        w("|--:|:--|--:|--:|--:|--:|--:|--:|--:|")
        for n in Ns:
            for v in vars_:
                s = get(v, n, p)
                if not s:
                    continue
                k = 1e3 / max(1, sum(s["its"]))
                w("| %d³ | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    n, SHORT[v], fmt(s["t_it"] * 1e3, 2), *(fmt(s[x] * k, 2) for x in ("local", "coarse", "spmv", "orth",
                                                                                         "stop", "other"))))

        w("\n### Cost per solve [s] (initial setup + updates + iterations, / solves)\n")
        w("| N | variant | setup + updates | iterations | **cost** | vs GMRES+RAS | vs GAMG | rebuilds |")
        w("|--:|:--|--:|--:|--:|--:|--:|--:|")
        for n in Ns:
            ref = get("amg", n, p)
            g = gamg.get((n, p))
            for v in vars_:
                s = get(v, n, p)
                if not s:
                    continue
                w("| %d³ | %s | %s | %s | **%s** | %s | %s | %d |" % (
                    n, SHORT[v], fmt((s["init"] + s["upd"]) / s["solves"]), fmt(s["iter"] / s["solves"]), fmt(s["cost"]),
                    ("%.2f" % (s["cost"] / ref["cost"])) if ref else "-",
                    ("%.2f" % (s["cost"] / g["cost"])) if g else "-", s["rebuilds"]))
            if g:
                w("| %d³ | GAMG (DIC) | %s | %s | **%s** | %s | 1.00 | - |" % (
                    n, fmt((g["init"] + g["upd"]) / g["solves"]), fmt(g["iter"] / g["solves"]), fmt(g["cost"]),
                    ("%.2f" % (g["cost"] / ref["cost"])) if ref else "-"))
            r2 = get("amg_scaling", n, p)
            if r2 and ref:
                w("| %d³ | (GMRES RAS/deflated of the scaling campaign) | | | %s | %.2f | | |" % (
                    n, fmt(r2["cost"]), r2["cost"] / ref["cost"]))

        # Decomposition of the net effect.
        if all(v in vars_ for v in ("amg", "bicgstab")):
            w("\n### BiCGStab / GMRES\n")
            w("| N | applications of M | time per application | iterations cost | setup + updates | cost per solve | rebuilds |")
            w("|--:|--:|--:|--:|--:|--:|--:|")
            for n in Ns:
                a, b = get("amg", n, p), get("bicgstab", n, p)
                if not (a and b):
                    continue
                w("| %d³ | %.2f | %.2f | %.2f | %.2f | %.2f | %d / %d |" % (
                    n, b["its_avg"] / a["its_avg"], b["t_it"] / a["t_it"], b["iter"] / a["iter"],
                    (b["init"] + b["upd"]) / (a["init"] + a["upd"]), b["cost"] / a["cost"], b["rebuilds"], a["rebuilds"]))
            w("\nIterations cost = applications × time per application. Cost per solve below 1: BiCGStab is "
              "faster than the GMRES of the GAMG and scaling campaigns. Break-even of the iterations: applications "
              "ratio = 1 / (time per application ratio).")

    made = plots(outdir, data, gamg, vars_, Ns, NPs)
    for pth in made:
        w("\n![%s](plots/%s)" % (os.path.basename(pth)[:-4], os.path.basename(pth)))
    open(os.path.join(outdir, "report.md"), "w").write("\n".join(out) + "\n")

    keys = ["var", "N", "NP", "run", "status", "solves", "converged", "its_avg", "its_min", "its_max", "t_it", "init", "upd",
            "iter", "local", "coarse", "krylov", "spmv", "orth", "stop", "other", "rebuilds", "refact", "total", "cost",
            "steady", "final_of", "true_res", "wall"]
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for k in sorted(data, key=lambda k: (k[2], k[1], ORDER.index(k[0]) if k[0] in ORDER else 99)):
            s = data[k]
            wr.writerow([s.get(x, k[0] if x == "var" else "") for x in keys])
        for (n, p), g in sorted(gamg.items()):
            wr.writerow([dict(g, var="gamg").get(x, "") for x in keys])
    print("\n".join(out))
    print("\nwritten: %s (report.md, summary.csv, %d plots)" % (outdir, len(made)))
    return 0


def plots(outdir, data, gamg, vars_, Ns, NPs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib not available: tables only")
        return []
    pdir = os.path.join(outdir, "plots")
    os.makedirs(pdir, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9.5, "axes.labelcolor": INK2, "text.color": INK,
                         "axes.titlesize": 10.5, "axes.titlecolor": INK, "legend.frameon": False, "legend.fontsize": 8.5})
    made = []
    get = lambda v, n, p: data.get((v, n, p))
    bar = dict(width=0.62, edgecolor=SURFACE, linewidth=2)   # 2px surface gap between stacked segments

    def save(fig, name):
        pth = os.path.join(pdir, name)
        fig.savefig(pth, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        made.append(pth)

    for p in NPs:
        # 1. Cost per solve, split by phase; one panel per mesh size.
        cols = vars_ + (["gamg"] if any((n, p) in gamg for n in Ns) else [])
        fig, axes = plt.subplots(1, len(Ns), figsize=(1.05 * len(cols) * len(Ns) + 1.2, 4.3), facecolor=SURFACE,
                                 squeeze=False)
        for j, n in enumerate(Ns):
            ax = axes[0][j]
            style_axes(ax)
            ax.grid(axis="x", visible=False)
            for x, v in enumerate(cols):
                if v == "gamg":
                    g = gamg.get((n, p))
                    if not g:
                        continue
                    parts = [((g["init"] + g["upd"]) / g["solves"], GAMG_PHASES[0][1]), (g["iter"] / g["solves"], GAMG_PHASES[1][1])]
                    total = g["cost"]
                else:
                    s = get(v, n, p)
                    if not s:
                        continue
                    k = 1.0 / s["solves"]
                    parts = [((s["init"] + s["upd"]) * k, COST_PHASES[0][2]), (s["local"] * k, COST_PHASES[1][2]),
                             (s["coarse"] * k, COST_PHASES[2][2]), (s["krylov"] * k, COST_PHASES[3][2])]
                    total = s["cost"]
                y0 = 0.0
                for val, col in parts:
                    ax.bar(x, val, bottom=y0, color=col, **bar)
                    y0 += val
                ax.annotate(fmt(total, 2), (x, total), xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=8, color=INK2)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([LABEL[v] for v in cols], fontsize=7.5)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
            ax.set_title("%d³ (%.1fM cells)" % (n, n ** 3 / 1e6), loc="left")
            if j == 0:
                ax.set_ylabel("cost per solve [s]")
        handles = [Patch(color=c, label=l) for _, l, c in COST_PHASES]
        if "gamg" in cols:
            handles += [Patch(color=GAMG_PHASES[0][1], label="GAMG: setup + updates"),
                        Patch(color=GAMG_PHASES[1][1], label="GAMG: V-cycles")]
        fig.legend(handles=handles, loc="lower center", ncol=min(3, len(handles)), bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.13 if len(handles) > 3 else 0.08, 1, 1))
        save(fig, "cost_per_solve_np%d.png" % p)

        # 2. Time per iteration, split by phase.
        fig, axes = plt.subplots(1, len(Ns), figsize=(1.05 * len(vars_) * len(Ns) + 1.2, 4.1), facecolor=SURFACE,
                                 squeeze=False)
        for j, n in enumerate(Ns):
            ax = axes[0][j]
            style_axes(ax)
            ax.grid(axis="x", visible=False)
            for x, v in enumerate(vars_):
                s = get(v, n, p)
                if not s:
                    continue
                k = 1e3 / max(1, sum(s["its"]))
                y0 = 0.0
                for key, _, col in IT_PHASES:
                    ax.bar(x, s[key] * k, bottom=y0, color=col, **bar)
                    y0 += s[key] * k
                ax.annotate(fmt(s["t_it"] * 1e3, 2), (x, y0), xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=8, color=INK2)
            ax.set_xticks(range(len(vars_)))
            ax.set_xticklabels([LABEL[v] for v in vars_], fontsize=7.5)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
            ax.set_title("%d³" % n, loc="left")
            if j == 0:
                ax.set_ylabel("time per iteration [ms]")
        fig.legend(handles=[Patch(color=c, label=l) for _, l, c in IT_PHASES], loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        save(fig, "time_per_iteration_np%d.png" % p)
    return made


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
