#!/usr/bin/env python3
"""Report of the scaling campaign (Allrun_scaling.sh): GAMG of OpenFOAM vs two-level Schwarz.

    python3 report_scaling.py run_scaling_np240_<jobid> run_scaling_np336_<jobid> ... [--out DIR]
    python3 report_scaling.py .                 (every run_scaling_np*_* directory found in .)

Merges the runs by (mesh size N, processes NP); a pair present in several runs is taken from the
newest run. Writes DIR/report.md (tables), DIR/summary.csv (one line per code, N and NP) and, if
matplotlib is available, DIR/plots/*.png. Default DIR: scaling_report (next to the runs).

Times (all from csv/<run>.csv, every solve between two synchronizations of all processes):
  schwarz2lvl  initial setup = t_update of the first solve; updates = t_update of the other solves
               (rebuilds, refactorizations, reuse); iterations = t_solve (GMRES). The iterations are
               split into local solves, coarse correction and Krylov (SpMV, orthogonalization,
               stopping test) with the timer table of log.<run> (averages over the processes).
  GAMG         initial setup = t_setup of the first solve (agglomeration + coarse matrices);
               updates = t_setup of the other solves (coarse matrices of the new matrix);
               iterations = t_cycles (V-cycles and residuals).
  cost per solve = (initial setup + updates + iterations) / solves: the setup is amortized over the
  STEPS x CORR solves of the run; steady state = (updates + iterations) / solves.
Matrix assembly (OpenFOAM) and matrix generation (schwarz2lvl) are not part of the costs.
"""
import csv
import glob
import os
import re
import shlex
import statistics as st
import sys
from collections import OrderedDict

# Reference palette of the dataviz rules: codes = categorical slots 1 and 2 (validated pair);
# phases = ordinal steps of the code's own hue (blue ramp; orange tints) with 2px gaps and labels.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S2L, OF = "s2l", "of"
COLOR = {S2L: "#2a78d6", OF: "#eb6834"}
NAME = {S2L: "two-level Schwarz (AMG local)", OF: "GAMG"}
PHASES = {
    S2L: [("init", "initial setup", "#104281"), ("upd", "updates (rebuild / refactor)", "#256abf"),
          ("local", "iterations: local solves", "#3987e5"), ("coarse", "iterations: coarse correction", "#6da7ec"),
          ("krylov", "iterations: Krylov (SpMV, orth., stop test)", "#9ec5f4")],
    OF: [("init", "initial setup (agglomeration)", "#a44924"), ("upd", "updates (coarse matrices)", "#eb6834"),
         ("cycles", "iterations: V-cycles", "#f3a485")],
}
NP_STYLE = ["-", "--", ":", "-."]


# ---------------------------------------------------------------------------------------------
def read_config(run):
    cfg = {}
    p = os.path.join(run, "config.txt")
    if os.path.exists(p):
        for tok in shlex.split(open(p).read()):
            if "=" in tok:
                k, v = tok.split("=", 1)
                cfg[k] = v
    return cfg


def read_status(run):
    out = {}
    p = os.path.join(run, "status.txt")
    if os.path.exists(p):
        for line in open(p):
            f = line.split()
            if len(f) >= 4:
                out[f[0]] = (f[2], float(f[3]))
    return out


def read_timers(log):
    """Timer table at the end of a schwarz2lvl log: phase -> (min, avg, max)."""
    t = {}
    if not os.path.exists(log):
        return t
    on = False
    for line in open(log, errors="replace"):
        if line.startswith("Timings [s] over"):
            on = True
            continue
        if on:
            f = line.split()
            if len(f) == 4 and f[0] != "phase":
                try:
                    t[f[0]] = tuple(float(x) for x in f[1:])
                except ValueError:
                    break
            elif f and f[0] != "phase":
                break
    return t


def grep_float(path, pattern):
    if not os.path.exists(path):
        return None
    m = re.search(pattern, open(path, errors="replace").read())
    return float(m.group(1)) if m else None


def summarize_s2l(rows, timers):
    upd = [float(r["t_update"]) for r in rows]
    sol = [float(r["t_solve"]) for r in rows]
    its = [int(float(r["iterations"])) for r in rows]
    s = OrderedDict()
    s["init"] = upd[0]
    s["upd"] = sum(upd[1:])
    s["iter"] = sum(sol)
    s["rebuilds"] = sum(1 for r in rows[1:] if r["level"] == "rebuild")
    s["refact"] = sum(1 for r in rows[1:] if r["level"] == "refactorize")
    rb = [float(r["t_update"]) for r in rows[1:] if r["level"] == "rebuild"]
    s["rebuild_med"] = st.median(rb) if rb else None
    s["final_of"] = max(float(r.get("final_res_of") or 0) for r in rows)
    # Split of the iterations with the timers (averages over the processes, inclusive).
    avg = lambda k: timers.get(k, (0, 0, 0))[1]
    tot = avg("solve")
    if tot > 0:
        loc = avg("solve.preconditioner.local")
        crs = avg("solve.preconditioner.coarse") + avg("solve.preconditioner.spmv")
        f_loc, f_crs = loc / tot, crs / tot
    else:
        f_loc = f_crs = 0.0
    s["local"] = s["iter"] * f_loc
    s["coarse"] = s["iter"] * f_crs
    s["krylov"] = s["iter"] - s["local"] - s["coarse"]
    s["t_eig"] = avg("setup.eigensolver") + avg("update.eigensolver")
    s["t_generate"] = timers.get("decomposition", (0, 0, 0))[2] + timers.get("decomposition.update", (0, 0, 0))[2]
    return s, its, [u + v for u, v in zip(upd, sol)]


def summarize_of(rows):
    setup = [float(r["t_setup"]) for r in rows]
    cyc = [float(r["t_cycles"]) for r in rows]
    its = [int(float(r["iterations"])) for r in rows]
    s = OrderedDict()
    s["init"] = setup[0]
    s["upd"] = sum(setup[1:])
    s["iter"] = sum(cyc)
    s["cycles"] = s["iter"]
    s["final_of"] = max(float(r["final_res"]) for r in rows)
    s["t_generate"] = sum(float(r["t_assembly"]) for r in rows)
    return s, its, [u + v for u, v in zip(setup, cyc)]


def load_run(run):
    cfg = read_config(run)
    status = read_status(run)
    np_ = int(cfg.get("NP", "0") or 0)
    ofname = cfg.get("OF_SOLVER", "GAMG_DIC")
    expected = int(cfg.get("STEPS", "0") or 0) * int(cfg.get("CORR", "0") or 0)
    recs = []
    for f in sorted(glob.glob(os.path.join(run, "csv", "bub*_*.csv"))):
        name = os.path.basename(f)[:-4]
        m = re.match(r"bub(\d+)_(.+)$", name)
        if not m:
            continue
        n, var = int(m.group(1)), m.group(2)
        code = S2L if var == "amg" else (OF if var == ofname else None)
        if code is None:
            continue
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        if code == S2L:
            s, its, per = summarize_s2l(rows, read_timers(os.path.join(run, "log." + name)))
            s["warmup"] = grep_float(os.path.join(run, "log." + name), r"Warm-up of the local direct solver[^\n]*max ([0-9.]+) s")
        else:
            s, its, per = summarize_of(rows)
        st_ = status.get(name, ("missing", 0.0))
        s.update(code=code, N=n, NP=np_ or int(rows[0]["np"]), run=os.path.basename(run.rstrip("/")), name=name,
                 status=st_[0], wall=st_[1], solves=len(rows), expected=expected or len(rows),
                 converged=sum(int(float(r["converged"])) for r in rows),
                 true_res=max(float(r["true_rel_res"]) for r in rows), its=its, per_solve=per,
                 its_avg=st.mean(its), its_min=min(its), its_max=max(its))
        s["total"] = s["init"] + s["upd"] + s["iter"]
        s["cost"] = s["total"] / s["solves"]
        s["steady"] = (s["upd"] + s["iter"]) / s["solves"]
        s["t_it"] = s["iter"] / max(1, sum(its))
        s["ok"] = s["status"] == "ok" and s["solves"] == s["expected"]
        recs.append(s)
    return cfg, recs


def job_key(run):
    m = re.search(r"_(\d+)$", run.rstrip("/"))
    return (int(m.group(1)) if m else 0, os.path.getmtime(run))


# ---------------------------------------------------------------------------------------------
def fmt(x, nd=3):
    if x is None:
        return "-"
    ax = abs(x)
    if ax >= 100:
        return "%.0f" % x
    if ax >= 10:
        return "%.1f" % x
    if ax >= 1:
        return "%.2f" % x
    return ("%%.%df" % nd) % x


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8.5)


def plots(outdir, data, Ns, NPs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter, LogLocator
    except ImportError:
        print("matplotlib not available: tables only")
        return []
    pdir = os.path.join(outdir, "plots")
    os.makedirs(pdir, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9.5, "axes.labelcolor": INK2,
                         "text.color": INK, "axes.titlesize": 10.5, "axes.titlecolor": INK,
                         "legend.frameon": False, "legend.fontsize": 8.5})
    made = []
    get = lambda c, n, p: data.get((c, n, p))
    sec = FuncFormatter(lambda v, _: fmt(v, 2))

    def save(fig, name):
        p = os.path.join(pdir, name)
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        made.append(p)

    def logy(ax):
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(sec)

    def npaxis(ax):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(NPs))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%d" % v))
        ax.set_xlim(NPs[0] / 1.12, NPs[-1] * 1.12)

    # 1. strong scaling: cost per solve and parallel efficiency vs NP, one column per N.
    if len(NPs) > 1:
        fig, axes = plt.subplots(2, len(Ns), figsize=(3.7 * len(Ns) + 0.6, 6.4), facecolor=SURFACE,
                                 squeeze=False, sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
        for j, n in enumerate(Ns):
            ax, ae = axes[0][j], axes[1][j]
            style_axes(ax)
            style_axes(ae)
            for c in (S2L, OF):
                pts = [(p, get(c, n, p)["cost"]) for p in NPs if get(c, n, p)]
                if not pts:
                    continue
                x, y = zip(*pts)
                ax.plot(x, y, color=COLOR[c], linewidth=2, marker="o", markersize=6, label=NAME[c],
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
                ax.plot(NPs, [y[0] * x[0] / p for p in NPs], color=COLOR[c], linewidth=1, linestyle=(0, (3, 3)),
                        alpha=0.7, zorder=2)
                ax.annotate(fmt(y[-1], 2) + " s", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK2)
                ae.plot(x, [y[0] * x[0] / (yy * xx) for xx, yy in zip(x, y)], color=COLOR[c], linewidth=2,
                        marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
            logy(ax)
            ax.set_title("%d³ (%.1fM cells)" % (n, n ** 3 / 1e6), loc="left")
            ae.set_ylim(0, 1.25)
            ae.axhline(1.0, color=AXIS, linewidth=1)
            npaxis(ae)
            ae.set_xlabel("processes")
            if j == 0:
                ax.set_ylabel("cost per solve [s]")
                ae.set_ylabel("parallel efficiency\n(vs %d processes)" % NPs[0])
        axes[0][0].legend(loc="lower left")
        fig.text(0.01, 0.005, "dashed: ideal strong scaling from %d processes; cost per solve = (initial setup + updates "
                 "+ iterations) / solves" % NPs[0], fontsize=7.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        save(fig, "scaling_strong.png")

    # 2. size scaling: cost per solve vs cells, one panel per NP.
    if len(Ns) > 1:
        fig, axes = plt.subplots(1, len(NPs), figsize=(3.7 * len(NPs) + 0.6, 3.8), facecolor=SURFACE, squeeze=False,
                                 sharey=True)
        cells = [n ** 3 / 1e6 for n in Ns]
        for j, p in enumerate(NPs):
            ax = axes[0][j]
            style_axes(ax)
            for c in (S2L, OF):
                pts = [(n ** 3 / 1e6, get(c, n, p)["cost"]) for n in Ns if get(c, n, p)]
                if not pts:
                    continue
                x, y = zip(*pts)
                ax.plot(x, y, color=COLOR[c], linewidth=2, marker="o", markersize=6, label=NAME[c],
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
                ax.plot(cells, [y[0] * cc / x[0] for cc in cells], color=COLOR[c], linewidth=1, linestyle=(0, (3, 3)),
                        alpha=0.7)
            ax.set_xscale("log")
            logy(ax)
            ax.xaxis.set_major_locator(FixedLocator(cells))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%.1fM" % v))
            ax.set_xlabel("cells (%s³)" % "³, ".join(str(n) for n in Ns))
            ax.set_title("%d processes" % p, loc="left")
            if j == 0:
                ax.set_ylabel("cost per solve [s]")
        axes[0][0].legend(loc="upper left")
        fig.text(0.01, 0.005, "dashed: cost proportional to the number of cells (optimal at fixed processes)",
                 fontsize=7.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        save(fig, "scaling_size.png")

    # 3. ratio GAMG / Schwarz of the cost per solve: diverging blue (Schwarz faster) - red (GAMG faster).
    import math
    ratio = [[(get(OF, n, p)["cost"] / get(S2L, n, p)["cost"]) if get(OF, n, p) and get(S2L, n, p) else None
              for p in NPs] for n in Ns]
    vals = [r for row in ratio for r in row if r]
    if vals:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
        lim = max(0.3, max(abs(math.log2(v)) for v in vals))
        fig, ax = plt.subplots(figsize=(1.25 * len(NPs) + 2.6, 0.75 * len(Ns) + 1.6), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        for i, n in enumerate(Ns):
            for j, p in enumerate(NPs):
                r = ratio[i][j]
                col = cmap(0.5 + 0.5 * math.log2(r) / lim) if r else SURFACE
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=col, edgecolor=SURFACE, linewidth=2))
                ax.text(j + 0.5, i + 0.5, ("%.2f×" % r) if r else "-", ha="center", va="center", fontsize=10,
                        color=INK)
        ax.set_xlim(0, len(NPs))
        ax.set_ylim(0, len(Ns))
        ax.set_xticks([j + 0.5 for j in range(len(NPs))])
        ax.set_xticklabels([str(p) for p in NPs])
        ax.set_yticks([i + 0.5 for i in range(len(Ns))])
        ax.set_yticklabels(["%d³" % n for n in Ns])
        ax.set_xlabel("processes")
        ax.tick_params(length=0, colors=INK2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title("cost per solve, GAMG / Schwarz\n(> 1: Schwarz faster)", loc="left", fontsize=10)
        fig.tight_layout()
        save(fig, "scaling_ratio.png")

    # 4. phases: seconds per solve, stacked; one panel per N, two bars (Schwarz, GAMG) per NP.
    fig, axes = plt.subplots(1, len(Ns), figsize=(3.9 * len(Ns) + 0.6, 4.6), facecolor=SURFACE, squeeze=False)
    for j, n in enumerate(Ns):
        ax = axes[0][j]
        style_axes(ax)
        ax.grid(axis="x", visible=False)
        xt, xl = [], []
        for k, p in enumerate(NPs):
            for d, c in ((-0.21, S2L), (0.21, OF)):
                s = get(c, n, p)
                if not s:
                    continue
                y0 = 0.0
                for key, lab, col in PHASES[c]:
                    h = s[key] / s["solves"]
                    ax.bar(k + d, h, bottom=y0, width=0.38, color=col, edgecolor=SURFACE, linewidth=1.5)
                    y0 += h
                ax.text(k + d, y0, fmt(y0, 2), ha="center", va="bottom", fontsize=7.5, color=INK2)
            xt.append(k)
            xl.append("%d\nS   G" % p)
        ax.set_xticks(xt)
        ax.set_xticklabels(xl)
        ax.tick_params(axis="x", length=0)
        ax.set_title("%d³" % n, loc="left")
        ax.set_xlabel("processes  (S = Schwarz, G = GAMG)")
        if j == 0:
            ax.set_ylabel("seconds per solve")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    handles, labels = [], []
    rows = max(len(PHASES[S2L]), len(PHASES[OF]))
    for c in (S2L, OF):  # one legend column per code
        for key, lab, col in PHASES[c]:
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=col, edgecolor=SURFACE))
            labels.append(("Schwarz: " if c == S2L else "GAMG: ") + lab)
        for _ in range(rows - len(PHASES[c])):
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="none"))
            labels.append("")
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    save(fig, "scaling_phases.png")

    # 5. iterations of every solve, one panel per N; line style = processes.
    fig, axes = plt.subplots(1, len(Ns), figsize=(3.9 * len(Ns) + 0.6, 3.8), facecolor=SURFACE, squeeze=False,
                             sharey=True)
    for j, n in enumerate(Ns):
        ax = axes[0][j]
        style_axes(ax)
        for c in (S2L, OF):
            for k, p in enumerate(NPs):
                s = get(c, n, p)
                if s:
                    ax.plot(range(1, len(s["its"]) + 1), s["its"], color=COLOR[c], linewidth=1.6,
                            linestyle=NP_STYLE[k % len(NP_STYLE)])
        ax.set_title("%d³" % n, loc="left")
        ax.set_xlabel("solve (steps × correctors)")
        if j == 0:
            ax.set_ylabel("iterations per solve\n(GMRES / V-cycles)")
    itmax = max([max(s["its"]) for s in data.values()] or [1])
    axes[0][0].set_ylim(0, itmax * 1.08)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOR[c], linewidth=2) for c in (S2L, OF)]
    h += [Line2D([], [], color=INK2, linewidth=1.6, linestyle=NP_STYLE[k % len(NP_STYLE)]) for k in range(len(NPs))]
    fig.legend(h, [NAME[S2L], NAME[OF]] + ["%d processes" % p for p in NPs], loc="lower center",
               ncol=2 + len(NPs), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "scaling_iterations.png")
    return made


# ---------------------------------------------------------------------------------------------
def main(argv):
    args, outdir = [], None
    it = iter(argv)
    for a in it:
        if a == "--out":
            outdir = next(it)
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            args.append(a.rstrip("/"))
    runs = []
    for a in args or ["."]:
        if os.path.exists(os.path.join(a, "config.txt")):
            runs.append(a)
        else:
            runs += sorted(d for d in glob.glob(os.path.join(a, "run_scaling_np*_*")) if os.path.isdir(d))
    if not runs:
        print(__doc__)
        print("no run directories found")
        return 1
    runs.sort(key=job_key)
    outdir = outdir or os.path.join(os.path.dirname(os.path.abspath(runs[0])), "scaling_report")
    os.makedirs(outdir, exist_ok=True)

    data, cfgs, replaced, notok = {}, OrderedDict(), [], []
    for run in runs:  # oldest first: newer runs replace
        cfg, recs = load_run(run)
        cfgs[run] = cfg
        for s in recs:
            if not s["ok"]:
                notok.append(s)
                continue
            k = (s["code"], s["N"], s["NP"])
            if k in data:
                replaced.append((data[k]["run"], s["run"], s["name"]))
            data[k] = s
    Ns = sorted({k[1] for k in data})
    NPs = sorted({k[2] for k in data})
    get = lambda c, n, p: data.get((c, n, p))

    out = []
    w = out.append
    w("# Scaling campaign: GAMG vs two-level Schwarz\n")
    c0 = next(iter(cfgs.values()))
    w("Runs: " + ", ".join("`%s`" % os.path.basename(r) for r in runs) + "  ")
    w("Problem: bubbles3d, %s bubbles, density ratio %s, CFL %s; %s steps × %s correctors; every solve from 0.  " % (
        c0.get("NBUB"), c0.get("RHO"), c0.get("CFL"), c0.get("STEPS"), c0.get("CORR")))
    w("Stopping test (both codes): OpenFOAM normalised residual, relTol **%s** (Σ|b − Ax| ≤ relTol Σ|b|).  " % c0.get("RTOL"))
    w("Schwarz: `%s %s --stop-norm openfoam`; OpenFOAM: `%s` (%s).\n" % (
        c0.get("BASE"), c0.get("COARSE"), c0.get("OF_SOLVER"), "OpenFOAM-" + c0.get("OpenFOAM", "?")))
    diff = [r for r, c in cfgs.items() if {k: v for k, v in c.items() if k not in ("NP", "nodes", "job")} !=
            {k: v for k, v in c0.items() if k not in ("NP", "nodes", "job")}]
    if diff:
        w("**Warning**: the configuration differs in " + ", ".join(os.path.basename(r) for r in diff) + "\n")

    w("## Checks\n")
    for r in runs:
        b = os.path.join(r, "binaries.txt")
        if os.path.exists(b):
            w("- `%s` binaries: %s" % (os.path.basename(r), open(b).read().strip()))
    alls = list(data.values())
    if alls:
        w("- solves: %d, converged %d; final normalised residual (max): Schwarz %.2e, GAMG %.2e; "
          "true 2-norm residual ||b − Ax||₂/||b||₂ (max): Schwarz %.2e, GAMG %.2e" % (
              sum(s["solves"] for s in alls), sum(s["converged"] for s in alls),
              max([s["final_of"] for s in alls if s["code"] == S2L] or [0]),
              max([s["final_of"] for s in alls if s["code"] == OF] or [0]),
              max([s["true_res"] for s in alls if s["code"] == S2L] or [0]),
              max([s["true_res"] for s in alls if s["code"] == OF] or [0])))
    w("- runs not used (not ok or incomplete): " + (", ".join("%s/%s (%s, %d/%d solves)" % (
        s["run"], s["name"], s["status"], s["solves"], s["expected"]) for s in notok) if notok else "none"))
    if replaced:
        w("- replaced by newer runs: " + ", ".join("%s from %s by %s" % (n, a, b) for a, b, n in replaced))
    # Anomaly 1 (cold first factorization): warm-up times and initial setup vs rebuilds.
    wu = []
    for r in runs:
        t = grep_float(os.path.join(r, "log.warmup"), r"Warm-up of the local direct solver[^\n]*max ([0-9.]+) s")
        if t is not None:
            wu.append("%s %.2f s" % (os.path.basename(r), t))
    if wu:
        w("- warm-up of the direct solver in the first (unmeasured) run of each job: " + ", ".join(wu))
    rows = [s for s in alls if s["code"] == S2L]
    if rows:
        w("- warm-up in the measured runs (max): %s s; initial setup / median rebuild: %s "
          "(≈ 1 expected; 11 in the first schwarz2lvl run of run_gamg_23038420: cold first factorization)" % (
              fmt(max(s["warmup"] or 0 for s in rows), 2),
              ", ".join("%d³/%d %.2f" % (s["N"], s["NP"], s["init"] / s["rebuild_med"]) for s in sorted(
                  rows, key=lambda s: (s["N"], s["NP"])) if s["rebuild_med"]) or "no rebuilds"))
    w("")

    w("## Cost per solve [s] (setup amortized over the solves of the run)\n")
    w("| N \\ processes | " + " | ".join("%d" % p for p in NPs) + " |")
    w("|:--|" + "--:|" * len(NPs))
    for n in Ns:
        cells = []
        for p in NPs:
            a, g = get(S2L, n, p), get(OF, n, p)
            if a and g:
                cells.append("S %s · G %s · **%.2f×**" % (fmt(a["cost"]), fmt(g["cost"]), g["cost"] / a["cost"]))
            elif a or g:
                cells.append(("S %s" % fmt(a["cost"])) if a else ("G %s" % fmt(g["cost"])))
            else:
                cells.append("-")
        w("| %d³ (%.1fM) | %s |" % (n, n ** 3 / 1e6, " | ".join(cells)))
    w("\nS = two-level Schwarz (AMG local solves), G = GAMG (%s); **ratio** = G / S (> 1: Schwarz faster).\n"
      % c0.get("OF_SOLVER"))

    w("## Details\n")
    w("| N | NP | code | its avg (min-max) | time/it [ms] | initial setup [s] | updates [s] | iterations [s] | "
      "cost/solve [s] | steady/solve [s] | wall [s] |")
    w("|--:|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
    for n in Ns:
        for p in NPs:
            for c in (S2L, OF):
                s = get(c, n, p)
                if not s:
                    continue
                upd = fmt(s["upd"])
                if c == S2L:
                    upd += " (%d reb., %d ref.)" % (s["rebuilds"], s["refact"])
                w("| %d | %d | %s | %.1f (%d-%d) | %.1f | %s | %s | %s | %s | %s | %.0f |" % (
                    n, p, "Schwarz" if c == S2L else "GAMG", s["its_avg"], s["its_min"], s["its_max"],
                    1e3 * s["t_it"], fmt(s["init"]), upd, fmt(s["iter"]), fmt(s["cost"]), fmt(s["steady"]),
                    s["wall"]))
    w("\n*steady/solve* = (updates + iterations) / solves, without the initial setup (the cost per solve of a "
      "long simulation, if the rebuild frequency stays the same). *wall* = whole run, with start-up, mesh "
      "reading / matrix generation and assembly (not compared).\n")

    w("## Phases [s per solve]\n")
    w("| N | NP | Schwarz: initial setup | updates | local solves | coarse corr. | Krylov | GAMG: initial setup | "
      "updates | V-cycles |")
    w("|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for n in Ns:
        for p in NPs:
            a, g = get(S2L, n, p), get(OF, n, p)
            fa = [fmt(a[k] / a["solves"]) if a else "-" for k in ("init", "upd", "local", "coarse", "krylov")]
            fg = [fmt(g[k] / g["solves"]) if g else "-" for k in ("init", "upd", "cycles")]
            w("| %d | %d | %s | %s |" % (n, p, " | ".join(fa), " | ".join(fg)))
    w("\nSchwarz iterations split with the timer averages of `log.bub<N>_amg` (Krylov = SpMV, orthogonalization, "
      "stopping test and vector updates of GMRES).\n")

    if len(NPs) > 1:
        w("## Strong scaling: parallel efficiency vs %d processes\n" % NPs[0])
        w("| N | code | " + " | ".join("%d" % p for p in NPs) + " |")
        w("|--:|:--|" + "--:|" * len(NPs))
        for n in Ns:
            for c in (S2L, OF):
                b = get(c, n, NPs[0])
                if not b:
                    continue
                w("| %d | %s | %s |" % (n, "Schwarz" if c == S2L else "GAMG", " | ".join(
                    ("%.2f" % (b["cost"] * NPs[0] / (get(c, n, p)["cost"] * p))) if get(c, n, p) else "-" for p in NPs)))
        w("")

    made = plots(outdir, data, Ns, NPs) if data else []
    for p in made:
        w("![%s](plots/%s)\n" % (os.path.basename(p)[:-4], os.path.basename(p)))
    open(os.path.join(outdir, "report.md"), "w").write("\n".join(out) + "\n")

    keys = ["code", "N", "NP", "run", "status", "solves", "converged", "its_avg", "its_min", "its_max", "t_it", "init",
            "upd", "iter", "local", "coarse", "krylov", "cycles", "rebuilds", "refact", "total", "cost", "steady",
            "final_of", "true_res", "wall", "t_generate"]
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for k in sorted(data):
            s = data[k]
            wr.writerow([s.get(x, "") for x in keys])
    print("\n".join(out))
    print("\nwritten: %s (report.md, summary.csv, %d plots)" % (outdir, len(made)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
