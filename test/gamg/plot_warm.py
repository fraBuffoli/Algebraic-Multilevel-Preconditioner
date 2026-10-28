#!/usr/bin/env python3
"""Plots of the warm-start campaign (Allrun_warm.sh): iterations and time of every phase,
two-level / one-level Schwarz against OpenFOAM, on the same systems.

    python3 plot_warm.py run_warm_np240_<jobid> run_warm_np432_<jobid> [--out DIR] [--steady]
                         [--with-1L] [--format png|pdf]

One row of panels per process count (one run directory each), one column per mesh size or per
corrector. Writes into DIR (default: warm_plots next to the first run):

  iterations.png        iterations per solve for the first corrector, the second one and the last
                        one of every step (marker: mean; line: min - max), log scale
  last_corrector.png    iterations of the last corrector of every step along the time loop
                        (aging of the coarse space; vertical lines: rebuilds of the two-level Schwarz)
  phases.png            cost per time step split by phase:
                          Schwarz  setup: eigensolver / setup: rest (initial setup and rebuilds),
                                   refactorizations, iterations: local solves / coarse correction /
                                   Krylov (SpMV, vector operations, stopping test)
                          OpenFOAM setup (coarse matrices of GAMG) / iterations
  correctors.png        cost per time step split by corrector (updates and setup apart)
  ratio.png             best OpenFOAM variant / two-level Schwarz, cost per step: iterations only,
                        + refactorizations, + setup and rebuilds (all costs)
  time_per_iteration.png  ms per application of the preconditioner / per V-cycle or PCG iteration
  phases.csv            the numbers of phases.png and correctors.png

--steady leaves the first step out of every cost (its initial setup and its solve from 0): what a
long run pays per step. --with-1L adds the one-level RAS to phases.png, correctors.png and
last_corrector.png (its last corrector costs 5-6x the other variants and flattens the axes).

Times (csv/<run>.csv): schwarz2lvl t_update (initial setup, rebuild, refactorization, reuse) and
t_solve; OpenFOAM t_setup and t_cycles; all between two synchronizations of all processes. The
setup and the iterations of schwarz2lvl are split with the timer table at the end of log.<run>
(averages over the processes): eigensolver share of setup.*, local solves
(solve.preconditioner.local), coarse correction (solve.preconditioner.coarse + .spmv), Krylov = the
rest of solve. Matplotlib only (no numpy).
"""
import csv
import glob
import os
import re
import sys
from collections import OrderedDict

# ---- palette (dataviz reference instance, light mode): categorical slots in fixed order --------
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ORDER = ["s2l_2L", "W_GAMG", "s2l_1L", "W_GAMG_PCG", "W_PCG_DIC"]       # color follows the variant
COLOR = {"s2l_2L": "#2a78d6", "W_GAMG": "#eb6834", "s2l_1L": "#1baf7a", "W_GAMG_PCG": "#eda100",
         "W_PCG_DIC": "#e87ba4"}
MARKER = {"s2l_2L": "o", "W_GAMG": "s", "s2l_1L": "^", "W_GAMG_PCG": "D", "W_PCG_DIC": "v"}
LABEL = {"s2l_2L": "Schwarz 2-level", "W_GAMG": "GAMG (DIC)", "s2l_1L": "RAS 1-level",
         "W_GAMG_PCG": "GAMG, PCG+GAMG last", "W_PCG_DIC": "PCG + DIC"}
CORR_LABEL = ["first corrector", "second corrector", "last corrector"]


def mix(c, other, t):
    """Color c moved by t (0..1) towards other."""
    a = [int(c[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(other[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join("%02x" % round(x + (y - x) * t) for x, y in zip(a, b))


def ramp(c, n):
    """n ordinal steps of the hue of c, dark to light (setup first, iterations last)."""
    if n == 1:
        return [c]
    hi = 0.0 if n == 2 else 0.40               # two parts: shade + the hue itself (the large one)
    out = []
    for k in range(n):
        t = -0.45 + (hi + 0.45) * k / (n - 1)    # -0.45 (shade) .. hi (tint)
        out.append(mix(c, "#000000", -t) if t < 0 else mix(c, "#ffffff", t))
    return out


# ---- data -------------------------------------------------------------------------------------
def read_config(run):
    cfg = {}
    p = os.path.join(run, "config.txt")
    if os.path.exists(p):
        for tok in re.findall(r"(\w+)=('[^']*'|\S+)", open(p).read()):
            cfg[tok[0]] = tok[1].strip("'")
    return cfg


def read_timers(log):
    """Timer table at the end of a schwarz2lvl log: phase -> average over the processes."""
    t, on = {}, False
    if not os.path.exists(log):
        return t
    for line in open(log, errors="replace"):
        if line.startswith("Timings [s] over"):
            on = True
            continue
        if on:
            f = line.split()
            if len(f) == 4 and f[0] != "phase":
                try:
                    t[f[0]] = float(f[2])
                except ValueError:
                    break
            elif f and f[0] != "phase":
                break
    return t


def load(run, steady):
    cfg = read_config(run)
    np_ = int(cfg.get("NP", "0") or 0)
    out = OrderedDict()
    for f in sorted(glob.glob(os.path.join(run, "csv", "bub*_*.csv"))):
        m = re.match(r"bub(\d+)_(.+)\.csv$", os.path.basename(f))
        if not m:
            continue
        N, var = int(m.group(1)), m.group(2)
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        np_ = np_ or int(rows[0]["np"])
        first = 1 if steady else 0  # first step kept
        keep = [r for r in rows if int(r["step"]) >= first]
        steps = len({int(r["step"]) for r in keep})
        ncorr = max(int(r["corrector"]) for r in rows) + 1
        kind = lambda r: 2 if int(r["corrector"]) == ncorr - 1 else min(int(r["corrector"]), 1)  # noqa: E731
        s = dict(var=var, N=N, NP=np_, steps=steps, step=[int(r["step"]) for r in rows],
                 its=[int(float(r["iterations"])) for r in rows], kind=[kind(r) for r in rows],
                 converged=sum(int(float(r["converged"])) for r in rows), solves=len(rows))
        corr_t = [0.0, 0.0, 0.0]
        if var.startswith("s2l_"):
            tm = read_timers(os.path.join(run, "log.bub%d_%s" % (N, var)))
            setup = sum(float(r["t_update"]) for r in keep if r["level"] == "rebuild")
            upd = sum(float(r["t_update"]) for r in keep if r["level"] != "rebuild")
            it = sum(float(r["t_solve"]) for r in keep)
            for r in keep:
                corr_t[kind(r)] += float(r["t_solve"])
            f_eig = tm.get("setup.eigensolver", 0.0) / tm["setup"] if tm.get("setup") else 0.0
            tot = tm.get("solve", 0.0)
            if tot > 0 and "solve.preconditioner.local" in tm:
                f_loc = tm["solve.preconditioner.local"] / tot
                f_crs = (tm.get("solve.preconditioner.coarse", 0.0) + tm.get("solve.preconditioner.spmv", 0.0)) / tot
            elif tot > 0:  # one level: the preconditioner is the local solves
                f_loc, f_crs = tm.get("solve.preconditioner", 0.0) / tot, 0.0
            else:
                f_loc = f_crs = 0.0
            ph = OrderedDict([("setup: eigensolver", setup * f_eig), ("setup: rest", setup * (1 - f_eig)),
                              ("refactorizations", upd), ("iterations: local solves", it * f_loc),
                              ("iterations: coarse correction", it * f_crs),
                              ("iterations: Krylov", it * (1 - f_loc - f_crs))])
            if var == "s2l_1L":
                del ph["setup: eigensolver"], ph["iterations: coarse correction"]
                ph["setup"] = ph.pop("setup: rest")
                ph.move_to_end("setup", last=False)
            s["rebuild_steps"] = [int(r["step"]) for r in rows[1:] if r["level"] == "rebuild"]
            s["setup"], s["updates"], s["iter"] = setup, upd, it
        else:
            setup = sum(float(r["t_setup"]) for r in keep)
            it = sum(float(r["t_cycles"]) for r in keep)
            for r in keep:
                corr_t[kind(r)] += float(r["t_cycles"])
            ph = OrderedDict([("setup (coarse matrices)", setup), ("iterations", it)])
            s["rebuild_steps"] = []
            s["setup"], s["updates"], s["iter"] = setup, 0.0, it
        s["phases"] = OrderedDict((k, v / steps) for k, v in ph.items())
        s["corr_t"] = [v / steps for v in corr_t]
        s["total"] = sum(s["phases"].values())
        s["t_it"] = 1e3 * s["iter"] / max(1, sum(i for i, k in zip(s["its"], s["step"]) if k >= first))
        out[(N, var)] = s
    return np_, out


# ---- plotting helpers --------------------------------------------------------------------------
def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8.5)


def variants_of(data, allowed=None):
    vs = {v for (_, v) in data}
    return [v for v in ORDER if v in vs and (allowed is None or v in allowed)] + \
           sorted(v for v in vs if v not in ORDER and (allowed is None or v in allowed))


def header(fig, title, handles, ncol, legend_below=False):
    """Title on top, legend under it (or at the bottom); returns the tight_layout rect."""
    h = fig.get_figheight()
    fig.suptitle(title, y=1 - 0.12 / h, va="top", fontsize=10.5, color=INK)
    if legend_below:
        rows = (len(handles) + ncol - 1) // ncol
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False,
                   fontsize=7.5)
        return (0, (0.22 * rows + 0.1) / h, 1, 1 - 0.42 / h)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.40 / h), ncol=ncol, frameon=False,
               fontsize=8.5)
    return (0, 0, 1, 1 - 0.72 / h)


def plain_log(ax):
    """Log y axis with plain tick labels (2, 5, 10, 20, ...)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax.yaxis.set_minor_formatter(NullFormatter())


def save(fig, out, name, fmt):
    p = os.path.join(out, "%s.%s" % (name, fmt))
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    return p


def main(argv):
    runs, out, steady, with1, fmt = [], None, False, False, "png"
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--out":
            out, i = argv[i + 1], i + 2
        elif a == "--format":
            fmt, i = argv[i + 1], i + 2
        elif a == "--steady":
            steady, i = True, i + 1
        elif a == "--with-1L":
            with1, i = True, i + 1
        else:
            runs.append(a)
            i += 1
    runs = [r for r in runs if os.path.isdir(os.path.join(r, "csv"))]
    if not runs:
        print(__doc__)
        return 1
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib not available")
        return 1
    out = out or os.path.join(os.path.dirname(os.path.abspath(runs[0].rstrip("/"))), "warm_plots")
    os.makedirs(out, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
                         "font.family": "sans-serif"})

    sets = []                               # [(NP, {(N, var): summary})], by process count
    for r in runs:
        sets.append(load(r, steady))
    sets.sort(key=lambda x: x[0])
    Ns = sorted({N for _, d in sets for (N, _) in d})
    allv = variants_of({k: 0 for _, d in sets for k in d})
    main_v = [v for v in allv if v != "s2l_1L" or with1]
    tag = " (steps 1.., without the first step)" if steady else ""
    written = []

    # 1. Iterations per corrector kind: marker = mean, line = min - max, log scale.
    fig, axs = plt.subplots(len(sets), 3, figsize=(10.5, 3.0 * len(sets)), squeeze=False, facecolor=SURFACE)
    for row, (np_, d) in enumerate(sets):
        for k in range(3):
            ax = axs[row][k]
            style(ax)
            w = 0.8 / max(1, len(allv))
            for j, v in enumerate(allv):
                for x, N in enumerate(Ns):
                    s = d.get((N, v))
                    if not s:
                        continue
                    its = [i for i, kk, st in zip(s["its"], s["kind"], s["step"]) if kk == k and (st > 0 or not steady)]
                    if not its:
                        continue
                    xx = x - 0.4 + w * (j + 0.5)
                    ax.plot([xx, xx], [max(min(its), 0.8), max(its)], color=COLOR.get(v, MUTED), linewidth=1.6,
                            solid_capstyle="round")
                    ax.plot(xx, sum(its) / len(its), MARKER.get(v, "o"), color=COLOR.get(v, MUTED), markersize=6,
                            markeredgecolor=SURFACE, markeredgewidth=1.2)
            plain_log(ax)
            ax.set_xticks(range(len(Ns)))
            ax.set_xticklabels(["%d³" % N for N in Ns])
            if row == 0:
                ax.set_title(CORR_LABEL[k], fontsize=10, color=INK)
            if k == 0:
                ax.set_ylabel("%d processes\niterations per solve" % np_)
    handles = [Line2D([], [], marker=MARKER.get(v, "o"), color=COLOR.get(v, MUTED), linewidth=1.6, markersize=6,
                      markeredgecolor=SURFACE, label=LABEL.get(v, v)) for v in allv]
    fig.tight_layout(rect=header(fig, "Iterations per solve: mean (marker) and min - max (line)" + tag, handles,
                                 len(handles)))
    written.append(save(fig, out, "iterations", fmt))
    plt.close(fig)

    # 2. Last corrector along the time loop: aging of the coarse space.
    lv = [v for v in allv if v != "s2l_1L" or with1]
    fig, axs = plt.subplots(len(sets), len(Ns), figsize=(3.6 * len(Ns), 2.9 * len(sets)), squeeze=False,
                            facecolor=SURFACE)
    for row, (np_, d) in enumerate(sets):
        for col, N in enumerate(Ns):
            ax = axs[row][col]
            style(ax)
            for v in lv:
                s = d.get((N, v))
                if not s:
                    continue
                pts = [(st, i) for i, kk, st in zip(s["its"], s["kind"], s["step"]) if kk == 2]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOR.get(v, MUTED), linewidth=1.6,
                        marker=MARKER.get(v, "o"), markersize=3.5, markeredgecolor=SURFACE, markeredgewidth=0.6)
                if v == "s2l_2L":
                    for rb in s["rebuild_steps"]:
                        ax.axvline(rb, color=COLOR[v], linewidth=0.8, alpha=0.45)
            if with1:
                plain_log(ax)
            else:
                ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title("%d³" % N, fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel("%d processes\niterations, last corrector" % np_)
            if row == len(sets) - 1:
                ax.set_xlabel("time step")
    handles = [Line2D([], [], marker=MARKER.get(v, "o"), color=COLOR.get(v, MUTED), linewidth=1.6, markersize=5,
                      markeredgecolor=SURFACE, label=LABEL.get(v, v)) for v in lv]
    handles.append(Line2D([], [], color=COLOR["s2l_2L"], linewidth=0.8, alpha=0.45, label="rebuild of the coarse space"))
    fig.tight_layout(rect=header(fig, "Last corrector of every step (tolerance only): aging of the coarse space",
                                 handles, len(handles)))
    written.append(save(fig, out, "last_corrector", fmt))
    plt.close(fig)

    # 3. Cost per step by phase (stacked), and 4. by corrector.
    def stacked(name, title, parts_of):
        fig, axs = plt.subplots(len(sets), len(Ns), figsize=(3.7 * len(Ns), 3.3 * len(sets)), squeeze=False,
                                facecolor=SURFACE)
        legend = OrderedDict()
        for row, (np_, d) in enumerate(sets):
            for col, N in enumerate(Ns):
                ax = axs[row][col]
                style(ax)
                vs = [v for v in main_v if (N, v) in d]
                for x, v in enumerate(vs):
                    parts = parts_of(d[(N, v)])
                    cols = ramp(COLOR.get(v, MUTED), len(parts))
                    bottom = 0.0
                    for (lab, val), c in zip(parts.items(), cols):
                        ax.bar(x, val, bottom=bottom, width=0.62, color=c, edgecolor=SURFACE, linewidth=1.5)
                        bottom += val
                        legend.setdefault(v, OrderedDict())[lab] = c
                    ax.text(x, bottom, "%.2f" % bottom, ha="center", va="bottom", fontsize=8, color=INK2)
                ax.set_xticks(range(len(vs)))
                ax.set_xticklabels([LABEL.get(v, v).replace(", ", ",\n").replace(" (", "\n(") for v in vs], fontsize=7.5)
                ax.margins(y=0.12)
                if row == 0:
                    ax.set_title("%d³" % N, fontsize=10, color=INK)
                if col == 0:
                    ax.set_ylabel("%d processes\ncost per time step [s]" % np_)
        handles = []
        for v, parts in legend.items():
            handles += [Patch(facecolor=c, edgecolor=SURFACE, label="%s: %s" % (LABEL.get(v, v), lab))
                        for lab, c in parts.items()]
        fig.tight_layout(rect=header(fig, title + tag, handles, 3, legend_below=True))
        written.append(save(fig, out, name, fmt))
        plt.close(fig)

    stacked("phases", "Cost per time step by phase (all correctors, setup and updates amortized)",
            lambda s: s["phases"])

    def by_corrector(s):
        p = OrderedDict()
        p["setup and updates"] = s["total"] - sum(s["corr_t"])
        for k in range(3):
            p["iterations, " + CORR_LABEL[k]] = s["corr_t"][k]
        return p

    stacked("correctors", "Cost per time step by corrector", by_corrector)

    # 5. Ratio best OpenFOAM / two-level Schwarz, built up phase by phase.
    fig, axs = plt.subplots(1, len(sets), figsize=(4.4 * len(sets), 3.2), squeeze=False, facecolor=SURFACE)
    steps3 = ramp(COLOR["s2l_2L"], 3)
    labels3 = ["iterations only", "+ refactorizations", "+ setup and rebuilds (all costs)"]
    for col, (np_, d) in enumerate(sets):
        ax = axs[0][col]
        style(ax)
        for x, N in enumerate(Ns):
            s = d.get((N, "s2l_2L"))
            of = [d[(N, v)]["total"] for v in allv if not v.startswith("s2l_") and (N, v) in d]
            if not s or not of:
                continue
            best = min(of)
            k = s["steps"]
            vals = [best / (s["iter"] / k), best / ((s["iter"] + s["updates"]) / k), best / s["total"]]
            for j, (val, c) in enumerate(zip(vals, steps3)):
                xx = x - 0.27 + 0.27 * j
                ax.bar(xx, val, width=0.25, color=c, edgecolor=SURFACE, linewidth=1.5)
                if j == 2:
                    ax.text(xx, val, "%.2f×" % val, ha="center", va="bottom", fontsize=8, color=INK)
        ax.axhline(1.0, color=INK2, linewidth=0.9)
        ax.axhline(1.5, color=MUTED, linewidth=0.9)
        ax.text(-0.55, 1.5, "1.5×", va="bottom", ha="left", fontsize=7.5, color=MUTED)
        ax.set_xticks(range(len(Ns)))
        ax.set_xticklabels(["%d³" % N for N in Ns])
        ax.set_title("%d processes" % np_, fontsize=10, color=INK)
        if col == 0:
            ax.set_ylabel("best OpenFOAM / Schwarz 2-level")
    fig.tight_layout(rect=header(fig, "Where the advantage goes: cost per step, ratio > 1 = Schwarz faster" + tag,
                                 [Patch(facecolor=c, label=l) for c, l in zip(steps3, labels3)], 3))
    written.append(save(fig, out, "ratio", fmt))
    plt.close(fig)

    # 6. Time per iteration.
    fig, axs = plt.subplots(1, len(sets), figsize=(4.4 * len(sets), 3.1), squeeze=False, facecolor=SURFACE)
    for col, (np_, d) in enumerate(sets):
        ax = axs[0][col]
        style(ax)
        for v in allv:
            xs = [x for x, N in enumerate(Ns) if (N, v) in d]
            if not xs:
                continue
            ys = [d[(Ns[x], v)]["t_it"] for x in xs]
            ax.plot(xs, ys, color=COLOR.get(v, MUTED), marker=MARKER.get(v, "o"), linewidth=1.6, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, label=LABEL.get(v, v))
        ax.set_xticks(range(len(Ns)))
        ax.set_xticklabels(["%d³" % N for N in Ns])
        plain_log(ax)
        ax.set_title("%d processes" % np_, fontsize=10, color=INK)
        if col == 0:
            ax.set_ylabel("ms per iteration")
    h, _ = axs[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=header(fig, "Time per iteration (application of the preconditioner / V-cycle / PCG iteration)",
                                 h, len(h)))
    written.append(save(fig, out, "time_per_iteration", fmt))
    plt.close(fig)

    # Table: phases.csv and a short summary on stdout.
    with open(os.path.join(out, "phases.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["NP", "N", "variant", "steps", "phase", "seconds_per_step"])
        for np_, d in sets:
            for (N, v), s in sorted(d.items()):
                for lab, val in s["phases"].items():
                    w.writerow([np_, N, v, s["steps"], lab, "%.6g" % val])
                for k in range(3):
                    w.writerow([np_, N, v, s["steps"], "iterations, " + CORR_LABEL[k], "%.6g" % s["corr_t"][k]])
    print("cost per time step [s]%s" % tag)
    for np_, d in sets:
        for N in Ns:
            line = ["%4d %4d³" % (np_, N)]
            for v in allv:
                s = d.get((N, v))
                if s:
                    line.append("%s %.3f" % (LABEL.get(v, v), s["total"]))
            print("  " + " | ".join(line))
    written.append(os.path.join(out, "phases.csv"))
    print("\n".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
