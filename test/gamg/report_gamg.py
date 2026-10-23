#!/usr/bin/env python3
"""Summary of a GAMG-reference campaign (Allrun_gamg.sh): OpenFOAM solvers vs schwarz2lvl.

    python3 report_gamg.py run_gamg_<jobid>

Reads <run>/csv/<group>_<variant>.csv (one line per solve; group = bub192, bub256), <run>/variants.txt
("<variant> <openfoam|schwarz2lvl> <options>"), <run>/status.txt, <run>/config.txt and the check
logs, and writes <run>/report.md and, if matplotlib is available, <run>/plots/gamg_cost.png (cost
per solve of every variant) and <run>/plots/gamg_solves.png (time of every solve, largest group).

Costs, per run (all solves of all steps):
  * total = sum of the solve times (OpenFOAM: GAMG agglomerates its coarse matrices in every solve,
    the agglomeration itself is cached after the first); schwarz2lvl: sum of setup/update + solve
    times (initial setup in the first solve, then refactorizations / rebuilds of the reuse policy);
  * cost/solve = total / solves; median = median cost of the solves after the first;
  * first = first solve (schwarz2lvl: initial setup included).
Every solve starts from 0. Iterations: GAMG V-cycles, PCG / GMRES iterations (not comparable).
"""
import csv
import os
import statistics as st
import sys
from collections import OrderedDict

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
KIND_COLOR = {"openfoam": CAT[0], "schwarz2lvl": CAT[1]}
LABEL = {
    "GAMG_GS": "GAMG, GaussSeidel",
    "GAMG_DIC": "GAMG, DIC",
    "GAMG_ALG": "GAMG, algebraicPair",
    "PCG_GAMG": "PCG + GAMG",
    "PCG_DIC": "PCG + DIC",
    "amg": "Schwarz, AMG local",
    "exact": "Schwarz, exact local",
}
BASELINE = "GAMG_GS"
# Fixed colour per variant (the colour follows the variant, whatever the runs present).
SERIES_COLOR = {v: CAT[i] for i, v in enumerate(["GAMG_GS", "GAMG_DIC", "GAMG_ALG", "PCG_GAMG", "PCG_DIC", "amg", "exact"])}


def read_variants(run):
    v = OrderedDict()
    p = os.path.join(run, "variants.txt")
    if os.path.exists(p):
        for line in open(p):
            parts = line.split(None, 2)
            if len(parts) >= 2:
                v[parts[0]] = (parts[1], parts[2].strip() if len(parts) > 2 else "")
    return v


def summarize(rows, kind):
    if not rows:
        return None
    if kind == "openfoam":
        cost = [float(r["t_solve"]) for r in rows]
        upd = [0.0] * len(rows)
    else:
        cost = [float(r["t_update"]) + float(r["t_solve"]) for r in rows]
        upd = [float(r["t_update"]) for r in rows]
    its = [int(float(r["iterations"])) for r in rows]
    s = OrderedDict()
    s["solves"] = len(rows)
    s["converged"] = sum(int(float(r["converged"])) for r in rows)
    s["max_res"] = max(float(r["true_rel_res"]) for r in rows)
    s["its_avg"] = st.mean(its)
    s["its_max"] = max(its)
    s["first"] = cost[0]
    s["total"] = sum(cost)
    s["cost"] = s["total"] / len(rows)
    s["median"] = st.median(cost[1:]) if len(cost) > 1 else cost[0]
    s["series"] = cost
    if kind == "schwarz2lvl":
        s["rebuilds"] = sum(1 for r in rows[1:] if r["level"] == "rebuild")
        s["refact"] = sum(1 for r in rows[1:] if r["level"] == "refactorize")
        s["update_share"] = sum(upd) / s["total"] if s["total"] > 0 else 0.0
    return s


def load(run, variants):
    groups = OrderedDict()
    csvdir = os.path.join(run, "csv")
    names = sorted(os.listdir(csvdir)) if os.path.isdir(csvdir) else []
    order = list(variants)
    for f in names:
        if not f.endswith(".csv") or "_" not in f:
            continue
        group, v = f[:-4].split("_", 1)
        kind = variants.get(v, ("schwarz2lvl" if v in ("amg", "exact") else "openfoam", ""))[0]
        rows = list(csv.DictReader(open(os.path.join(csvdir, f))))
        s = summarize(rows, kind)
        if s:
            s["kind"] = kind
            groups.setdefault(group, {})[v] = s
    key = lambda g: int("".join(c for c in g if c.isdigit()) or 0)
    out = OrderedDict()
    for g in sorted(groups, key=key):
        items = groups[g]
        out[g] = OrderedDict((v, items[v]) for v in sorted(items, key=lambda v: order.index(v) if v in order else 99))
    return out


def fmt(x, nd=3):
    if x is None:
        return "-"
    if x >= 100:
        return "%.0f" % x
    if x >= 10:
        return "%.1f" % x
    return ("%%.%df" % nd) % x


def last_line(path):
    if not os.path.exists(path):
        return None
    lines = [l.strip() for l in open(path, errors="replace") if l.strip()]
    return lines[-1] if lines else None


def plot(run, groups):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    os.makedirs(os.path.join(run, "plots"), exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    made = []

    # 1. cost per solve of every variant, one panel per group
    ng = len(groups)
    nmax = max(len(items) for items in groups.values())
    fig, axes = plt.subplots(1, ng, figsize=(5.6 * ng + 1.2, 0.48 * nmax + 1.9), facecolor=SURFACE, squeeze=False)
    for ax, (g, items) in zip(axes[0], groups.items()):
        ax.set_facecolor(SURFACE)
        names = list(items)[::-1]
        vals = [items[v]["cost"] for v in names]
        cols = [KIND_COLOR[items[v]["kind"]] for v in names]
        y = range(len(names))
        ax.barh(y, vals, height=0.56, color=cols, edgecolor=SURFACE, linewidth=2)
        vmax = max(vals)
        for yi, v in zip(y, names):
            s = items[v]
            ok = s["converged"] == s["solves"]
            ax.text(s["cost"] + 0.015 * vmax, yi, "%s s  (%.0f it.)%s" % (fmt(s["cost"]), s["its_avg"], "" if ok else "  not conv."),
                    va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(list(y))
        ax.set_yticklabels([LABEL.get(v, v) for v in names])
        ax.set_xlim(0, vmax * 1.42)
        ax.set_title("%s: cost per solve [s]" % g, fontsize=10.5, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=KIND_COLOR[k]) for k in ("openfoam", "schwarz2lvl")]
    fig.legend(handles, ["OpenFOAM", "schwarz2lvl"], loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p = os.path.join(run, "plots", "gamg_cost.png")
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    made.append(p)

    # 2. time of every solve, largest group
    g = list(groups)[-1]
    items = groups[g]
    fig, ax = plt.subplots(figsize=(9.5, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (v, s) in enumerate(items.items()):
        ax.plot(range(1, len(s["series"]) + 1), s["series"], color=SERIES_COLOR.get(v, CAT[7]), linewidth=2,
                solid_capstyle="round", label=LABEL.get(v, v))
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale("log")
    ax.set_xlabel("solve (steps x correctors)")
    ax.set_ylabel("time per solve [s]  (schwarz2lvl: with setup / update)")
    ax.set_title("%s: time of every solve" % g, fontsize=10.5, color=INK, loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    p = os.path.join(run, "plots", "gamg_solves.png")
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    made.append(p)
    return made


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    run = sys.argv[1].rstrip("/")
    variants = read_variants(run)
    groups = load(run, variants)
    out = []
    w = out.append
    w("# GAMG reference: %s\n" % os.path.basename(run))
    for f in ("config.txt", "log.git"):
        p = os.path.join(run, f)
        if os.path.exists(p):
            w("`%s`\n" % " ".join(open(p).read().split()))
    w("## Checks\n")
    t = last_line(os.path.join(run, "log.test_bubbles"))
    if t:
        w("- test_bubbles: %s" % t)
    m = last_line(os.path.join(run, "log.check_matrix"))
    if m or os.path.exists(os.path.join(run, "log.check_app")):
        w("- matrix OpenFOAM vs generator (24^3, step 3): %s" % (m or "FAILED (see log.check_*)"))
    p = os.path.join(run, "cases.txt")
    if os.path.exists(p):
        for line in open(p):
            w("- mesh: `%s`" % line.strip())
    stp = os.path.join(run, "status.txt")
    if os.path.exists(stp):
        bad = [l.split() for l in open(stp) if len(l.split()) > 2 and l.split()[2] != "ok"]
        w("- runs not ok: " + (", ".join("%s (%s)" % (b[0], b[2]) for b in bad) if bad else "none"))
    allr = [s for items in groups.values() for s in items.values()]
    if allr:
        w("- solves: %d, converged %d, max true relative residual %.2e" % (
            sum(s["solves"] for s in allr), sum(s["converged"] for s in allr), max(s["max_res"] for s in allr)))
    w("")

    w("## Cost per solve\n")
    w("Every solve starts from 0. schwarz2lvl stops at ||b - Ax||_2 / ||b||_2 = 1e-8, OpenFOAM at its normalised "
      "L1 residual relTol (fvSolution: 1e-9, so that its 2-norm residual is also ~1e-8); *true res.* is "
      "||b - Ax||_2 / ||b||_2 for both. *cost/solve* = total / solves "
      "(schwarz2lvl: setup and updates included); *median* = median cost of the solves after the first; "
      "*first* = first solve (schwarz2lvl: with the initial setup); **vs %s** = cost/solve relative to GAMG "
      "with GaussSeidel (the usual OpenFOAM setup). Iterations: V-cycles (GAMG), CG / GMRES iterations.\n" % BASELINE)
    for g, items in groups.items():
        base = items.get(BASELINE)
        w("### %s\n" % g)
        w("| variant | solver | its avg / max | true res. (max) | first [s] | median [s] | cost/solve [s] | total [s] | vs %s |"
          % BASELINE)
        w("|:--|:--|--:|--:|--:|--:|--:|--:|--:|")
        for v, s in items.items():
            ratio = "%.2fx" % (s["cost"] / base["cost"]) if base and base["cost"] > 0 else "-"
            extra = ""
            if s["kind"] == "schwarz2lvl":
                extra = " (%d rebuilds, %d refact.)" % (s["rebuilds"], s["refact"])
            conv = "" if s["converged"] == s["solves"] else " **%d not conv.**" % (s["solves"] - s["converged"])
            w("| %s | %s%s | %.1f / %d%s | %.1e | %s | %s | %s | %s | %s |" % (
                v, LABEL.get(v, v), extra, s["its_avg"], s["its_max"], conv, s["max_res"], fmt(s["first"]),
                fmt(s["median"]), fmt(s["cost"]), fmt(s["total"], 1), ratio))
        of = {v: s for v, s in items.items() if s["kind"] == "openfoam"}
        us = {v: s for v, s in items.items() if s["kind"] == "schwarz2lvl"}
        if of and us:
            bo = min(of, key=lambda v: of[v]["cost"])
            bu = min(us, key=lambda v: us[v]["cost"])
            w("\nFastest OpenFOAM variant: **%s** (%s s per solve); best schwarz2lvl: **%s** (%s s per solve, "
              "%.2fx; steady-state median %.2fx)." % (
                  bo, fmt(of[bo]["cost"]), bu, fmt(us[bu]["cost"]), us[bu]["cost"] / of[bo]["cost"],
                  us[bu]["median"] / of[bo]["median"] if of[bo]["median"] > 0 else float("nan")))
        w("")
    made = plot(run, groups) if groups else []
    for p in made:
        w("![%s](plots/%s)\n" % (os.path.basename(p)[:-4], os.path.basename(p)))
    open(os.path.join(run, "report.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
