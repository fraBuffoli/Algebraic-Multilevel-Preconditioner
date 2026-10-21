#!/usr/bin/env python3
"""Summary of the setup-reuse campaign (Allrun_reuse.sh).

    python3 plot_reuse.py run_reuse_<jobid>

Reads <run>/csv/*.csv (one line per solve, written by --steps-csv) and writes
<run>/report.md and, if matplotlib is available, <run>/plots/<group>.png.

Runs are grouped by problem: the name without the last "_<mode>" suffix
(bub128_r1000_off, bub128_r1000_auto -> group bub128_r1000). Runs of one group may
have different numbers of steps (off runs fewer, its cost per solve is stationary),
so they are compared by the mean wall time per solve, initial setup included:
    cost per solve = (initial setup + updates + solves) / number of solves.
"""
import csv
import glob
import os
import sys
from collections import OrderedDict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # the report is written anyway
    plt = None

LEVELS = ("reuse", "refactorize", "rebuild")


def load(path):
    rows = list(csv.DictReader(open(path, newline="")))
    for r in rows:
        for k in ("t_update", "t_solve", "per_decade", "true_rel_res", "wasted"):
            r[k] = float(r.get(k) or 0.0)
        for k in ("step", "corrector", "iterations", "converged", "n0", "np"):
            r[k] = int(float(r.get(k) or 0))
    return rows


def summarize(rows):
    n = len(rows)
    first = rows[0]
    upd = rows[1:]
    s = OrderedDict()
    s["np"] = first["np"]
    s["solves"] = n
    s["converged"] = sum(r["converged"] for r in rows)
    for lv in LEVELS:
        s[lv] = sum(1 for r in upd if r["level"] == lv)
    s["t_init"] = first["t_update"]
    s["t_upd"] = sum(r["t_update"] for r in upd)
    s["t_ref"] = sum(r["t_update"] for r in upd if r["level"] == "refactorize")
    s["t_reb"] = sum(r["t_update"] for r in upd if r["level"] == "rebuild")
    s["t_solve"] = sum(r["t_solve"] for r in rows)
    s["t_total"] = s["t_init"] + s["t_upd"] + s["t_solve"]
    s["per_solve"] = s["t_total"] / n
    s["its_avg"] = sum(r["iterations"] for r in rows) / n
    s["its_max"] = max(r["iterations"] for r in rows)
    s["n0"] = first["n0"]
    s["rebuild_steps"] = [r["step"] for r in upd if r["level"] == "rebuild" and r["corrector"] == 0]
    return s


def mean(v):
    return sum(v) / len(v) if v else 0.0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run = sys.argv[1].rstrip("/")
    files = sorted(glob.glob(os.path.join(run, "csv", "*.csv")))
    if not files:
        sys.exit("no csv files in %s/csv" % run)
    data = OrderedDict()
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        rows = load(f)
        if rows:
            data[name] = rows
    groups = OrderedDict()
    for name in data:
        g, _, mode = name.rpartition("_")
        groups.setdefault(g or name, OrderedDict())[mode or name] = name
    order = {"off": 0, "refactor": 1, "auto": 2}
    for g in groups:  # off, refactor, auto, then the others
        groups[g] = OrderedDict(sorted(groups[g].items(), key=lambda kv: (order.get(kv[0], 3), kv[0])))

    out = []
    out.append("# Setup reuse (P1): %s\n" % os.path.basename(run))
    cfg = os.path.join(run, "config.txt")
    if os.path.exists(cfg):
        out.append("`%s`\n" % open(cfg).read().strip())
    out.append("Times are wall times between barriers, summed over the solves of each run. "
               "**cost/solve** = (initial setup + updates + solves) / solves; **speedup** is relative "
               "to the `off` run of the same group (full setup before every solve).\n")
    out.append("| run | np | solves | reuse / refactorize / rebuild | initial setup [s] | "
               "mean refactorize [s] | mean rebuild [s] | mean solve [s] | its avg / max | n0 | "
               "cost/solve [s] | speedup |")
    out.append("|:--|--:|--:|:--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for g, modes in groups.items():
        ref = None
        if "off" in modes:
            ref = summarize(data[modes["off"]])["per_solve"]
        for mode, name in modes.items():
            s = summarize(data[name])
            rows = data[name]
            mref = s["t_ref"] / s["refactorize"] if s["refactorize"] else 0.0
            mreb = s["t_reb"] / s["rebuild"] if s["rebuild"] else 0.0
            sp = ("%.1fx" % (ref / s["per_solve"])) if ref else "-"
            conv = "" if s["converged"] == s["solves"] else " (%d not conv.)" % (s["solves"] - s["converged"])
            out.append("| %s | %d | %d%s | %d / %d / %d | %.3f | %.3f | %.3f | %.3f | %.1f / %d | %d | %.3f | %s |" % (
                name, s["np"], s["solves"], conv, s["reuse"], s["refactorize"], s["rebuild"], s["t_init"], mref,
                mreb, s["t_solve"] / s["solves"], s["its_avg"], s["its_max"], s["n0"], s["per_solve"], sp))
    out.append("")

    out.append("## Rebuild steps and aging of the coarse basis\n")
    out.append("Iterations of the first corrector of every step (the matrix has just changed). "
               "`R` marks the steps where `auto` rebuilt the coarse basis.\n")
    for g, modes in groups.items():
        for mode, name in modes.items():
            rows = [r for r in data[name] if r["corrector"] == 0]
            seq = " ".join("%d%s" % (r["iterations"],
                                     "R" if (r["level"] == "rebuild" and r["step"] > 0 and mode != "off") else "")
                           for r in rows)
            out.append("- `%s`: %s" % (name, seq))
    out.append("")

    out.append("## Where the update time goes\n")
    out.append("Means over the solves of each level (from the per-solve CSV). For the split inside a "
               "refactorization see the `update.*` timers at the end of `log.<run>`: "
               "`update.subdomain` (new values, ghost exchange), `update.local_refactorization` "
               "(numerical LU of A_ii), `update.coarse_assembly` (A_00 = V^T A V with the stored W), "
               "`update.coarse_factorization` (numerical LU of A_00).\n")
    for g, modes in groups.items():
        for mode, name in modes.items():
            rows = data[name][1:]
            line = []
            for lv in LEVELS:
                t = [r["t_update"] for r in rows if r["level"] == lv]
                if t:
                    line.append("%s %.4f s (%d)" % (lv, mean(t), len(t)))
            out.append("- `%s`: %s" % (name, ", ".join(line) if line else "-"))
    out.append("")

    report = os.path.join(run, "report.md")
    open(report, "w").write("\n".join(out) + "\n")
    print("wrote", report)

    if plt is None:
        print("matplotlib not available: no plots")
        return
    os.makedirs(os.path.join(run, "plots"), exist_ok=True)
    for g, modes in groups.items():
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for mode, name in modes.items():
            rows = data[name]
            x = list(range(len(rows)))
            ax[0].plot(x, [r["iterations"] for r in rows], label=mode, lw=1.2)
            cum, acc = [], 0.0
            for r in rows:
                acc += r["t_update"] + r["t_solve"]
                cum.append(acc)
            ax[1].plot(x, cum, label=mode, lw=1.5)
            reb = [i for i, r in enumerate(rows) if r["level"] == "rebuild" and i > 0 and mode != "off"]
            if reb:
                ax[0].plot(reb, [rows[i]["iterations"] for i in reb], "o", ms=4, color=ax[0].lines[-1].get_color())
        ax[0].set_xlabel("solve")
        ax[0].set_ylabel("GMRES iterations")
        ax[0].set_title("%s: iterations (dots: rebuilds)" % g)
        ax[1].set_xlabel("solve")
        ax[1].set_ylabel("cumulative setup + solve time [s]")
        ax[1].set_title("%s: cumulative time" % g)
        for a in ax:
            a.grid(alpha=0.3)
            a.legend()
        fig.tight_layout()
        png = os.path.join(run, "plots", "%s.png" % g)
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print("wrote", png)


if __name__ == "__main__":
    main()
