#!/usr/bin/env python3
"""Summary of the P5 campaigns (Allrun_p5.sh, Allrun_p5b.sh): size of the coarse space,
coarse solver, precision of the local factors, tolerance of the eigensolver.

    python3 report_p5.py run_p5_<jobid>
    python3 report_p5.py run_p5b_<jobid>

Reads <run>/csv/*.csv (one line per solve) and <run>/log.* (configuration, coarse space,
timers) and writes <run>/report.md and, if matplotlib is available, <run>/plots/p5_sweep.png
(relative cost and iterations of every variant).

Run names: <group>_<variant>, e.g. bub192_n16 (sweep) or bub192_cs_cp5 (coarse solver). If the
run directory has a variants.txt (Allrun_p5b.sh), its lines "<variant> <options>" give the order and
the options of the variants.
Costs are compared with the `ref` variant of the same group (--tau 0.3 --nev 60, as P1/P2):
  * cost/solve = (initial setup + updates + solves) / solves, as measured;
  * cost/solve (median) = the same with the median time per iteration and the median time of
    each update level of the run: insensitive to the timing bursts seen on G100 (P1 RESULTS.md);
  * per-iteration split from the timers (average over the ranks) divided by the iterations;
  * "fresh" iterations = first solve after every (re)build of the coarse basis: the quality of
    the coarse space without the aging of P1.
"""
import csv
import glob
import os
import re
import statistics as st
import sys
from collections import OrderedDict

TIMER_RE = re.compile(r"^  ([A-Za-z0-9_.~]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s*$")
SWEEP_ORDER = ["ref", "t010", "t003", "n32", "n16", "n8", "n4"]


def read_variants(run):
    """variants.txt: '<variant> <options...>' per line (order of the campaign), or empty."""
    v = OrderedDict()
    p = os.path.join(run, "variants.txt")
    if os.path.exists(p):
        for line in open(p):
            parts = line.split(None, 1)
            if parts:
                v[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
    return v


def short_options(opts):
    """'--tau 0.3 --nev 16 --local-precision single --eig-tol 1e-4' -> 'nev 16, single, tol 1e-4'."""
    out = []
    m = re.search(r"--tau\s+(\S+)", opts)
    if m and m.group(1) != "0.3":
        out.append("tau " + m.group(1))
    m = re.search(r"--nev\s+(\S+)", opts)
    if m:
        out.append("nev " + m.group(1))
    if re.search(r"--local-precision\s+single", opts):
        out.append("single")
    m = re.search(r"--eig-tol\s+(\S+)", opts)
    if m:
        out.append("tol " + m.group(1))
    return ", ".join(out)


def parse_log(path):
    info = {"timers": {}, "tau": None, "nev": None, "m_min": None, "m_avg": None, "m_max": None,
            "solver": "", "n0_last": None, "single": False}
    if not os.path.exists(path):
        return info
    in_timers = in_coarse = False
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        if line.startswith("Timings [s] over"):
            in_timers, in_coarse = True, False
            continue
        if in_timers:
            m = TIMER_RE.match(line)
            if m:
                info["timers"][m.group(1)] = float(m.group(3))
            continue
        if line.startswith("Coarse space"):
            in_coarse = True
            continue
        s = line.strip()
        m = re.match(r"^tau\s+([0-9.eE+-]+)$", s)
        if m and info["tau"] is None:
            info["tau"] = float(m.group(1))
        m = re.match(r"^nev\s+(\d+)$", s)
        if m and info["nev"] is None:
            info["nev"] = int(m.group(1))
        if s.startswith("local solver") and "single precision" in s:
            info["single"] = True
        if in_coarse:
            m = re.match(r"^m_i min/avg/max\s+(\d+)\s*/\s*([0-9.]+)\s*/\s*(\d+)", s)
            if m:
                info["m_min"], info["m_avg"], info["m_max"] = int(m.group(1)), float(m.group(2)), int(m.group(3))
            m = re.match(r"^n0\s+(\d+)$", s)
            if m:
                info["n0_last"] = int(m.group(1))
            if s.startswith("coarse solver"):
                info["solver"] = s[len("coarse solver"):].strip()
    return info


def short_solver(s):
    if s.startswith("root("):
        return "root"
    m = re.match(r"cpardiso\((\d+) procs, mtype (-?\d+)", s)
    if m:
        return "cpardiso %s (mtype %s)" % (m.group(1), m.group(2))
    return s or "-"


def summarize(rows, info):
    first, upd = rows[0], rows[1:]
    n = len(rows)
    np_ = int(float(first["np"]))
    its = [int(float(r["iterations"])) for r in rows]
    tsolve = [float(r["t_solve"]) for r in rows]
    s = OrderedDict()
    s["np"] = np_
    s["solves"] = n
    s["converged"] = sum(int(float(r["converged"])) for r in rows)
    s["max_res"] = max(float(r["true_rel_res"]) for r in rows)
    s["n0_mean"] = st.mean(int(float(r["n0"])) for r in rows)
    s["m_avg"] = s["n0_mean"] / np_
    s["refact"] = sum(1 for r in upd if r["level"] == "refactorize")
    s["rebuild"] = sum(1 for r in upd if r["level"] == "rebuild")
    s["its_total"] = sum(its)
    s["its_avg"] = s["its_total"] / n
    s["its_max"] = max(its)
    fresh = [int(float(rows[0]["iterations"]))] + [int(float(r["iterations"])) for r in upd if r["level"] == "rebuild"]
    s["its_fresh"] = st.mean(fresh)
    s["t_init"] = float(first["t_update"])
    by = {lv: [float(r["t_update"]) for r in upd if r["level"] == lv] for lv in ("reuse", "refactorize", "rebuild")}
    s["mean_ref"] = st.mean(by["refactorize"]) if by["refactorize"] else 0.0
    s["mean_reb"] = st.mean(by["rebuild"]) if by["rebuild"] else 0.0
    total = s["t_init"] + sum(float(r["t_update"]) for r in upd) + sum(tsolve)
    s["cost"] = total / n
    tpi = st.median(t / max(1, i) for t, i in zip(tsolve, its))
    robust = s["t_init"] + sum(len(v) * st.median(v) for v in by.values() if v) + s["its_total"] * tpi
    s["cost_med"] = robust / n
    s["ms_it_med"] = 1e3 * tpi
    tm = info["timers"]
    g = lambda k: tm.get(k, float("nan"))
    itn = max(1, s["its_total"])
    s["it_local"] = 1e3 * g("solve.preconditioner.local.lu") / itn
    s["it_coarse"] = 1e3 * g("solve.preconditioner.coarse.solve") / itn
    s["it_vprod"] = 1e3 * (g("solve.preconditioner.coarse.restrict") + g("solve.preconditioner.coarse.prolong")) / itn
    s["it_total"] = 1e3 * g("solve") / itn
    s["it_other"] = s["it_total"] - s["it_local"] - s["it_coarse"] - s["it_vprod"]
    setups = 1 + s["rebuild"]
    refs = max(1, s["refact"])
    s["eig_setup"] = g("setup.eigensolver") / setups
    s["cf_setup"] = g("setup.coarse_factorization") / setups
    s["cf_ref"] = g("update.coarse_factorization") / refs if s["refact"] else float("nan")
    s["ca_ref"] = g("update.coarse_assembly") / refs if s["refact"] else float("nan")
    return s


def fmt(v, d=3):
    return "-" if v is None or v != v else ("%.*f" % (d, v))


def rel(v, ref):
    if ref is None or v != v or ref != ref or ref == 0:
        return "-"
    return "%+.0f%%" % (100.0 * (v / ref - 1.0))


def plot(sweeps, path, order, labels_of):
    """Dot plot per variant (categorical x): relative cost and iterations, one color per problem."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    # Reference palette (dataviz skill), light surface: categorical slots 1-2, text tokens.
    colors = ["#2a78d6", "#eb6834"]
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "text.color": ink})
    variants = [v for v in order if any(v in items for items in sweeps.values())]
    labels = ["%s\n%s" % (v, labels_of(v).replace(", ", "\n")) for v in variants]
    fig, (ax_c, ax_i) = plt.subplots(1, 2, figsize=(12 if len(variants) <= 6 else 14, 4.6), facecolor=surface)
    groups = [g for g in sweeps if "ref" in sweeps[g]]
    width = 0.18
    for k, grp in enumerate(groups):
        items, c = sweeps[grp], colors[k % len(colors)]
        off = (k - (len(groups) - 1) / 2) * 2 * width
        xs = [i + off for i, v in enumerate(variants) if v in items]
        ref = items["ref"]["cost_med"]
        ax_c.plot(xs, [items[v]["cost_med"] / ref for v in variants if v in items], "o", color=c, markersize=9,
                  markeredgecolor=surface, markeredgewidth=2, linestyle="none", label=grp)
        ax_i.plot(xs, [items[v]["its_avg"] for v in variants if v in items], "o", color=c, markersize=9,
                  markeredgecolor=surface, markeredgewidth=2, linestyle="none", label=grp + ": average")
        ax_i.plot(xs, [items[v]["its_fresh"] for v in variants if v in items], "s", color=c, markersize=8,
                  markerfacecolor=surface, markeredgecolor=c, markeredgewidth=2, linestyle="none",
                  label=grp + ": first solve after a rebuild")
    ax_c.axhline(1.0, color=ink2, linewidth=1, linestyle=":")
    ax_c.set_ylabel("cost per solve / cost of ref (median times)")
    ax_c.set_title("Cost per solve (lower is better)", color=ink, fontsize=11)
    ax_i.set_ylabel("GMRES iterations per solve")
    ax_i.set_title("Iterations", color=ink, fontsize=11)
    for ax in (ax_c, ax_i):
        ax.set_facecolor(surface)
        ax.grid(True, axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(labels, fontsize=8 if len(variants) > 6 else 8.5)
        ax.set_xlim(-0.6, len(variants) - 0.4)
        ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run = sys.argv[1].rstrip("/")
    runs = OrderedDict()
    for f in sorted(glob.glob(os.path.join(run, "csv", "*.csv"))):
        name = os.path.splitext(os.path.basename(f))[0]
        rows = list(csv.DictReader(open(f, newline="")))
        if rows:
            info = parse_log(os.path.join(run, "log." + name))
            runs[name] = (summarize(rows, info), info)
    variants_txt = read_variants(run)
    legacy = {"ref": "--tau 0.3 --nev 60", "t010": "--tau 0.1 --nev 60", "t003": "--tau 0.03 --nev 60",
              "n32": "--tau 0.3 --nev 32", "n16": "--tau 0.3 --nev 16", "n8": "--tau 0.3 --nev 8",
              "n4": "--tau 0.3 --nev 4"}
    options = variants_txt if variants_txt else legacy
    order = list(variants_txt.keys()) if variants_txt else SWEEP_ORDER
    labels_of = lambda v: short_options(options.get(v, "")) or v
    key = lambda k: order.index(k) if k in order else len(order)
    sweeps, coarse = OrderedDict(), OrderedDict()
    for name, (s, info) in runs.items():
        grp, _, var = name.partition("_")
        if var.startswith("cs_"):
            coarse.setdefault(grp, OrderedDict())[var] = (s, info)
        else:
            sweeps.setdefault(grp, OrderedDict())[var] = (s, info)

    out = []
    w = out.append
    w("# P5 - size of the coarse space and coarse solver: %s\n" % os.path.basename(run))
    for f in ("config.txt", "log.git"):
        p = os.path.join(run, f)
        if os.path.exists(p):
            w("`" + " ".join(open(p).read().split("\n")).strip() + "`\n")
    stp = os.path.join(run, "status.txt")
    if os.path.exists(stp):
        bad = [l.split() for l in open(stp) if l.split() and l.split()[2] != "ok"]
        w("Runs not ok: " + (", ".join("%s (%s)" % (b[0], b[2]) for b in bad) if bad else "none") + "\n")
    w("All solves: %d, converged %d, max true residual %.2e\n" % (
        sum(s["solves"] for s, _ in runs.values()), sum(s["converged"] for s, _ in runs.values()),
        max([s["max_res"] for s, _ in runs.values()] or [0.0])))

    if sweeps:
        w("## Sweep: coarse vectors per subdomain (--reuse auto)\n")
        w("m_i avg = mean n0 over the solves / processes; m_i max = last coarse space. *fresh its* = first "
          "solve after every (re)build. **vs ref** compares the median-time cost.\n")
        w("| run | options | m_i avg / max | n0 | coarse solver | its avg / max | fresh its | refact. / rebuilds "
          "| mean refact. [s] | mean rebuild [s] | cost/solve [s] | cost/solve, median [s] | vs ref |")
        w("|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
        for grp, items in sweeps.items():
            ref = items.get("ref", (None, None))[0]
            for v in sorted(items, key=key):
                s, info = items[v]
                opt = "tau %s, nev %s%s%s" % (info["tau"], info["nev"], ", single" if info["single"] else "",
                                             (", tol " + re.search(r"--eig-tol\s+(\S+)", options.get(v, "")).group(1))
                                             if re.search(r"--eig-tol\s+(\S+)", options.get(v, "")) else "")
                w("| %s_%s | %s | %.1f / %s | %.0f | %s | %.1f / %d | %.1f | %d / %d | %s | %s | %s | %s | %s |" % (
                    grp, v, opt, s["m_avg"], info["m_max"], s["n0_mean"],
                    short_solver(info["solver"]), s["its_avg"], s["its_max"], s["its_fresh"], s["refact"],
                    s["rebuild"], fmt(s["mean_ref"]), fmt(s["mean_reb"]), fmt(s["cost"]), fmt(s["cost_med"]),
                    "ref" if v == "ref" else rel(s["cost_med"], ref["cost_med"] if ref else None)))
        w("")
        w("### Where an iteration and an update go\n")
        w("Per GMRES iteration [ms], timers averaged over the ranks: solves with the factors of A_ii, coarse "
          "solve (A_00^{-1}, waiting included), products with V (restrict + prolong), the rest (SpMV, "
          "orthogonalization). Per operation [s]: eigensolver and A_00 factorization per setup, A_00 "
          "recomputation and refactorization per refactorize step.\n")
        w("| run | local [ms] | coarse [ms] | V products [ms] | rest [ms] | total [ms] | eigensolver [s] "
          "| A_00 fact. setup [s] | A_00 = V^T A V refact. [s] | A_00 refact. [s] |")
        w("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for grp, items in sweeps.items():
            for v in sorted(items, key=key):
                s, _ = items[v]
                w("| %s_%s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    grp, v, fmt(s["it_local"], 2), fmt(s["it_coarse"], 2), fmt(s["it_vprod"], 2),
                    fmt(s["it_other"], 2), fmt(s["it_total"], 2), fmt(s["eig_setup"], 2), fmt(s["cf_setup"], 3),
                    fmt(s["ca_ref"], 3), fmt(s["cf_ref"], 3)))
        w("")
        if plot(OrderedDict((g, OrderedDict((v, s) for v, (s, _) in it.items())) for g, it in sweeps.items()),
                os.path.join(run, "plots", "p5_sweep.png"), order, labels_of):
            w("![sweep](plots/p5_sweep.png)\n")
        w("Iterations of the first corrector of every step (`R` = rebuild of the coarse basis):\n")
        for grp, items in sweeps.items():
            for v in sorted(items, key=key):
                rows = list(csv.DictReader(open(os.path.join(run, "csv", "%s_%s.csv" % (grp, v)), newline="")))
                seq = ["%s%s" % (r["iterations"], "R" if r["level"] == "rebuild" and r["step"] != "0" else "")
                       for r in rows if r["corrector"] == "0"]
                w("- `%s_%s`: %s" % (grp, v, " ".join(seq)))
        w("")

    if coarse:
        w("## Coarse solver at the ref setting (--reuse refactor)\n")
        w("Same matrices and coarse space in every run: only where A_00 is factorized and solved changes. "
          "cpardiso K = K leaders, each owning the rows of N/K consecutive ranks (K = nodes: one per node).\n")
        w("| run | coarse solver | n0 | A_00 fact. setup [s] | A_00 refact. [s] | coarse solve [ms/it] "
          "| iteration [ms] | cost/solve, median [s] |")
        w("|:--|:--|--:|--:|--:|--:|--:|--:|")
        for grp, items in coarse.items():
            for v in sorted(items, key=lambda k: (k != "cs_root", int(re.sub(r"\D", "", k) or 0))):
                s, info = items[v]
                w("| %s_%s | %s | %.0f | %s | %s | %s | %s | %s |" % (
                    grp, v, short_solver(info["solver"]), s["n0_mean"], fmt(s["cf_setup"]), fmt(s["cf_ref"]),
                    fmt(s["it_coarse"], 2), fmt(s["it_total"], 2), fmt(s["cost_med"])))
        w("")

    path = os.path.join(run, "report.md")
    open(path, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("\nwritten: %s" % path)


if __name__ == "__main__":
    main()
