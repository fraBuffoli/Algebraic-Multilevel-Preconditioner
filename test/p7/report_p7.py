#!/usr/bin/env python3
"""Summary of a P7 campaign: exact vs approximate local solvers (AMG; IC in the first campaign).

    python3 report_p7.py run_p7b_<jobid>        (Allrun_p7b.sh, Allrun_p7b_960.sh)
    python3 report_p7.py run_p7_<jobid>         (first campaign)

Reads <run>/csv/*.csv (one line per solve), <run>/log.* (configuration, coarse space, timers)
and <run>/variants.txt ("<variant> <options>" in the order of the campaign), and writes
<run>/report.md and, if matplotlib is available, <run>/plots/p7_cost.png (relative cost and
iterations of every variant) and <run>/plots/p7_iteration.png (where a GMRES iteration goes).

Run names: <group>_<variant>, e.g. bub192_sa. Costs are compared with the `ref` variant of the
same group (exact Cholesky factors, the P5 setting) and with `sp` (exact, single precision):
  * cost/solve = (initial setup + updates + solves) / solves, as measured;
  * cost/solve (median) = the same with the median time per iteration and the median time of
    each update level of the run: insensitive to the timing bursts seen on G100;
  * per-iteration split from the timers (average over the ranks) divided by the iterations;
  * "fresh" iterations = first solve after every (re)build of the coarse basis.
"""
import csv
import glob
import os
import re
import statistics as st
import sys
from collections import OrderedDict

TIMER_RE = re.compile(r"^  ([A-Za-z0-9_.~]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s*$")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


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
    """Local-solver part of the options: 'exact', 'exact, single', 'amg sa+gs, single', 'amg sa+gs, reuse'.

    Handles both campaigns: P7 (run_p7_*: --amg-relax given, IC variants, hierarchy reused by
    default) and P7b (run_p7b_*: Gauss-Seidel only, full setup by default, coarsest <= 500).
    """
    approx = re.search(r"--local-approx\s+(\S+)", opts)
    approx = approx.group(1) if approx else "none"
    single = bool(re.search(r"--local-precision\s+single", opts))
    if approx == "none":
        return "exact" + (", single" if single else "")
    if approx == "ic":
        return "ic" + (", single" if single else "")
    get = lambda k, d: (re.search(r"--amg-%s\s+(\S+)" % k, opts).group(1)
                        if re.search(r"--amg-%s\s+(\S+)" % k, opts) else d)
    legacy = "--amg-relax" in opts  # first P7 campaign: smoother given, reuse on by default
    out = ["amg %s+%s" % (get("coarsening", "sa"), get("relax", "gs"))]
    if get("sweeps", "1") != "1":
        out.append(get("sweeps", "1") + " sweeps")
    if get("cycles", "1") != "1":
        out.append(get("cycles", "1") + " cycles")
    if get("strength", "0") not in ("0", "0.0"):
        out.append("strength " + get("strength", "0"))
    if get("coarse-enough", "500") != "500":
        out.append("coarsest <= " + get("coarse-enough", "500"))
    if single:
        out.append("single")
    reuse = get("reuse", "1" if legacy else "0") not in ("0", "false", "no", "off")
    if legacy:
        out.append("reuse" if reuse else "no reuse")
    elif reuse:
        out.append("reuse")
    return ", ".join(out)


def parse_log(path):
    info = {"timers": {}, "m_avg": None, "m_max": None, "solver": "", "aii": "", "levels": None, "oc": None}
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
        if in_coarse:
            m = re.match(r"^m_i min/avg/max\s+(\d+)\s*/\s*([0-9.]+)\s*/\s*(\d+)", s)
            if m:
                info["m_avg"], info["m_max"] = float(m.group(2)), int(m.group(3))
            if s.startswith("coarse solver"):
                info["solver"] = s[len("coarse solver"):].strip()
            if s.startswith("A_ii factorization"):
                a = re.sub(r"\s*\(rank 0;.*$", "", s[len("A_ii factorization"):].strip())
                m = re.search(r"\[(\d+) levels, op\. complexity ([0-9.]+)\]", a)
                if m:
                    info["levels"], info["oc"] = int(m.group(1)), float(m.group(2))
                info["aii"] = re.sub(r"\s*\[.*\]", "", a)
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
    its = [int(float(r["iterations"])) for r in rows]
    tsolve = [float(r["t_solve"]) for r in rows]
    s = OrderedDict()
    s["np"] = int(float(first["np"]))
    s["solves"] = n
    s["converged"] = sum(int(float(r["converged"])) for r in rows)
    s["max_res"] = max(float(r["true_rel_res"]) for r in rows)
    s["n0_mean"] = st.mean(int(float(r["n0"])) for r in rows)
    s["refact"] = sum(1 for r in upd if r["level"] == "refactorize")
    s["rebuild"] = sum(1 for r in upd if r["level"] == "rebuild")
    s["its_total"] = sum(its)
    s["its_avg"] = s["its_total"] / n
    s["its_max"] = max(its)
    fresh = [its[0]] + [int(float(r["iterations"])) for r in upd if r["level"] == "rebuild"]
    s["its_fresh"] = st.mean(fresh)
    s["t_init"] = float(first["t_update"])
    by = {lv: [float(r["t_update"]) for r in upd if r["level"] == lv] for lv in ("reuse", "refactorize", "rebuild")}
    s["mean_ref"] = st.mean(by["refactorize"]) if by["refactorize"] else float("nan")
    s["mean_reb"] = st.mean(by["rebuild"]) if by["rebuild"] else float("nan")
    total = s["t_init"] + sum(float(r["t_update"]) for r in upd) + sum(tsolve)
    s["cost"] = total / n
    tpi = st.median(t / max(1, i) for t, i in zip(tsolve, its))
    robust = s["t_init"] + sum(len(v) * st.median(v) for v in by.values() if v) + s["its_total"] * tpi
    s["cost_med"] = robust / n
    s["solve_share"] = s["its_total"] * tpi / robust
    tm = info["timers"]
    g = lambda k: tm.get(k, float("nan"))
    itn = max(1, s["its_total"])
    s["it_local"] = 1e3 * g("solve.preconditioner.local.lu") / itn
    s["it_coarse"] = 1e3 * g("solve.preconditioner.coarse.solve") / itn
    s["it_vprod"] = 1e3 * (g("solve.preconditioner.coarse.restrict") + g("solve.preconditioner.coarse.prolong")) / itn
    s["it_total"] = 1e3 * g("solve") / itn
    s["it_other"] = s["it_total"] - s["it_local"] - s["it_coarse"] - s["it_vprod"]
    setups = 1 + s["rebuild"]
    s["loc_setup"] = g("setup.local_factorization") / setups
    s["loc_ref"] = g("update.local_refactorization") / s["refact"] if s["refact"] else float("nan")
    s["eig_setup"] = g("setup.eigensolver") / setups
    return s


def fmt(v, d=3):
    return "-" if v is None or v != v else ("%.*f" % (d, v))


def rel(v, ref):
    if ref is None or v != v or ref != ref or ref == 0:
        return "-"
    return "%+.0f%%" % (100.0 * (v / ref - 1.0))


def style(plt):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})


def finish(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_cost(groups, path, order, labels_of):
    """Dot plot per variant (categorical x): cost relative to ref and iterations, one color per problem."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    style(plt)
    variants = [v for v in order if any(v in items for items in groups.values())]
    labels = ["%s\n%s" % (v, labels_of(v).replace(", ", "\n")) for v in variants]
    fig, (ax_c, ax_i) = plt.subplots(2, 1, figsize=(max(9, 1.15 * len(variants) + 2) + 2.6, 8.2), facecolor=SURFACE)
    names = [g for g in groups if "ref" in groups[g]]
    width = 0.16
    for k, grp in enumerate(names):
        items, c = groups[grp], CAT[k % len(CAT)]
        off = (k - (len(names) - 1) / 2) * 2 * width
        xs = [i + off for i, v in enumerate(variants) if v in items]
        ref = items["ref"]["cost_med"]
        ax_c.plot(xs, [items[v]["cost_med"] / ref for v in variants if v in items], "o", color=c, markersize=9,
                  markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none", label=grp)
        ax_i.plot(xs, [items[v]["its_avg"] for v in variants if v in items], "o", color=c, markersize=9,
                  markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none", label=grp + ": average")
        ax_i.plot(xs, [items[v]["its_fresh"] for v in variants if v in items], "s", color=c, markersize=8,
                  markerfacecolor=SURFACE, markeredgecolor=c, markeredgewidth=2, linestyle="none",
                  label=grp + ": first solve after a rebuild")
    ax_c.axhline(1.0, color=INK2, linewidth=1, linestyle=":")
    ax_c.set_ylabel("cost per solve / cost of ref\n(median times)")
    ax_c.set_title("Cost per solve (lower is better; ref = exact Cholesky factors)", color=INK, fontsize=11)
    ax_i.set_ylabel("GMRES iterations per solve")
    ax_i.set_title("Iterations", color=INK, fontsize=11)
    if any(items[v]["its_avg"] > 4 * items["ref"]["its_avg"] for items in groups.values() if "ref" in items
           for v in items):
        from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter
        ax_i.set_yscale("log")
        ax_i.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax_i.yaxis.set_major_formatter(ScalarFormatter())
        ax_i.yaxis.set_minor_formatter(NullFormatter())
    for ax in (ax_c, ax_i):
        finish(ax)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(-0.6, len(variants) - 0.4)
        ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))  # outside: no overlap
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_iteration(groups, path, order, labels_of):
    """Stacked horizontal bars: time per GMRES iteration split in local / coarse / V products / rest."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    style(plt)
    rows = []
    for grp, items in groups.items():
        if rows:
            rows.append(None)
        for v in [x for x in order if x in items] + [x for x in items if x not in order]:
            s = items[v]
            rows.append(("%s, %s: %s" % (grp, v, labels_of(v)), s))
    keys = ["it_local", "it_coarse", "it_vprod", "it_other"]
    labels = ["local solves (A_ii)", "coarse solve (A_00)", "products with V", "rest (SpMV, orthogonalization)"]
    fig, ax = plt.subplots(figsize=(11, 0.36 * len(rows) + 1.6), facecolor=SURFACE)
    y, ticks, names, first = 0.0, [], [], True
    for r in rows:
        if r is None:
            y += 0.6
            continue
        lab, s = r
        left = 0.0
        for k, key in enumerate(keys):
            v = s[key] if s[key] == s[key] else 0.0
            ax.barh(y, v, left=left, height=0.72, color=CAT[k], edgecolor=SURFACE, linewidth=2,
                    label=labels[k] if first else None)
            left += v
        first = False
        ax.annotate("%.1f ms" % s["it_total"], (left, y), textcoords="offset points", xytext=(5, 0), va="center",
                    fontsize=8.5, color=INK2)
        ticks.append(y)
        names.append(lab)
        y += 1
    ax.set_yticks(ticks)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    finish(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("time per GMRES iteration [ms] (timers averaged over the ranks)")
    ax.set_title("Where a GMRES iteration goes", color=INK, fontsize=11, pad=28)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.1)
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run = sys.argv[1].rstrip("/")
    options = read_variants(run)
    order = list(options.keys())
    labels_of = lambda v: short_options(options.get(v, "")) if v in options else v
    key = lambda k: order.index(k) if k in order else len(order)
    groups, infos = OrderedDict(), {}
    for f in sorted(glob.glob(os.path.join(run, "csv", "*.csv"))):
        name = os.path.splitext(os.path.basename(f))[0]
        rows = list(csv.DictReader(open(f, newline="")))
        if not rows:
            continue
        info = parse_log(os.path.join(run, "log." + name))
        grp, _, var = name.partition("_")
        groups.setdefault(grp, OrderedDict())[var] = summarize(rows, info)
        infos[name] = info
    for grp in groups:
        groups[grp] = OrderedDict(sorted(groups[grp].items(), key=lambda kv: key(kv[0])))

    out = []
    w = out.append
    w("# P7 - inexact local solvers: %s\n" % os.path.basename(run))
    for f in ("config.txt", "log.git"):
        p = os.path.join(run, f)
        if os.path.exists(p):
            w("`" + " ".join(open(p).read().split("\n")).strip() + "`\n")
    for t in ("test_inexact", "test_reuse", "test_factorization"):
        p = os.path.join(run, "log." + t)
        if os.path.exists(p):
            lines = [l.strip() for l in open(p, errors="replace") if l.strip()]
            fails = [l for l in lines if l.startswith("[FAIL]")]
            w("%s: %s%s\n" % (t, lines[-1] if lines else "?", (" - " + "; ".join(fails)) if fails else ""))
    stp = os.path.join(run, "status.txt")
    if os.path.exists(stp):
        bad = [l.split() for l in open(stp) if l.split() and l.split()[2] != "ok"]
        w("Runs not ok: " + (", ".join("%s (%s)" % (b[0], b[2]) for b in bad) if bad else "none") + "\n")
    allr = [s for items in groups.values() for s in items.values()]
    w("All solves: %d, converged %d, max true residual %.2e\n" % (
        sum(s["solves"] for s in allr), sum(s["converged"] for s in allr), max([s["max_res"] for s in allr] or [0])))

    w("## Cost per solve (--reuse auto)\n")
    w("*A_ii solver* as reported by rank 0 (AMG: levels and operator complexity = nonzeros of all levels / "
      "nnz(A_ii)). *fresh its* = first solve after every (re)build. *refact.* = mean time of a refactorize "
      "step (A_ii: numerical refactorization or AMG rebuild; A_00 recomputed), *rebuild* = mean time of a "
      "full setup (eigensolver included). **vs ref** / **vs sp** compare the median-time cost with exact "
      "factors in double / single precision.\n")
    w("| run | A_ii solver | AMG levels / op. compl. | its avg / max | fresh its | refact. / rebuilds | refact. [s] "
      "| rebuild [s] | cost/solve [s] | cost/solve, median [s] | vs ref | vs sp |")
    w("|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for grp, items in groups.items():
        ref, sp = items.get("ref"), items.get("sp")
        for v, s in items.items():
            info = infos["%s_%s" % (grp, v)]
            amg = "%d / %.2f" % (info["levels"], info["oc"]) if info["levels"] else "-"
            w("| %s_%s | %s | %s | %.1f / %d | %.1f | %d / %d | %s | %s | %s | %s | %s | %s |" % (
                grp, v, info["aii"] or labels_of(v), amg, s["its_avg"], s["its_max"], s["its_fresh"], s["refact"],
                s["rebuild"], fmt(s["mean_ref"]), fmt(s["mean_reb"]), fmt(s["cost"]), fmt(s["cost_med"]),
                "ref" if v == "ref" else rel(s["cost_med"], ref["cost_med"] if ref else None),
                "sp" if v == "sp" else rel(s["cost_med"], sp["cost_med"] if sp else None)))
    w("")

    w("## Where an iteration and a setup go\n")
    w("Per GMRES iteration [ms], timers averaged over the ranks: local solves (A_ii factors or approximate "
      "inverse), coarse solve (A_00^{-1}, waiting included), products with V (restrict + prolong), the rest "
      "(SpMV, orthogonalization, halo). Per operation [s]: A_ii factorization / AMG setup per (re)build, A_ii "
      "refactorization / AMG rebuild per refactorize step, eigensolver per (re)build (A~_ii is always "
      "factorized exactly). *solves* = share of the median-time cost spent in GMRES.\n")
    w("| run | local [ms] | coarse [ms] | V products [ms] | rest [ms] | total [ms] | A_ii setup [s] "
      "| A_ii refact. [s] | eigensolver [s] | solves |")
    w("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for grp, items in groups.items():
        for v, s in items.items():
            w("| %s_%s | %s | %s | %s | %s | %s | %s | %s | %s | %.0f%% |" % (
                grp, v, fmt(s["it_local"], 2), fmt(s["it_coarse"], 2), fmt(s["it_vprod"], 2), fmt(s["it_other"], 2),
                fmt(s["it_total"], 2), fmt(s["loc_setup"], 3), fmt(s["loc_ref"], 3), fmt(s["eig_setup"], 2),
                100 * s["solve_share"]))
    w("")
    if plot_cost(groups, os.path.join(run, "plots", "p7_cost.png"), order, labels_of):
        w("![cost and iterations](plots/p7_cost.png)\n")
    if plot_iteration(groups, os.path.join(run, "plots", "p7_iteration.png"), order, labels_of):
        w("![where an iteration goes](plots/p7_iteration.png)\n")

    w("Iterations of the first corrector of every step (`R` = rebuild of the coarse basis):\n")
    for grp, items in groups.items():
        for v in items:
            rows = list(csv.DictReader(open(os.path.join(run, "csv", "%s_%s.csv" % (grp, v)), newline="")))
            seq = ["%s%s" % (r["iterations"], "R" if r["level"] == "rebuild" and r["step"] != "0" else "")
                   for r in rows if r["corrector"] == "0"]
            w("- `%s_%s` (%s): %s" % (grp, v, labels_of(v), " ".join(seq)))
    w("")

    path = os.path.join(run, "report.md")
    open(path, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("\nwritten: %s" % path)


if __name__ == "__main__":
    main()
