#!/usr/bin/env python3
"""
plot_coarse.py - tables and plots of the runs of Allrun_coarse.sh: distributed factorization
of A_00 (MKL cluster_sparse_solver, "cpK" = K processes) vs rank 0 ("root"), and Boolean vs
multiplicity partition of unity, compared with the paper (Al Daas, Jolivet, Rees, SISC 2023).

    python3 plot_coarse.py run_coarse_<jobid>

Writes into the run directory:
    report.md      coarse solvers, partitions of unity vs paper, Fig. 2, Table 2, Fig. 4, phases
    summary.csv    one row per run
    phases.csv     every timer of every run (min / avg / max over the ranks)
    plots/*.png    only if matplotlib is available (pip install --user matplotlib)

Python >= 3.6, standard library only (matplotlib optional). Needs plot_paper.py next to it.
"""
import os
import re
import shlex
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_paper as pp                                     # noqa: E402
from plot_paper import T, ft, table, its                    # noqa: E402

POU = OrderedDict([("bool", "Boolean"), ("mult", "multiplicity")])
SUB = [("exchange", "setup.coarse_factorization.exchange"), ("analysis", "setup.coarse_factorization.analysis"),
       ("numeric", "setup.coarse_factorization.numeric")]


# ----------------------------------------------------------------------------- reading
def coarse_info(r):
    """Coarse solver line of the log: cpardiso(K procs, mtype M, nnz(LU) X, perturbed pivots P) or root(...)."""
    path = os.path.join(pp.RUN, "log." + r["name"])
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = re.findall(r"^\s+coarse solver\s{2,}(.+?)\s*$", text, re.M)
    m = re.search(r"Arnoldi nev avg/max\s+([0-9.]+) / ([0-9]+)", text)
    r["k"] = (float(m.group(1)), int(m.group(2))) if m else None
    m = re.search(r"max operator applications\s+(\d+)", text)
    r["ops"] = int(m.group(1)) if m else None
    s = lines[-1] if lines else ""
    r["cs_line"] = s
    for key, rx in (("cs_procs", r"(\d+) procs"), ("cs_mtype", r"mtype (-?\d+)"), ("cs_nnz", r"nnz\(LU\) (\d+)"),
                    ("cs_pert", r"perturbed pivots (\d+)")):
        m = re.search(rx, s)
        r[key] = int(m.group(1)) if m else None


def load():
    runs = pp.load()
    for r in runs.values():
        coarse_info(r)
    cfg = OrderedDict()
    p = os.path.join(pp.RUN, "config.txt")
    if os.path.isfile(p):
        with open(p) as f:
            for kv in shlex.split(f.read()):
                k, _, v = kv.partition("=")
                cfg[k] = v
    return runs, cfg


def tab1(runs):
    """Two-level runs {(matrix, pou, tau): {tag: run}} from tab1_<matrix>_<pou>_t<tau>_<tag>, and the
    one-level runs {(matrix, pou): run} from tab1_<matrix>_<pou>_RAS."""
    d, ras = OrderedDict(), {}
    for r in runs.values():
        m = re.match(r"tab1_(.+)_(bool|mult)_t([0-9.]+)_(root|cp\d+)$", r["name"])
        if m:
            d.setdefault((m.group(1), m.group(2), float(m.group(3))), OrderedDict())[m.group(4)] = r
            continue
        m = re.match(r"tab1_(.+)_(bool|mult)_RAS$", r["name"])
        if m:
            ras[(m.group(1), m.group(2))] = r
    return d, ras


def tags_of(d):
    tags = {t for v in d.values() for t in v}
    return sorted(tags, key=lambda t: (t != "root", -int(t[2:]) if t.startswith("cp") else 0))


def cf(r):
    """Time of the coarse factorization (max over ranks) of a finished run."""
    return T(r, ["setup.coarse_factorization"]) if r and r["timers"] else None


def ok(r):
    return r is not None and r["status"] in ("ok", "not_converged") and bool(r["timers"])


def two_level(v):
    """A finished run of the group (n0 and iterations do not depend on the coarse solver)."""
    return next((r for r in v.values() if ok(r)), None) or next(iter(v.values()), None)


def lab(key):
    return "%s, tau %s" % (POU[key[1]], key[2])


# ----------------------------------------------------------------------------- sections
def sec_coarse(runs, cfg):
    d, _ = tab1(runs)
    d = OrderedDict((k, v) for k, v in d.items() if len(v) > 1)     # groups with several coarse solvers
    if not d:
        return ""
    tags = tags_of(d)
    cps = [t for t in tags if t != "root"]
    out = ["## A_00 factorization: rank 0 vs MKL cluster_sparse_solver (Table 1 matrices, nev = 300)\n\n",
           "Time of `setup.coarse_factorization` (max over ranks, it includes sending A_00 to the processes that "
           "factorize it). root = gather on rank 0 + sequential PARDISO LU; cpK = MKL cluster_sparse_solver on K "
           "processes (the first rank of every group of %s/K consecutive ranks), LDL^T (mtype -2) for symmetric "
           "matrices, LU otherwise. Iterations must not depend on the coarse solver (exact solvers): `!=` flags "
           "a difference. Speedup = root / best cpK.\n\n" % cfg.get("NP", "N")]
    rows = []
    for key, v in d.items():
        ref = v.get("root") or two_level(v)
        itset = {r.get("its") for r in v.values() if ok(r)}
        it = its(ref) + ("" if len(itset) <= 1 else " != " + "/".join(str(i) for i in sorted(itset)))
        times = OrderedDict((t, cf(v.get(t)) if ok(v.get(t)) else None) for t in tags)
        best = min(((times[t], t) for t in cps if times.get(t) is not None), default=(None, None))
        sp = "%.1f (%s)" % (times["root"] / best[0], best[1]) if times.get("root") and best[0] else ""
        cells = [ft(times[t]) if times[t] is not None else ("-" if ok(v.get(t)) else its(v.get(t))) for t in tags]
        solve = [ft(v[t].get("t_solve")) if ok(v.get(t)) else "" for t in ("root", best[1]) if t]
        rows.append([key[0], POU[key[1]], key[2], ref.get("n0", "")] + [it] + cells + [sp] +
                    [ft(v["root"].get("t_setup")) if ok(v.get("root")) else "",
                     ft(v[best[1]].get("t_setup")) if best[1] else ""] + (solve + ["", ""])[:2])
    out.append(table(["matrix", "PoU", "tau", "n0", "its"] + ["%s [s]" % t for t in tags] +
                     ["speedup", "setup root [s]", "setup best cp [s]", "GMRES root [s]", "GMRES best cp [s]"], rows))

    rows = []
    for key, v in d.items():
        for t in cps:
            r = v.get(t)
            if ok(r):
                rows.append([key[0], POU[key[1]], key[2], t, r.get("n0", "")] + [ft(T(r, [k])) for _, k in SUB] +
                            [ft(cf(r)), "%.2e" % r["cs_nnz"] if r.get("cs_nnz") else "", r.get("cs_pert", "")])
    if rows:
        out.append("cluster_sparse_solver phases (max over ranks): exchange = coefficients sent to the owners of "
                   "their rows, analysis = METIS ordering + symbolic factorization (sequential, on the first "
                   "process), numeric = distributed factorization. The GMRES columns above show the cost of the "
                   "coarse solve, once per iteration.\n\n")
        out.append(table(["matrix", "PoU", "tau", "solver", "n0", "exchange", "analysis", "numeric", "total",
                          "nnz(LU)", "perturbed"], rows))
    out.append(plot_coarse(d, tags))
    out.append(plot_tab1_phases(d, tags, cfg))
    return "".join(out)


def plot_coarse(d, tags):
    if not pp.HAVE_MPL:
        return ""
    groups = sorted({(k[1], k[2]) for k in d})
    f, axes = pp.fig(len(groups), w=6.6, h=4.2)
    for ax, (pou, tau) in zip(axes, groups):
        keys = sorted([k for k in d if (k[1], k[2]) == (pou, tau)], key=lambda k: two_level(d[k]).get("n0") or 0)
        w = 0.8 / max(1, len(tags))
        for k, t in enumerate(tags):
            xs, ys = [], []
            for i, key in enumerate(keys):
                r = d[key].get(t)
                if ok(r) and cf(r):
                    xs.append(i - 0.4 + w * (k + 0.5))
                    ys.append(cf(r))
            ax.bar(xs, ys, width=w, color=pp.COLORS[1] if t == "root" else pp.SEQ[(2 * k) % len(pp.SEQ)],
                   label=t, edgecolor=pp.SURF, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(["%s\nn0=%s" % (key[0], two_level(d[key]).get("n0", "")) for key in keys],
                           rotation=45, ha="right", fontsize=7)
        ax.xaxis.grid(False)
        pp.finish(ax, "%s PoU, tau = %s" % (POU[pou], tau), "", "A_00 factorization [s]")
    return pp.save(f, "coarse_factorization.png")


def plot_tab1_phases(d, tags, cfg):
    """Stacked phases of every matrix: root vs the cpK of the other experiments."""
    cp = cfg.get("CP") if cfg.get("CP") in tags else next((t for t in tags if t != "root"), None)
    out = ""
    for pou, tau in sorted({(k[1], k[2]) for k in d}):
        rr, labels = [], []
        for key, v in d.items():
            if (key[1], key[2]) != (pou, tau):
                continue
            for t in ("root", cp):
                if ok(v.get(t)):
                    rr.append(v[t])
                    labels.append("%s %s" % (key[0], t))
        out += pp.stacked(rr, labels, "tab1_phases_%s_t%s.png" % (pou, tau),
                          "Table 1, %s PoU, tau = %s: time per phase" % (POU[pou], tau))
    return out


def sec_pou(runs):
    d, ras = tab1(runs)
    if not d:
        return ""
    keys = sorted({(k[1], k[2]) for k in d}, key=lambda k: (k[0] != "bool", k[1]))     # bool 0.6, bool 0.9, mult 0.6
    rows, N = [], None
    for m, p in pp.PAPER_T1.items():
        rr = [two_level(d[(m,) + k]) if (m,) + k in d else None for k in keys]
        if not any(rr) and (m, "bool") not in ras:
            continue
        N = N or next((r["np"] for r in rr if r and r.get("np")), None)
        n0n = ["%.0f" % (r["n0"] / r["np"]) if r and r.get("n0") is not None and r.get("np") else "" for r in rr]
        rows.append([m] + n0n + ["%.0f" % (p[6] / pp.PAPER_N_T1)] + [its(r) for r in rr] + [p[5]] +
                    [its(ras.get((m, q))) for q in POU])
    out = ["## Coarse space vs paper (Table 1): threshold tau and partition of unity\n\n",
           "n0/N = coarse unknowns per subdomain (N = %s here, 256 in the paper, nev = 300 caps it at 300). "
           "Paper: tau = 0.6, but its G3_circuit row equals the tau = 0.9 point of its Fig. 2. Two-level "
           "iterations do not depend on the coarse solver; the one-level RAS depends on D_i.\n\n" % N]
    out.append(table(["matrix"] + ["n0/N %s" % lab(("", k[0], k[1])) for k in keys] + ["n0/N paper"] +
                     ["its %s" % lab(("", k[0], k[1])) for k in keys] + ["its paper"] +
                     ["RAS its %s" % POU[q] for q in POU], rows))
    if pp.HAVE_MPL and rows:
        mats = [r[0] for r in rows]
        nk = len(keys)
        f, axes = pp.fig(2, w=6.6, h=4.0)
        colors = [pp.COLORS[0], pp.COLORS[2], pp.COLORS[1], pp.COLORS[3]]
        w = 0.8 / (nk + 1)
        for panel, (ax, off) in enumerate(zip(axes, (1, 2 + nk))):
            for k in range(nk + 1):
                ys = [pp.num(str(r[off + k]).rstrip("*")) or 0 for r in rows]
                ax.bar([i - 0.4 + w * (k + 0.5) for i in range(len(mats))], ys, width=w,
                       color=pp.INK2 if k == nk else colors[k % len(colors)],
                       label="paper" if k == nk else lab(("",) + keys[k]), edgecolor=pp.SURF, linewidth=0.8)
            ax.set_xticks(range(len(mats)))
            ax.set_xticklabels(mats, rotation=45, ha="right", fontsize=7)
            ax.xaxis.grid(False)
        axes[0].set_yscale("log")
        pp.finish(axes[0], "Coarse space size per subdomain", "", "n0 / N")
        pp.finish(axes[1], "Two-level GMRES iterations", "", "iterations")
        out.append(pp.save(f, "tab1_vs_paper.png"))
    return "".join(out)


def sec_eig(runs):
    rows, pairs = [], []
    for m in pp.PAPER_T1:
        fx, ad = runs.get("eig_%s_nevfixed" % m), runs.get("tab1_%s_bool_t0.6_root" % m)
        if not fx and not ad:
            continue
        pairs.append((m, fx, ad))

        def arn(r, i):
            t = r["timers"].get("setup.eigensolver.arnoldi") if r and r["timers"] else None
            return ft(t[i]) if t else ""

        def opshare(r):
            t, o = (r["timers"].get("setup.eigensolver.arnoldi"), r["timers"].get("setup.eigensolver.arnoldi.operator")) \
                if r and r["timers"] else (None, None)
            return "%.0f%%" % (100 * o[1] / t[1]) if t and o and t[1] > 0 else ""
        rows.append([m, fx.get("n0", "") if fx else "", ad.get("n0", "") if ad else "", its(fx), its(ad),
                     "%.0f / %d" % ad["k"] if ad and ad.get("k") else "", fx.get("ops", "") if fx else "",
                     ad.get("ops", "") if ad else "", arn(fx, 1), arn(ad, 1), arn(fx, 2), arn(ad, 2), opshare(fx),
                     opshare(ad), ft(fx.get("t_setup")) if fx else "", ft(ad.get("t_setup")) if ad else ""])
    if not rows:
        return ""
    out = ["## Local eigensolver: nev = 300 always computed vs adaptive k (Boolean, tau = 0.6, root)\n\n",
           "fixed = `--nev-start 0` (as in run 22838900), adaptive = default `--nev-start 32`: k grows until the "
           "smallest computed |lambda| is below 1/tau. n0 and iterations must be equal. Arnoldi time = max (and "
           "avg) over the ranks; operator = share of the Arnoldi time spent applying T (sparse solves), the rest "
           "is dense Krylov work.\n\n"]
    out.append(table(["matrix", "n0 fixed", "n0 adaptive", "its fixed", "its adaptive", "k avg / max",
                      "max op. appl. fixed", "max op. appl. adaptive", "Arnoldi avg fixed [s]",
                      "Arnoldi avg adaptive [s]", "Arnoldi max fixed [s]", "Arnoldi max adaptive [s]",
                      "operator share fixed", "operator share adaptive", "setup fixed [s]", "setup adaptive [s]"], rows))
    if pp.HAVE_MPL:
        ok_pairs = [(m, fx, ad) for m, fx, ad in pairs if fx and ad and fx["timers"] and ad["timers"]]
        if ok_pairs:
            f, (ax,) = pp.fig(1, w=7.2, h=4.0)
            x = list(range(len(ok_pairs)))
            for k, (lab_, idx) in enumerate((("fixed nev = 300", 1), ("adaptive k", 2))):
                ys = [T(p[idx], ["setup.eigensolver.arnoldi"]) or 0 for p in ok_pairs]
                ax.bar([i - 0.2 + 0.4 * k for i in x], ys, width=0.4, color=pp.COLORS[k], label=lab_,
                       edgecolor=pp.SURF, linewidth=0.8)
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels([p[0] for p in ok_pairs], rotation=45, ha="right", fontsize=7)
            ax.xaxis.grid(False)
            pp.finish(ax, "Arnoldi time (max over ranks)", "", "seconds")
            out.append(pp.save(f, "eig_adaptive.png"))
    return "".join(out)


def sec_fig2(runs):
    pts = {}
    for r in runs.values():
        m = re.match(r"fig2_(bool|mult)_tau(.+)$", r["name"])
        if m:
            pts[(m.group(1), float(m.group(2)))] = r
    if not pts:
        return ""
    out = ["## Fig. 2 - G3_circuit, influence of tau (nev = 300), both PoU\n\n"]
    rows = []
    for tau in pp.PAPER_F2:
        rb, ru = pts.get(("bool", tau)), pts.get(("mult", tau))
        rows.append([tau, rb.get("n0", "") if rb else "", ru.get("n0", "") if ru else "", pp.PAPER_F2[tau][0],
                     its(rb), its(ru), pp.PAPER_F2[tau][1], ft(cf(rb)) if rb else "", ft(cf(ru)) if ru else "",
                     ft(rb.get("t_setup")) if rb else "", ft(ru.get("t_setup")) if ru else ""])
    out.append(table(["tau", "n0 Boolean", "n0 mult.", "n0 paper (N=256)", "its Boolean", "its mult.", "its paper",
                      "A_00 fact. Boolean [s]", "A_00 fact. mult. [s]", "setup Boolean [s]", "setup mult. [s]"], rows))
    if pp.HAVE_MPL:
        f, axes = pp.fig(2, w=5.2, h=3.9)
        taus = list(pp.PAPER_F2)
        for k, (pou, lab) in enumerate(POU.items()):
            for ax, key in zip(axes, ("n0", "its")):
                xy = [(t, pts[(pou, t)].get(key)) for t in taus if (pou, t) in pts and pts[(pou, t)].get(key) is not None]
                if xy:
                    ax.plot([a for a, _ in xy], [b for _, b in xy], color=pp.COLORS[k], marker=pp.MARK[k],
                            linewidth=1.6, label=lab)
        axes[0].plot(taus, [pp.PAPER_F2[t][0] for t in taus], color=pp.INK2, marker=pp.MARK[2], linestyle="--",
                     linewidth=1.4, label="paper (N=256)")
        axes[1].plot(taus, [pp.PAPER_F2[t][1] for t in taus], color=pp.INK2, marker=pp.MARK[2], linestyle="--",
                     linewidth=1.4, label="paper")
        axes[0].set_yscale("log")
        pp.finish(axes[0], "Coarse space size", "tau", "n0")
        pp.finish(axes[1], "GMRES iterations", "tau", "iterations")
        out.append(pp.save(f, "fig2_pou.png"))
    return "".join(out)


def sec_tab2(runs):
    data = {}
    for r in runs.values():
        m = re.match(r"tab2_(2d|3d)_nu([^_]+)_(bool|mult)$", r["name"])
        if m:
            data[(m.group(1), float(m.group(2)), m.group(3))] = r
    if not data:
        return ""
    out = ["## Table 2 - convection-diffusion (4.1), both PoU (tau = 0.3, nev = 60)\n\n"]
    rows = []
    for dim in ("2d", "3d"):
        Np, _, pv = pp.PAPER_T2[dim]
        for nu in sorted(pv, reverse=True):
            rb, ru = data.get((dim, nu, "bool")), data.get((dim, nu, "mult"))
            N = (rb or ru or {}).get("np") or 1
            rows.append([dim.upper(), "%g" % nu, its(rb), its(ru), pv[nu][0]] +
                        ["%.0f" % (r["n0"] / N) if r and r.get("n0") is not None else "" for r in (rb, ru)] +
                        ["%.0f" % (pv[nu][1] / Np)])
    out.append(table(["dim", "nu", "its Boolean", "its mult.", "its paper", "n0/N Boolean", "n0/N mult.",
                      "n0/N paper"], rows))
    if pp.HAVE_MPL:
        f, axes = pp.fig(2, w=4.8, h=3.8)
        for ax, dim in zip(axes, ("2d", "3d")):
            nus = sorted(pp.PAPER_T2[dim][2])
            for k, (pou, lab) in enumerate(POU.items()):
                xy = [(nu, data[(dim, nu, pou)].get("its")) for nu in nus if (dim, nu, pou) in data
                      and data[(dim, nu, pou)]["status"] == "ok"]
                if xy:
                    ax.plot([a for a, _ in xy], [b for _, b in xy], color=pp.COLORS[k], marker=pp.MARK[k],
                            linewidth=1.6, label=lab)
            ax.plot(nus, [pp.PAPER_T2[dim][2][nu][0] for nu in nus], color=pp.INK2, marker=pp.MARK[2],
                    linestyle="--", linewidth=1.4, label="paper")
            ax.set_xscale("log")
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(pp.MaxNLocator(integer=True))
            pp.finish(ax, dim.upper(), "viscosity nu", "GMRES iterations")
        out.append(pp.save(f, "tab2_pou.png"))
    return "".join(out)


def sec_fig4(runs):
    data = OrderedDict()
    for r in runs.values():
        m = re.match(r"fig4_(bool|mult)_N(\d+)$", r["name"])
        if m and r.get("total") is not None:
            data.setdefault(m.group(1), []).append(r)
    if not data:
        return ""
    out = ["## Fig. 4 - strong scaling, 3D convection-diffusion, nu = 1e-2, both PoU\n\n"]
    rows = []
    for pou, rr in data.items():
        rr.sort(key=lambda r: r["np"])
        b = rr[0]
        for r in rr:
            sp = b["total"] / r["total"] * b["np"]
            rows.append([POU[pou], r["np"], r.get("n0", ""), its(r), pp.PAPER_F4.get(r["np"], ""), ft(cf(r)),
                         ft(r["t_setup"]), ft(r["t_solve"]), ft(r["total"]), "%.2f" % (sp / r["np"])])
    out.append(table(["PoU", "N", "n0", "its", "paper its", "A_00 fact. [s]", "setup [s]", "solve [s]", "total [s]",
                      "efficiency"], rows))
    if pp.HAVE_MPL:
        f, (ax,) = pp.fig(1, w=6.4, h=4.0)
        for k, (pou, rr) in enumerate(data.items()):
            ax.plot([r["np"] for r in rr], [r["total"] for r in rr], color=pp.COLORS[k], marker=pp.MARK[k],
                    linewidth=1.6, label="%s (its %s)" % (POU[pou], ", ".join(str(r.get("its", "?")) for r in rr)))
        b = next(iter(data.values()))[0]
        nps = sorted({r["np"] for rr in data.values() for r in rr})
        ax.plot(nps, [b["total"] * b["np"] / n for n in nps], color=pp.IDEAL, linestyle="--", linewidth=1.2,
                label="ideal")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(nps)
        ax.set_xticklabels([str(n) for n in nps])
        pp.finish(ax, "Strong scaling", "number of subdomains", "time to solution [s]")
        out.append(pp.save(f, "fig4_pou.png"))
    return "".join(out)


def main():
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        print(__doc__)
        sys.exit(1)
    pp.RUN = os.path.abspath(sys.argv[1])
    runs, cfg = load()
    pp.write_csvs(runs)
    parts = ["# schwarz2lvl: distributed coarse solver and partition of unity\n\n",
             "Run directory `%s` (%s).%s\n\n" % (os.path.basename(pp.RUN), ", ".join("%s=%s" % kv for kv in cfg.items()),
                                                 "" if pp.HAVE_MPL else " matplotlib not available: tables only "
                                                 "(`python3 -m pip install --user matplotlib`)."),
             pp.sec_status(runs), sec_eig(runs), sec_coarse(runs, cfg), sec_pou(runs), sec_fig2(runs), sec_tab2(runs), sec_fig4(runs),
             pp.sec_phases(runs)]
    with open(os.path.join(pp.RUN, "report.md"), "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print("written %s/report.md (%d runs, plots: %s)" % (pp.RUN, len(runs), "yes" if pp.HAVE_MPL else "no"))


if __name__ == "__main__":
    main()
