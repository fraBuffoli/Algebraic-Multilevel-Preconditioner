#!/usr/bin/env python3
"""Report of the warm-start campaign (Allrun_warm.sh): OpenFOAM vs two-level / one-level Schwarz in a
PIMPLE-like time loop (physical p_rgh source, warm start, correctors to relTol / tolerance, last
corrector to the tolerance only).

    python3 report_warm.py run_warm_np240_<jobid> [run_warm_np432_<jobid> ...] [--ref RUN_SCALING ...] [--out DIR]

Runs are merged by (mesh size N, processes NP, variant); a triple present in several runs is taken
from the newest. Runs with different SOURCE / WARM / RTOL / ATOL are not merged: give them in
separate calls. --ref adds the GAMG / Schwarz ratio of the scaling campaign (run_scaling_np<NP>_*:
alpha source, every solve from 0, relTol 1e-2) at the same N and NP, for comparison. Writes
DIR/report.md, DIR/summary.csv and, with matplotlib, DIR/plots/*.png. Default DIR: warm_report
(next to the first run).

Times (csv/<run>.csv, every solve between two synchronizations of all processes):
  schwarz2lvl  t_update (initial setup / rebuild / refactorization / reuse) + t_solve (BiCGStab)
  OpenFOAM     t_setup (construction of the solver: GAMG coarse matrices) + t_cycles
  cost per step = sum over the run / STEPS (the first step also carries the initial setup and starts
  from 0); steady = steps 1.. only (what a long run pays per step, rebuilds included).
Matrix assembly (OpenFOAM) and matrix generation (schwarz2lvl) are not part of the costs.
The iterations of schwarz2lvl are applications of the preconditioner (two per BiCGStab step), those
of OpenFOAM V-cycles (GAMG) or PCG iterations.
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

S2L_ORDER = ["s2l_2L", "s2l_1L"]
LABEL = {"s2l_2L": "Schwarz 2-level", "s2l_1L": "RAS 1-level", "W_GAMG": "GAMG (DIC)",
         "W_GAMG_PCG": "GAMG, PCG+GAMG final", "W_PCG_DIC": "PCG + DIC"}
COLOR = {"s2l_2L": "#2a78d6", "s2l_1L": "#7a5bd6", "W_GAMG": "#eb6834", "W_GAMG_PCG": "#f3a485", "W_PCG_DIC": "#a44924"}


def is_s2l(v):
    return v.startswith("s2l_")


def label(v):
    return LABEL.get(v, v)


def load_run(run):
    cfg = read_config(run)
    status = read_status(run)
    np_ = int(cfg.get("NP", "0") or 0)
    steps = int(cfg.get("STEPS", "0") or 0)
    corr = int(cfg.get("CORR", "0") or 0)
    recs = []
    for f in sorted(glob.glob(os.path.join(run, "csv", "bub*_*.csv"))):
        name = os.path.basename(f)[:-4]
        m = re.match(r"bub(\d+)_(.+)$", name)
        if not m:
            continue
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        var = m.group(2)
        s = OrderedDict(var=var, N=int(m.group(1)), NP=np_ or int(rows[0]["np"]), run=os.path.basename(run.rstrip("/")),
                        name=name)
        ncorr = max(int(r["corrector"]) for r in rows) + 1
        last = [int(r["corrector"]) == ncorr - 1 for r in rows]
        step = [int(r["step"]) for r in rows]
        its = [int(float(r["iterations"])) for r in rows]
        if is_s2l(var):
            per = [float(r["t_update"]) + float(r["t_solve"]) for r in rows]
            sol = [float(r["t_solve"]) for r in rows]
            init_res = [float(r.get("initial_res_of") or 0) for r in rows]
            fin_res = [float(r.get("final_res_of") or 0) for r in rows]
            timers = read_timers(os.path.join(run, "log." + name))
            sm, _, _ = summarize_s2l(rows, timers)
            if sm["local"] == 0.0 and sm["coarse"] == 0.0:
                # One level: the preconditioner is the local solves only (no solve.preconditioner.local timer).
                tot, pre = timers.get("solve", (0, 0, 0))[1], timers.get("solve.preconditioner", (0, 0, 0))[1]
                if tot > 0:
                    sm["local"] = sm["iter"] * pre / tot
                    sm["krylov"] = sm["iter"] - sm["local"]
            s.update(init=sm["init"], upd=sm["upd"], iter=sm["iter"], rebuilds=sm["rebuilds"], refact=sm["refact"],
                     rebuild_med=sm["rebuild_med"], local=sm["local"], coarse=sm["coarse"], krylov=sm["krylov"],
                     t_generate=sm["t_generate"])
        else:
            per = [float(r["t_solve"]) for r in rows]
            sol = [float(r["t_cycles"]) for r in rows]
            init_res = [float(r["initial_res"]) for r in rows]
            fin_res = [float(r["final_res"]) for r in rows]
            s.update(init=float(rows[0]["t_setup"]), upd=sum(float(r["t_setup"]) for r in rows[1:]), iter=sum(sol))
        st_ = status.get(name, ("missing", 0.0))
        nsteps = max(step) + 1
        s.update(status=st_[0], wall=st_[1], solves=len(rows), expected=(steps * corr) or len(rows), steps=nsteps,
                 converged=sum(int(float(r["converged"])) for r in rows), its=its, per=per, last=last, step=step,
                 init_res=init_res, fin_res=fin_res)
        s["total"] = sum(per)
        s["per_step"] = s["total"] / nsteps
        steady = [p for p, k in zip(per, step) if k > 0]
        s["steady_step"] = sum(steady) / max(1, nsteps - 1)
        c_its = [i for i, l in zip(its, last) if not l]
        f_its = [i for i, l in zip(its, last) if l]
        s["its_corr"] = st.mean(c_its) if c_its else 0.0
        s["its_corr_rng"] = (min(c_its), max(c_its)) if c_its else (0, 0)
        s["its_last"] = st.mean(f_its) if f_its else 0.0
        s["its_last_rng"] = (min(f_its), max(f_its)) if f_its else (0, 0)
        s["t_last"] = sum(p for p, l in zip(per, last) if l)
        s["t_it"] = s["iter"] / max(1, sum(its))
        # Initial normalised residual of the first corrector of the steps after the first (warm start state).
        r0 = [r for r, k, c in zip(init_res, step, (int(x["corrector"]) for x in rows)) if k > 0 and c == 0]
        s["res0_med"] = st.median(r0) if r0 else None
        s["setup_share"] = (s["init"] + s["upd"]) / s["total"] if s["total"] > 0 else 0.0
        s["ok"] = s["status"] == "ok" and s["solves"] == s["expected"] and s["converged"] == s["solves"]
        recs.append(s)
    return cfg, recs


def plots(outdir, data, Ns, NPs, variants):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    out = []
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "text.color": INK, "axes.labelcolor": INK2})
    # 1. Cost per step, grouped by mesh, one figure per NP.
    for np_ in NPs:
        fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
        style_axes(ax)
        vs = [v for v in variants if any((N, np_, v) in data for N in Ns)]
        w = 0.8 / max(1, len(vs))
        for i, v in enumerate(vs):
            xs, ys = [], []
            for j, N in enumerate(Ns):
                s = data.get((N, np_, v))
                if s:
                    xs.append(j - 0.4 + w * (i + 0.5))
                    ys.append(s["per_step"])
            bars = ax.bar(xs, ys, width=w * 0.92, color=COLOR.get(v, MUTED), label=label(v))
            for b, y in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, y, fmt(y), ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xticks(range(len(Ns)))
        ax.set_xticklabels(["%d$^3$" % N for N in Ns])
        ax.set_ylabel("cost per time step [s]")
        ax.set_title("Cost per time step, %d processes (all correctors, setup included)" % np_, fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        p = os.path.join(outdir, "plots", "cost_per_step_np%d.png" % np_)
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        out.append(p)
    # 2. Every solve of the largest mesh: applications / V-cycles and cost, last correctors filled.
    for np_ in NPs:
        Nmax = max((N for N in Ns if any((N, np_, v) in data for v in variants)), default=None)
        if Nmax is None:
            continue
        fig, axs = plt.subplots(2, 1, figsize=(7.6, 5.2), sharex=True, facecolor=SURFACE)
        for ax in axs:
            style_axes(ax)
        for v in variants:
            s = data.get((Nmax, np_, v))
            if not s:
                continue
            x = list(range(len(s["its"])))
            c = COLOR.get(v, MUTED)
            axs[0].plot(x, s["its"], color=c, linewidth=1.0, label=label(v))
            axs[0].scatter([i for i in x if s["last"][i]], [s["its"][i] for i in x if s["last"][i]], color=c, s=10)
            axs[1].plot(x, s["per"], color=c, linewidth=1.0)
            axs[1].scatter([i for i in x if s["last"][i]], [s["per"][i] for i in x if s["last"][i]], color=c, s=10)
        axs[0].set_ylabel("iterations per solve")
        axs[1].set_ylabel("cost per solve [s]")
        axs[1].set_yscale("log")
        axs[1].set_xlabel("solve (dots: last corrector of the step)")
        axs[0].set_title("Every solve, %d$^3$, %d processes" % (Nmax, np_), fontsize=10, color=INK, pad=20)
        axs[0].legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2)
        fig.tight_layout()
        p = os.path.join(outdir, "plots", "solves_N%d_np%d.png" % (Nmax, np_))
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        out.append(p)
    return out


def main(argv):
    runs, refs, outdir = [], [], None
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            outdir = argv[i + 1]
            i += 2
        elif argv[i] == "--ref":
            refs.append(argv[i + 1])
            i += 2
        else:
            runs.append(argv[i])
            i += 1
    runs = [r for r in runs if os.path.isdir(os.path.join(r, "csv"))]
    if not runs:
        print(__doc__)
        return 1
    runs.sort(key=job_key)
    outdir = outdir or os.path.join(os.path.dirname(os.path.abspath(runs[0].rstrip("/"))), "warm_report")
    os.makedirs(outdir, exist_ok=True)

    data, cfgs, setting = OrderedDict(), {}, None
    for r in runs:
        cfg, recs = load_run(r)
        key = tuple(cfg.get(k, "?") for k in ("SOURCE", "WARM", "RTOL", "ATOL"))
        if setting is None:
            setting = key
        elif key != setting:
            print("%s: SOURCE/WARM/RTOL/ATOL %s differ from %s: skipped (report it separately)" % (r, key, setting))
            continue
        cfgs[os.path.basename(r.rstrip("/"))] = (cfg, r)
        for s in recs:
            data[(s["N"], s["NP"], s["var"])] = s  # newest run wins (sorted by job id)
    ref = {}
    for r in refs:
        try:
            _, recs = load_scaling_run(r)
        except Exception as e:  # noqa: BLE001
            print("--ref %s: %s" % (r, e))
            continue
        for s in recs:
            ref.setdefault((s["N"], s["NP"]), {})[s["code"]] = s

    Ns = sorted({k[0] for k in data})
    NPs = sorted({k[1] for k in data})
    allv = {k[2] for k in data}
    variants = [v for v in S2L_ORDER if v in allv] + sorted(v for v in allv if not is_s2l(v))
    ofv = [v for v in variants if not is_s2l(v)]

    def best_of(N, np_, key):
        c = [(data[(N, np_, v)][key], v) for v in ofv if (N, np_, v) in data and data[(N, np_, v)]["ok"]]
        return min(c) if c else (None, None)

    L = []
    w = L.append
    src, warm, rtol, atol = setting
    w("# Warm-start campaign: report\n")
    w("Runs: %s.\n" % ", ".join("`%s`" % k for k in cfgs))
    c0 = next(iter(cfgs.values()))[0]
    w("Right-hand side **%s**, %s, correctors relTol %s or tolerance %s, last corrector of every step tolerance %s "
      "(relTol 0); %s steps x %s correctors; bubbles nb=%s, R=%s, rho ratio %s, cfl %s.\n"
      % (src, "every solve from the previous solution" if warm == "1" else "every solve from 0", rtol, atol, atol,
         c0.get("STEPS"), c0.get("CORR"), c0.get("NBUB"), c0.get("RADIUS"), c0.get("RHO"), c0.get("CFL")))
    w("\n## Checks\n")
    for k, (_, rd) in cfgs.items():
        for f, what in (("binaries.txt", "binaries"), ("log.check_matrix", "matrix check")):
            p = os.path.join(rd, f)
            if os.path.exists(p):
                txt = " / ".join(x.strip() for x in open(p) if x.strip())
                w("- `%s` %s: %s" % (k, what, txt))
    bad = [s for s in data.values() if not s["ok"]]
    w("- runs: %d, all complete and converged: %s" % (len(data), "yes" if not bad else "**no**"))
    for s in bad:
        w("  - `%s` (%s): status %s, %d of %d solves, %d converged" % (s["name"], s["run"], s["status"], s["solves"],
                                                                      s["expected"], s["converged"]))
    w("")

    w("## Cost per time step [s]\n")
    w("All correctors of a step, setup and updates included; *all*: the whole run / steps (the first step "
      "starts from 0 and carries the initial setup); *steady*: steps 1.. only. Ratio = best OpenFOAM variant / "
      "Schwarz variant (> 1: Schwarz faster).\n")
    hdr = "| N | NP | " + " | ".join(label(v) for v in variants) + " | best OF / 2-level | best OF / 1-level |"
    w(hdr)
    w("|" + "--:|" * (2 + len(variants) + 2))
    for N in Ns:
        for np_ in NPs:
            if not any((N, np_, v) in data for v in variants):
                continue
            cells = []
            for v in variants:
                s = data.get((N, np_, v))
                cells.append("-" if not s else "%s / %s%s" % (fmt(s["per_step"]), fmt(s["steady_step"]), "" if s["ok"] else " (!)"))
            ratios = []
            for v in S2L_ORDER:
                s = data.get((N, np_, v))
                b, bv = best_of(N, np_, "per_step")
                bs, _ = best_of(N, np_, "steady_step")
                ratios.append("-" if not (s and b) else "**%.2f×** / %.2f× (%s)" % (b / s["per_step"], bs / s["steady_step"], label(bv)))
            w("| %d³ | %d | %s | %s |" % (N, np_, " | ".join(cells), " | ".join(ratios)))
    w("\nEach cell: all / steady. (!): run incomplete or not converged.\n")

    if ref:
        w("### Against the earlier setting (scaling campaign: alpha source, every solve from 0, relTol 1e-2)\n")
        w("| N | NP | GAMG / Schwarz, scaling campaign (cost per solve) | best OF / 2-level here (cost per step, all) |")
        w("|--:|--:|--:|--:|")
        for N in Ns:
            for np_ in NPs:
                r = ref.get((N, np_), {})
                s = data.get((N, np_, "s2l_2L"))
                b, _ = best_of(N, np_, "per_step")
                if "s2l" in r and "of" in r and s and b:
                    w("| %d³ | %d | %.2f× | %.2f× |" % (N, np_, r["of"]["cost"] / r["s2l"]["cost"], b / s["per_step"]))
        w("")

    w("## Iterations\n")
    w("Correctors / last corrector: average (min - max) per solve. *res0*: median initial normalised residual of "
      "the first corrector of the steps after the first (state of the warm start). *ms/it*: time of the "
      "iterations / iterations. *last*: share of the cost spent in the last correctors.\n")
    w("| N | NP | variant | correctors | last corrector | res0 | ms/it | last | final residual of the last correctors (max) |")
    w("|--:|--:|:--|--:|--:|--:|--:|--:|--:|")
    for N in Ns:
        for np_ in NPs:
            for v in variants:
                s = data.get((N, np_, v))
                if not s:
                    continue
                w("| %d³ | %d | %s | %.1f (%d - %d) | %.1f (%d - %d) | %s | %s | %.0f%% | %.1e |"
                  % (N, np_, label(v), s["its_corr"], s["its_corr_rng"][0], s["its_corr_rng"][1], s["its_last"],
                     s["its_last_rng"][0], s["its_last_rng"][1],
                     "-" if s["res0_med"] is None else "%.1e" % s["res0_med"], fmt(1e3 * s["t_it"]),
                     100 * s["t_last"] / s["total"] if s["total"] else 0,
                     max((r for r, l in zip(s["fin_res"], s["last"]) if l), default=0.0)))
    w("")

    w("## Schwarz: setup and phases\n")
    w("Per step [s]: initial setup (amortized), updates (rebuilds, refactorizations), and the iterations split "
      "with the timers of the log (local solves, coarse correction, Krylov).\n")
    w("| N | NP | variant | initial setup | updates | local solves | coarse corr. | Krylov | rebuilds (median [s]) | setup share |")
    w("|--:|--:|:--|--:|--:|--:|--:|--:|--:|--:|")
    for N in Ns:
        for np_ in NPs:
            for v in S2L_ORDER:
                s = data.get((N, np_, v))
                if not s:
                    continue
                k = s["steps"]
                w("| %d³ | %d | %s | %s | %s | %s | %s | %s | %d (%s) | %.0f%% |"
                  % (N, np_, label(v), fmt(s["init"] / k), fmt(s["upd"] / k), fmt(s["local"] / k), fmt(s["coarse"] / k),
                     fmt(s["krylov"] / k), s["rebuilds"], fmt(s["rebuild_med"]), 100 * s["setup_share"]))
    w("")

    pl = plots(outdir, data, Ns, NPs, variants)
    if pl:
        w("## Plots\n")
        for p in pl:
            w("![%s](%s)\n" % (os.path.basename(p)[:-4], os.path.relpath(p, outdir)))

    with open(os.path.join(outdir, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["N", "NP", "variant", "run", "status", "solves", "converged", "cost_per_step", "steady_per_step",
                     "its_correctors", "its_last", "res0_median", "ms_per_it", "init", "updates", "iterations",
                     "setup_share"])
        for (N, np_, v), s in sorted(data.items()):
            cw.writerow([N, np_, v, s["run"], s["status"], s["solves"], s["converged"], "%.6g" % s["per_step"],
                         "%.6g" % s["steady_step"], "%.3f" % s["its_corr"], "%.3f" % s["its_last"],
                         "" if s["res0_med"] is None else "%.3e" % s["res0_med"], "%.4g" % (1e3 * s["t_it"]),
                         "%.6g" % s["init"], "%.6g" % s["upd"], "%.6g" % s["iter"], "%.4f" % s["setup_share"]])
    print("%s, %s" % (os.path.join(outdir, "report.md"), os.path.join(outdir, "summary.csv")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
