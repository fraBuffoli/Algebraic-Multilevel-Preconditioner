#!/usr/bin/env python3
"""
plot_paper.py - tables and plots of the runs of Allrun.sh, compared with the paper
(Al Daas, Jolivet, Rees, SISC 2023, Section 4).

    python3 plot_paper.py run_<jobid>

Writes into the run directory:
    report.md      tables: Table 1, Fig. 2, Fig. 4, Table 2 (ours vs paper) + time of every phase
    summary.csv    one row per run
    phases.csv     every timer of every run (min / avg / max over the ranks)
    plots/*.png    only if matplotlib is available (pip install --user matplotlib)

Python >= 3.6, standard library only (matplotlib optional).
"""
import csv
import glob
import os
import re
import sys
from collections import OrderedDict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# ----------------------------------------------------------------------------- paper data
# Table 1: n, nnz, AGMG, BoomerAMG, GAMG, M_deflated iterations, n0  (N = 256 subdomains)
# "" = more than 100 iterations, "F" = failure (dagger), "C" = complex valued (double dagger)
PAPER_T1 = OrderedDict([
    ("light_in_tissue", (29282, 406084, "15", "C", "53", 6, 7230)),
    ("finan512", (74752, 596992, "9", "7", "8", 6, 2591)),
    ("consph", (83334, 6010480, "", "", "", 93, 31136)),
    ("Dubcova3", (146689, 3636643, "", "72", "71", 7, 21047)),
    ("CO", (221119, 7666057, "", "25", "", 26, 56135)),
    ("nxp1", (414604, 2655880, "F", "F", "F", 20, 19707)),
    ("CoupCons3D", (416800, 17277420, "", "F", "26", 20, 28925)),
    ("parabolic_fem", (525825, 3674625, "12", "8", "16", 5, 24741)),
    ("Chevron4", (711450, 6376412, "", "C", "F", 5, 22785)),
    ("apache2", (715176, 4817870, "14", "11", "35", 8, 45966)),
    ("tmt_sym", (726713, 5080961, "14", "10", "17", 5, 28253)),
    ("tmt_unsym", (917825, 4584801, "23", "13", "18", 6, 32947)),
    ("ecology2", (999999, 4995991, "18", "12", "18", 6, 34080)),
    ("thermal2", (1228045, 8580313, "18", "14", "20", 26, 40098)),
    ("atmosmodj", (1270432, 8814880, "F", "8", "17", 7, 76368)),
    ("G3_circuit", (1585478, 7660826, "25", "12", "35", 8, 71385)),
    ("Transport", (1602111, 23487281, "18", "10", "98", 9, 76800)),
    ("memchip", (2707524, 13343948, "F", "15", "F", 36, 57942)),
    ("circuit5M_dc", (3523317, 14865409, "F", "5", "", 7, 8629)),
])
PAPER_N_T1 = 256
# Fig. 2 (G3_circuit, N = 256): tau -> (n0, iterations); one-level ASM does not converge
PAPER_F2 = OrderedDict([(0.02, (739, 75)), (0.05, (1831, 44)), (0.1, (3608, 31)),
                        (0.3, (11784, 18)), (0.6, (34402, 11)), (0.9, (71385, 8))])
# Fig. 4 (3D, nu = 1e-2, n = 8.1e6): N -> iterations
PAPER_F4 = OrderedDict([(32, 28), (64, 21), (128, 19), (256, 20), (512, 15)])
# Table 2: dim -> (N, n, {nu: (iterations, n0)})
PAPER_T2 = {
    "2d": (1024, 6.3e6, {1.0: (23, 52875), 1e-1: (20, 52872), 1e-2: (19, 52759), 1e-3: (20, 47497), 1e-4: (21, 28235)}),
    "3d": (4096, 8.1e6, {1.0: (18, 1.8e5), 1e-1: (14, 1.8e5), 1e-2: (11, 1.6e5), 1e-3: (16, 97657), 1e-4: (29, 76853)}),
}

# Phases of the stacked time plots: label -> timers (max over ranks). "eig: kernel/dense"
# = setup.eigensolver minus its LU and Arnoldi parts (kernel detection, dense path for small
# subdomains, selection of the eigenvectors).
PHASES = [
    ("subdomain + halo", ["setup.subdomain"]),
    ("local LU of A_ii", ["setup.local_factorization"]),
    ("eig: LU of A~+sI", ["setup.eigensolver.factorization"]),
    ("eig: kernel/dense", None),
    ("eig: Arnoldi", ["setup.eigensolver.arnoldi"]),
    ("coarse basis + A_00", ["setup.coarse_basis", "setup.coarse_assembly"]),
    ("coarse LU", ["setup.coarse_factorization"]),
    ("GMRES", ["solve"]),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]   # ordinal blue ramp
MARK = ["o", "s", "^", "D", "v", "P", "X"]
INK, INK2, GRID, SURF, IDEAL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#8a8984"

RUN = None


# ----------------------------------------------------------------------------- reading
def num(x):
    for c in (int, float):
        try:
            return c(x)
        except (TypeError, ValueError):
            pass
    return None


def parse_log(path):
    r = {"timers": OrderedDict()}
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    m = re.search(r"^Problem\n\s+(.*)$", text, re.M)
    if m:
        r["desc"] = m.group(1).strip()
        mm = re.search(r"n=(\d+), nnz=(\d+)", r["desc"])
        if mm:
            r["n"], r["nnz"] = int(mm.group(1)), int(mm.group(2))
    m = re.search(r"^\s+(NOT converged|converged) in (\d+) iterations", text, re.M)
    if m:
        r["converged"], r["its"] = (m.group(1) == "converged"), int(m.group(2))
    m = re.search(r"relative residual \(true\)\s+(\S+)", text)
    if m:
        r["true_res"] = float(m.group(1))
    for key, rx in (("n0", r"^\s+n0\s+(\d+)"), ("ker", r"subdomains with ker\(A~\)\s+(\d+)"),
                    ("arnoldi_nc", r"Arnoldi not converged\s+(\d+)")):
        m = re.search(rx, text, re.M)
        if m:
            r[key] = int(m.group(1))
    for key, rx in (("m", r"m_i min/avg/max\s+(\S+) / (\S+) / (\S+)"),
                    ("nI", r"owned unknowns n_I\s+(\S+) / (\S+) / (\S+)"),
                    ("nG", r"overlap unknowns n_Gamma\s+(\S+) / (\S+) / (\S+)")):
        m = re.search(rx, text)
        if m:
            r[key] = tuple(float(v) for v in m.groups())
    cs = re.findall(r"^\s+coarse solver\s{2,}(\S+)\s*$", text, re.M)
    if cs:
        r["coarse_solver"] = cs[-1]
    if "Timings [s]" in text:
        for line in text.split("Timings [s]", 1)[1].splitlines()[2:]:
            m = re.match(r"^\s+(\S+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", line)
            if m:
                r["timers"][m.group(1)] = tuple(float(v) for v in m.groups()[1:])
    errs = re.findall(r"\[rank \d+\] error: ([^\n\[]+)", text)
    if errs:
        r["error"] = max(errs, key=len).strip()
    return r


def load():
    status = {}
    if os.path.isfile(os.path.join(RUN, "status.txt")):
        with open(os.path.join(RUN, "status.txt")) as f:
            for line in f:
                p = line.split()
                if len(p) == 4:
                    status[p[0]] = (int(p[1]), p[2], int(p[3]))
    runs = OrderedDict()
    for path in sorted(glob.glob(os.path.join(RUN, "log.*"))):
        name = os.path.basename(path)[4:]
        if name in ("build", "modules", "git") or name.startswith("test_"):
            continue
        r = parse_log(path)
        r["name"] = name
        np_, st, wall = status.get(name, (None, "killed", None))   # no status: job ended during the run
        r["np"], r["status"], r["wall"] = np_, st, wall
        c = os.path.join(RUN, "csv", name + ".csv")
        if os.path.isfile(c):
            with open(c, encoding="utf-8", errors="replace") as f:
                rows = list(csv.DictReader(f))
            if rows:
                for k in ("t_setup", "t_solve", "t_eig", "t_coarse_asm", "t_coarse_fact", "true_rel_res"):
                    r[k] = num(rows[-1].get(k))
                r["np"] = r["np"] or num(rows[-1].get("np"))
        r["hist"] = []
        h = os.path.join(RUN, "hist", name + ".txt")
        if os.path.isfile(h):
            with open(h) as f:
                r["hist"] = [float(l.split()[1]) for l in f if len(l.split()) == 2]
        ts, tv = r.get("t_setup"), r.get("t_solve")
        r["total"] = ts + tv if ts is not None and tv is not None else None
        runs[name] = r
    return runs


# ----------------------------------------------------------------------------- helpers
def T(r, names):
    t = r["timers"]
    v = [t[n][2] for n in names if n in t]
    return sum(v) if v else None


def phase_times(r):
    out = []
    for lab, names in PHASES:
        if names is None:
            e = T(r, ["setup.eigensolver"])
            sub = sum(x for x in (T(r, ["setup.eigensolver.factorization"]), T(r, ["setup.eigensolver.arnoldi"])) if x)
            out.append(max(e - sub, 0.0) if e is not None else None)
        else:
            out.append(T(r, names))
    return out


def ft(x):
    if x is None:
        return ""
    return "%.0f" % x if x >= 100 else ("%.1f" % x if x >= 10 else "%.2f" % x)


def its(r):
    """Iterations as printed in the tables (n.c. = not converged, * = spurious convergence)."""
    if r is None:
        return ""
    if r["status"] in ("timeout", "error", "killed"):
        return {"timeout": "T/O", "error": "ERR", "killed": "killed"}[r["status"]]
    s = str(r.get("its", "?"))
    if r["status"] == "not_converged":
        return "n.c. (%s)" % s
    tr = r.get("true_rel_res") or r.get("true_res")
    if tr is not None and tr > 2e-8 and r["name"].startswith(("tab1", "fig2")):
        s += "*"
    return s


def table(head, rows, align=None):
    align = align or ["l"] + ["r"] * (len(head) - 1)
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(":--" if a == "l" else "--:" for a in align) + "|"]
    out += ["| " + " | ".join("" if c is None else str(c) for c in row) + " |" for row in rows]
    return "\n".join(out) + "\n\n"


def fig(ncols=1, w=6.4, h=3.8):
    f, axes = plt.subplots(1, ncols, figsize=(w * ncols, h), squeeze=False)
    f.patch.set_facecolor(SURF)
    for ax in axes[0]:
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    return f, axes[0]


def finish(ax, title, xl, yl, legend=True):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xl, color=INK2, fontsize=9)
    ax.set_ylabel(yl, color=INK2, fontsize=9)
    if legend:
        lg = ax.legend(frameon=False, fontsize=8)
        for t in lg.get_texts():
            t.set_color(INK)


def save(f, name):
    os.makedirs(os.path.join(RUN, "plots"), exist_ok=True)
    f.tight_layout()
    f.savefig(os.path.join(RUN, "plots", name), dpi=150, facecolor=SURF)
    plt.close(f)
    return "![%s](plots/%s)\n\n" % (name, name)


def stacked(runs, labels, name, title):
    """Stacked bars of the phases (max over ranks) of the given runs."""
    if not HAVE_MPL or not runs:
        return ""
    f, (ax,) = fig(1, w=max(6.4, 0.9 * len(runs) + 2.5), h=4.0)
    x = list(range(len(runs)))
    bottom = [0.0] * len(runs)
    for k, (lab, _) in enumerate(PHASES):
        vals = [phase_times(r)[k] or 0.0 for r in runs]
        if not any(vals):
            continue
        ax.bar(x, vals, bottom=bottom, width=0.62, color=COLORS[k], label=lab, edgecolor=SURF, linewidth=1.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30 if len(runs) > 6 else 0, ha="right" if len(runs) > 6 else "center")
    ax.xaxis.grid(False)
    finish(ax, title, "", "time [s], max over ranks")
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    return save(f, name)


# ----------------------------------------------------------------------------- sections
def sec_status(runs):
    out = ["## Runs\n\n"]
    for t in ("test_subdomain", "test_preconditioner"):
        p = os.path.join(RUN, "log." + t)
        if os.path.isfile(p):
            with open(p, errors="replace") as f:
                lines = f.read().splitlines()
            fails = [l.strip() for l in lines if "[FAIL]" in l]
            out.append("- `%s` (16 processes): **%s**%s\n" % (t, lines[-1] if lines else "?",
                       "" if not fails else " - " + "; ".join(fails)))
    for p in ("log.git", "log.modules"):
        if os.path.isfile(os.path.join(RUN, p)):
            with open(os.path.join(RUN, p), errors="replace") as f:
                txt = " ".join(f.read().split())
            out.append("- %s: `%s`\n" % (p[4:], txt[:400]))
    rows = []
    for r in runs.values():
        rows.append((r["name"], r["np"], r["status"], its(r), "%.1e" % r["true_rel_res"] if r.get("true_rel_res") else "",
                     r.get("n0", ""), ft(r.get("t_setup")), ft(r.get("t_solve")), "" if r["wall"] is None else r["wall"], r.get("error", "")[:80]))
    out.append("\n" + table(["run", "np", "status", "its", "true res.", "n0", "setup [s]", "solve [s]", "wall [s]", "error"], rows,
                            ["l", "r", "l", "r", "r", "r", "r", "r", "r", "l"]))
    out.append("`*` = GMRES reports convergence but the recomputed true residual is above 2 rtol; "
               "n.c. = not converged within maxit; T/O = stopped by RUN_TIMEOUT; killed = the job ended during the run.\n\n")
    return "".join(out)


def sec_tab1(runs):
    rows = []
    for m, p in PAPER_T1.items():
        r1, r2 = runs.get("tab1_%s_RAS" % m), runs.get("tab1_%s_2L" % m)
        if not r1 and not r2:
            continue
        r = r2 or r1
        N = r["np"] or 0
        n0 = r2.get("n0") if r2 else None
        n = r.get("n")
        nlab = "" if n is None else (str(n) if n == p[0] else "%s (paper %s)" % (n, p[0]))
        rows.append((m, nlab, N, its(r1), its(r2), n0 if n0 is not None else "", "%.0f" % (n0 / N) if n0 and N else "",
                     p[5], p[6], "%.0f" % (p[6] / PAPER_N_T1), "%s/%s/%s" % (p[2] or ">100", p[3] or ">100", p[4] or ">100"),
                     ft(r2.get("t_setup")) if r2 else "", ft(r2.get("t_solve")) if r2 else ""))
    if not rows:
        return ""
    out = ["## Table 1 - SuiteSparse matrices (tau = 0.6, nev = 300)\n\n",
           "Ours: N = %s subdomains, RAS = one-level, 2L = RAS + deflated coarse correction (2.4). Paper: N = 256; "
           "AMG columns = AGMG / BoomerAMG / GAMG (F = failure, C = complex). Compare n0/N, since N differs. "
           "Note: in the paper the G3_circuit row (8 it., n0 = 71,385) matches tau = 0.9 of Fig. 2, not tau = 0.6 "
           "(11 it., n0 = 34,402).\n\n" % rows[0][2]]
    out.append(table(["matrix", "n", "N", "RAS its", "2L its", "n0", "n0/N", "paper its", "paper n0", "paper n0/N",
                      "paper AMG its", "2L setup [s]", "2L solve [s]"], rows))
    two = [runs["tab1_%s_2L" % m] for m in PAPER_T1 if "tab1_%s_2L" % m in runs
           and runs["tab1_%s_2L" % m]["timers"]]
    out.append(stacked(two, [r["name"][5:-3] for r in two], "tab1_phases.png", "Table 1, two-level: time per phase"))
    return "".join(out)


def sec_fig2(runs):
    pts = []
    for tau in PAPER_F2:
        r = runs.get("fig2_tau%s" % tau) or (runs.get("tab1_G3_circuit_2L") if tau == 0.6 else None)
        if r:
            pts.append((tau, r))
    asm = runs.get("fig2_ASM")
    if not pts and not asm:
        return ""
    out = ["## Fig. 2 - G3_circuit, influence of tau (nev = 300)\n\n"]
    rows = [(t, r.get("n0", ""), its(r), PAPER_F2[t][0], PAPER_F2[t][1], ft(r.get("t_setup")), ft(r.get("t_solve")))
            for t, r in pts]
    if asm:
        rows.append(("one-level ASM", "-", its(asm), "-", "no conv.", ft(asm.get("t_setup")), ft(asm.get("t_solve"))))
    out.append(table(["tau", "n0", "its", "paper n0", "paper its", "setup [s]", "solve [s]"], rows))
    if HAVE_MPL:
        f, (ax,) = fig(1, w=6.8, h=4.2)
        for k, (t, r) in enumerate(pts):
            if r["hist"]:
                ax.plot(range(len(r["hist"])), r["hist"], color=SEQ[k % len(SEQ)], marker=MARK[k % len(MARK)],
                        markersize=4, markevery=max(1, len(r["hist"]) // 12), linewidth=1.6, label="tau = %s" % t)
        if asm and asm["hist"]:
            ax.plot(range(len(asm["hist"])), asm["hist"], color=COLORS[1], linewidth=1.6, linestyle="--",
                    label="one-level ASM")
        ax.axhline(1e-8, color=IDEAL, linewidth=1, linestyle=":")
        ax.set_yscale("log")
        finish(ax, "G3_circuit: GMRES residual history", "iteration", "||r|| / ||b||")
        out.append(save(f, "fig2_tau.png"))
    return "".join(out)


def sec_fig4(runs):
    rr = sorted([r for r in runs.values() if r["name"].startswith("fig4_") and r.get("total") is not None],
                key=lambda r: r["np"])
    if not rr:
        return ""
    b = rr[0]
    out = ["## Fig. 4 - strong scaling, 3D convection-diffusion, nu = 1e-2 (tau = 0.3, nev = 60)\n\n",
           "Ours: %s. Paper: n = 8.1e6 (P2 elements), iterations %s for N = %s. Speedup and efficiency relative "
           "to N = %d.\n\n" % (b.get("desc", ""), ", ".join(str(v) for v in PAPER_F4.values()),
                               ", ".join(str(k) for k in PAPER_F4), b["np"])]
    rows = []
    for r in rr:
        sp = b["total"] / r["total"] * b["np"]
        rows.append((r["np"], "%.0f" % r["nI"][1] if r.get("nI") else "", r.get("n0", ""), its(r),
                     PAPER_F4.get(r["np"], ""), ft(r["t_setup"]), ft(r["t_solve"]), ft(r["total"]),
                     "%.1f" % sp, "%.2f" % (sp / r["np"])))
    out.append(table(["N", "n_I avg", "n0", "its", "paper its", "setup [s]", "solve [s]", "total [s]", "speedup",
                      "efficiency"], rows))
    if HAVE_MPL:
        f, (ax,) = fig(1, w=6.4, h=4.0)
        x = list(range(len(rr)))
        ax.bar(x, [r["t_setup"] for r in rr], width=0.5, color=COLORS[0], label="setup", edgecolor=SURF, linewidth=1.5)
        ax.bar(x, [r["t_solve"] for r in rr], width=0.5, bottom=[r["t_setup"] for r in rr], color=COLORS[1],
               label="solve", edgecolor=SURF, linewidth=1.5)
        ax.plot(x, [b["total"] * b["np"] / r["np"] for r in rr], color=IDEAL, linestyle="--", linewidth=1.2, label="ideal")
        for xi, r in zip(x, rr):
            ax.annotate("(%s)" % r.get("its", "?"), (xi, r["total"]), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8, color=INK2)
        ax.set_yscale("log")
        plain = FuncFormatter(lambda v, _: "%g" % v)
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(plain)
        ax.tick_params(axis="y", which="minor", labelsize=7, colors=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(r["np"]) for r in rr])
        ax.xaxis.grid(False)
        finish(ax, "Strong scaling (iterations in parentheses)", "number of subdomains", "time to solution [s]")
        out.append(save(f, "fig4_strong.png"))
    out.append(stacked(rr, ["N=%d" % r["np"] for r in rr], "fig4_phases.png", "Strong scaling: time per phase"))
    return "".join(out)


def sec_tab2(runs):
    data = {}
    for r in runs.values():
        m = re.match(r"tab2_(2d|3d)_nu([^_]+)_(2L|RAS)$", r["name"])
        if m:
            data[(m.group(1), float(m.group(2)), m.group(3))] = r
    if not data:
        return ""
    out = ["## Table 2 - convection-diffusion (4.1), nu = 1 ... 1e-4 (tau = 0.3, nev = 60)\n\n",
           "Ours: finite volumes with upwind, N = %s, ~6k unknowns per subdomain in 2D and ~2k in 3D. Paper: SUPG "
           "finite elements, same subdomain sizes, N = 1024 (2D) and 4096 (3D). Compare iterations and n0/N.\n\n"
           % next(iter(data.values()))["np"]]
    rows = []
    for dim in ("2d", "3d"):
        Np, n_p, pv = PAPER_T2[dim]
        for nu in sorted(pv, reverse=True):
            r2, r1 = data.get((dim, nu, "2L")), data.get((dim, nu, "RAS"))
            N = (r2 or r1 or {}).get("np") or 1
            n0 = r2.get("n0") if r2 else None
            rows.append((dim.upper(), "%g" % nu, its(r1), its(r2), n0 if n0 is not None else "",
                         "%.0f" % (n0 / N) if n0 else "", pv[nu][0], "%.0f" % (pv[nu][1] / Np),
                         ft(r2.get("t_setup")) if r2 else "", ft(r2.get("t_solve")) if r2 else ""))
    out.append(table(["dim", "nu", "RAS its", "2L its", "n0", "n0/N", "paper its", "paper n0/N", "2L setup [s]",
                      "2L solve [s]"], rows))
    if HAVE_MPL:
        f, axes = fig(2, w=4.8, h=3.8)
        for ax, dim in zip(axes, ("2d", "3d")):
            nus = sorted(PAPER_T2[dim][2])
            for k, (lab, key) in enumerate((("RAS (ours)", "RAS"), ("two-level (ours)", "2L"))):
                ys = [data[(dim, nu, key)].get("its") if (dim, nu, key) in data
                      and data[(dim, nu, key)]["status"] == "ok" else None for nu in nus]
                pts = [(a, b) for a, b in zip(nus, ys) if b is not None]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORS[k], marker=MARK[k], linewidth=1.6,
                            label=lab)
            ax.plot(nus, [PAPER_T2[dim][2][nu][0] for nu in nus], color=COLORS[2], marker=MARK[2], linewidth=1.6,
                    linestyle="--", label="two-level (paper)")
            ax.set_xscale("log")
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            finish(ax, dim.upper(), "viscosity nu", "GMRES iterations")
        out.append(save(f, "tab2_nu.png"))
    return "".join(out)


def sec_phases(runs):
    out = ["## Time per phase (max over ranks, seconds)\n\n",
           "Same timers as the local runs; `phases.csv` has min / avg / max of every timer, the logs the full table. "
           "decomposition = rank 0 reads/generates, partitions and scatters the matrix (OpenFOAM: decomposePar).\n\n"]
    rows = []
    for r in runs.values():
        if not r["timers"]:
            continue
        rows.append([r["name"], r["np"], ft(T(r, ["decomposition"]))] + [ft(v) for v in phase_times(r)] +
                    [ft(r.get("t_setup")), ft(r.get("t_solve")), ft(r.get("total"))])
    out.append(table(["run", "np", "decomp."] + [p[0] for p in PHASES[:-1]] + ["GMRES", "setup", "solve", "total"], rows))
    return "".join(out)


def write_csvs(runs):
    with open(os.path.join(RUN, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "np", "status", "n", "nnz", "iterations", "converged", "true_rel_res", "n0", "m_min", "m_avg",
                    "m_max", "kernel_subdomains", "t_setup", "t_solve", "total", "wall"] + [p[0] for p in PHASES])
        for r in runs.values():
            m = r.get("m", ("", "", ""))
            w.writerow([r["name"], r["np"], r["status"], r.get("n", ""), r.get("nnz", ""), r.get("its", ""),
                        r.get("converged", ""), r.get("true_rel_res", ""), r.get("n0", ""), m[0], m[1], m[2],
                        r.get("ker", ""), r.get("t_setup", ""), r.get("t_solve", ""), r.get("total", ""),
                        r["wall"] or ""] + [v if v is not None else "" for v in phase_times(r)])
    with open(os.path.join(RUN, "phases.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "np", "timer", "min", "avg", "max"])
        for r in runs.values():
            for k, v in r["timers"].items():
                w.writerow([r["name"], r["np"], k] + list(v))


def main():
    global RUN
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        print(__doc__)
        sys.exit(1)
    RUN = os.path.abspath(sys.argv[1])
    runs = load()
    write_csvs(runs)
    parts = ["# schwarz2lvl vs Al Daas, Jolivet, Rees (SISC 2023)\n\n",
             "Run directory `%s`.%s\n\n" % (os.path.basename(RUN), "" if HAVE_MPL else
             " matplotlib not available: tables only (`python3 -m pip install --user matplotlib`)."),
             sec_status(runs), sec_tab1(runs), sec_fig2(runs), sec_fig4(runs), sec_tab2(runs), sec_phases(runs)]
    with open(os.path.join(RUN, "report.md"), "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print("written %s/report.md (%d runs, plots: %s)" % (RUN, len(runs), "yes" if HAVE_MPL else "no"))


if __name__ == "__main__":
    main()
