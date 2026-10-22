#!/usr/bin/env python3
"""Summary of the P2 campaign (Allrun_p2.sh): symmetric factorizations and single precision.

    python3 report_p2.py run_p2_<jobid>

Reads <run>/csv/*.csv (time loops, one line per solve), <run>/single/*.csv (single solves),
<run>/log.* (timers and factorizations actually used) and writes <run>/report.md.

Runs are grouped by problem: <group>_<variant>, variant in lu, chol, cholsp, chol_cp,
chol_root (bub128_lu, bub128_chol -> group bub128). Every run is compared with the `lu` run
of its group (the P1 configuration):
  * cost/solve = (initial setup + updates + solves) / solves (wall times, as in P1);
  * cost of each operation, from the timers (average over the ranks) divided by the number
    of times the operation was done: this separates the speed of the factorizations from
    the rebuild decisions of --reuse auto, which can differ between runs.
"""
import csv
import glob
import os
import re
import sys
from collections import OrderedDict

VARIANTS = ("lu", "chol", "cholsp", "chol_cp", "chol_root")
NAME_RE = re.compile(r"^(.*?)_(%s)$" % "|".join(VARIANTS))
TIMER_RE = re.compile(r"^  ([A-Za-z0-9_.~]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s*$")


def split_name(name):
    m = NAME_RE.match(name)
    return (m.group(1), m.group(2)) if m else (name, "")


def load_loop(path):
    rows = list(csv.DictReader(open(path, newline="")))
    for r in rows:
        for k in ("t_update", "t_solve", "true_rel_res"):
            r[k] = float(r.get(k) or 0.0)
        for k in ("step", "corrector", "iterations", "converged", "n0", "np"):
            r[k] = int(float(r.get(k) or 0))
    return rows


def parse_log(path):
    """Timers (avg over ranks) and the factorizations reported by the last printStats()."""
    info = {"timers": {}, "aii": "", "atilde": "", "a00": "", "fallbacks": ""}
    if not os.path.exists(path):
        return info
    in_timers = False
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        if line.startswith("Timings [s] over"):
            in_timers = True
            continue
        if in_timers:
            m = TIMER_RE.match(line)
            if m:
                info["timers"][m.group(1)] = float(m.group(3))
                continue
            if line.strip() and not line.startswith("  phase"):
                in_timers = False
        s = line.strip()
        if s.startswith("A_ii factorization"):
            info["aii"] = s[len("A_ii factorization"):].split("(rank 0")[0].strip()
            info["aii_fb"] = s.split("chain fallbacks on")[-1].strip(" )") if "chain fallbacks" in s else ""
        elif s.startswith("A~_ii factorization"):
            info["atilde"] = s[len("A~_ii factorization"):].split("(rank 0")[0].strip()
        elif s.startswith("coarse solver") and ("(" in s):  # stats line, not the configuration
            info["a00"] = s[len("coarse solver"):].strip()
        elif s.startswith("Local solver (rank 0):"):
            info["aii"] = s.split(":", 1)[1].strip()
        elif s.startswith("Refactorizations falling back"):
            info["fallbacks"] = s.split(":", 1)[1].strip()
    return info


def short_a00(s):
    """'cpardiso(48 procs, mtype 2 cholesky, nnz(L) ...)' -> 'cpardiso mtype 2 cholesky'."""
    if s.startswith("root("):
        return "root " + s[5:-1]
    m = re.match(r"cpardiso\((\d+) procs, (mtype -?\d+ \w+)( \(fallback from cholesky\))?", s)
    if m:
        return "cpardiso %s%s" % (m.group(2), " (fallback)" if m.group(3) else "")
    return s


def summarize_loop(rows, tm):
    first, upd = rows[0], rows[1:]
    n = len(rows)
    s = OrderedDict()
    s["np"] = first["np"]
    s["solves"] = n
    s["converged"] = sum(r["converged"] for r in rows)
    s["max_res"] = max(r["true_rel_res"] for r in rows)
    s["refactorize"] = sum(1 for r in upd if r["level"] == "refactorize")
    s["rebuild"] = sum(1 for r in upd if r["level"] == "rebuild")
    s["t_init"] = first["t_update"]
    t_ref = sum(r["t_update"] for r in upd if r["level"] == "refactorize")
    t_reb = sum(r["t_update"] for r in upd if r["level"] == "rebuild")
    s["mean_ref"] = t_ref / s["refactorize"] if s["refactorize"] else 0.0
    s["mean_reb"] = t_reb / s["rebuild"] if s["rebuild"] else 0.0
    s["t_solve"] = sum(r["t_solve"] for r in rows)
    s["its_total"] = sum(r["iterations"] for r in rows)
    s["its_avg"] = s["its_total"] / n
    s["its_max"] = max(r["iterations"] for r in rows)
    s["ms_per_it"] = 1e3 * s["t_solve"] / max(1, s["its_total"])
    s["t_total"] = s["t_init"] + sum(r["t_update"] for r in upd) + s["t_solve"]
    s["per_solve"] = s["t_total"] / n
    s["n0"] = first["n0"]
    setups = 1 + s["rebuild"]
    refs = max(1, s["refactorize"])
    its = max(1, s["its_total"])
    g = lambda k: tm.get(k, float("nan"))
    s["ops"] = OrderedDict([
        ("A_ii factorization [s]", g("setup.local_factorization") / setups),
        ("A_ii refactorization [s]", g("update.local_refactorization") / refs),
        ("A~_ii factorization [s]", g("setup.eigensolver.factorization") / setups),
        ("eigensolver [s]", g("setup.eigensolver") / setups),
        ("A_00 factorization [s]", g("setup.coarse_factorization") / setups),
        ("A_00 refactorization [s]", g("update.coarse_factorization") / refs),
        ("A_ii solve [ms/it]", 1e3 * g("solve.preconditioner.local.lu") / its),
        ("A_00 solve [ms/it]", 1e3 * g("solve.preconditioner.coarse.solve") / its),
    ])
    return s


def rel(v, ref):
    if ref is None or ref != ref or v != v or ref == 0:
        return "-"
    return "%+.0f%%" % (100.0 * (v / ref - 1.0))


def fmt(v, d=3):
    return "-" if v != v else ("%.*f" % (d, v))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run = sys.argv[1].rstrip("/")
    out = []
    w = out.append
    w("# P2 - symmetric factorizations and single precision: %s\n" % os.path.basename(run))
    for f in ("config.txt", "log.git"):
        p = os.path.join(run, f)
        if os.path.exists(p):
            w("`" + " ".join(open(p).read().split("\n")).strip() + "`\n")
    st = os.path.join(run, "status.txt")
    if os.path.exists(st):
        bad = [l.split() for l in open(st) if l.split() and l.split()[2] != "ok"]
        w("Runs not ok: " + (", ".join("%s (%s)" % (b[0], b[2]) for b in bad) if bad else "none") + "\n")
    tests = []
    for t in ("test_subdomain", "test_preconditioner", "test_reuse", "test_factorization"):
        p = os.path.join(run, "log." + t)
        if os.path.exists(p):
            lines = [l.strip() for l in open(p, errors="replace") if l.strip()]
            tests.append("%s: %s" % (t, lines[-1] if lines else "empty"))
    if tests:
        w("Unit tests: " + "; ".join(tests) + "\n")

    # ------------------------------------------------------------------ time loops
    loops = OrderedDict()
    for f in sorted(glob.glob(os.path.join(run, "csv", "*.csv"))):
        name = os.path.splitext(os.path.basename(f))[0]
        rows = load_loop(f)
        if not rows:
            continue
        info = parse_log(os.path.join(run, "log." + name))
        loops[name] = (summarize_loop(rows, info["timers"]), info)
    groups = OrderedDict()
    for name in loops:
        g, v = split_name(name)
        groups.setdefault(g, OrderedDict())[v] = name
    order = lambda d: sorted(d.items(), key=lambda kv: VARIANTS.index(kv[0]) if kv[0] in VARIANTS else 99)

    if loops:
        w("## Time loops (--reuse auto)\n")
        w("Factorizations as reported by the run (rank 0). **cost/solve** = (initial setup + updates + "
          "solves) / solves; **vs lu** is relative to the `lu` run of the group (P1 configuration).\n")
        w("| run | A_ii | A~_ii | A_00 | solves (conv.) | its avg / max | refact. / rebuilds | initial setup [s] "
          "| mean refact. [s] | mean rebuild [s] | solve [ms/it] | cost/solve [s] | vs lu |")
        w("|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for g, vs in groups.items():
            ref = loops[vs["lu"]][0] if "lu" in vs else None
            for v, name in order(vs):
                s, info = loops[name]
                w("| %s | %s | %s | %s | %d (%d) | %.1f / %d | %d / %d | %s | %s | %s | %s | %s | %s |" % (
                    name, info["aii"], info["atilde"], short_a00(info["a00"]), s["solves"], s["converged"],
                    s["its_avg"], s["its_max"], s["refactorize"], s["rebuild"], fmt(s["t_init"]),
                    fmt(s["mean_ref"]), fmt(s["mean_reb"]), fmt(s["ms_per_it"], 2), fmt(s["per_solve"]),
                    rel(s["per_solve"], ref["per_solve"] if ref else None) if v != "lu" else "ref"))
        w("")
        w("## Cost of each operation\n")
        w("Timers averaged over the ranks, divided by the number of operations: factorizations per "
          "setup (initial + rebuilds), refactorizations per refactorize step, solves per GMRES "
          "iteration (one preconditioner application each). In brackets: change vs `lu`.\n")
        for g, vs in groups.items():
            names = [n for _, n in order(vs)]
            ref = loops[vs["lu"]][0]["ops"] if "lu" in vs else None
            w("**%s**\n" % g)
            w("| operation | " + " | ".join(names) + " |")
            w("|:--|" + "--:|" * len(names))
            keys = list(loops[names[0]][0]["ops"].keys())
            for k in keys:
                cells = []
                for n in names:
                    v = loops[n][0]["ops"][k]
                    d = 2 if "ms" in k else 3
                    r = rel(v, ref[k]) if ref and n != vs.get("lu") else ""
                    cells.append(fmt(v, d) + (" (%s)" % r if r and r != "-" else ""))
                w("| %s | %s |" % (k, " | ".join(cells)))
            w("")
        w("Correctness: max true relative residual over all solves, and refactorizations that fell back "
          "to a full factorization.\n")
        for name, (s, info) in loops.items():
            w("- `%s`: max true residual %.2e; %s" % (name, s["max_res"], info["fallbacks"] or "-"))
        w("")
        w("Iterations of the first corrector of every step (`R` = rebuild of the coarse basis):\n")
        for f in sorted(glob.glob(os.path.join(run, "csv", "*.csv"))):
            name = os.path.splitext(os.path.basename(f))[0]
            rows = load_loop(f)
            seq = ["%d%s" % (r["iterations"], "R" if r["level"] == "rebuild" and r["step"] > 0 else "")
                   for r in rows if r["corrector"] == 0]
            w("- `%s`: %s" % (name, " ".join(seq)))
        w("")

    # ------------------------------------------------------------------ single solves
    singles = OrderedDict()
    for f in sorted(glob.glob(os.path.join(run, "single", "*.csv"))):
        name = os.path.splitext(os.path.basename(f))[0]
        rows = list(csv.DictReader(open(f, newline="")))
        if not rows:
            continue
        r = rows[-1]
        info = parse_log(os.path.join(run, "log." + name))
        singles[name] = (r, info)
    if singles:
        w("## Single solves (setup + GMRES)\n")
        w("| run | A_ii | A~_ii | A_00 | n0 | its | setup [s] | eigensolver [s] | A_00 fact. [s] | solve [s] | total [s] | vs lu |")
        w("|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
        sg = OrderedDict()
        for name in singles:
            g, v = split_name(name)
            sg.setdefault(g, OrderedDict())[v] = name
        for g, vs in sg.items():
            ref = None
            if "lu" in vs:
                rr = singles[vs["lu"]][0]
                ref = float(rr["t_setup"]) + float(rr["t_solve"])
            for v, name in order(vs):
                r, info = singles[name]
                tot = float(r["t_setup"]) + float(r["t_solve"])
                w("| %s | %s | %s | %s | %s | %s%s | %s | %s | %s | %s | %s | %s |" % (
                    name, info["aii"], info["atilde"], short_a00(info["a00"]), r["n0"], r["iterations"],
                    "" if r["converged"] == "1" else " (NOT conv.)", fmt(float(r["t_setup"])),
                    fmt(float(r["t_eig"])), fmt(float(r["t_coarse_fact"])), fmt(float(r["t_solve"])), fmt(tot),
                    rel(tot, ref) if v != "lu" else "ref"))
        w("")

    path = os.path.join(run, "report.md")
    open(path, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("\nwritten: %s" % path)


if __name__ == "__main__":
    main()
