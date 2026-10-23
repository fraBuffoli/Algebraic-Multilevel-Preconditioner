#!/usr/bin/env python3
"""Compare two systems written in Matrix Market format (PREFIX.mtx + PREFIX_rhs.mtx).

    python3 compare_mtx.py PREFIX_A PREFIX_B [tolerance]

Used by Allrun_gamg.sh: OpenFOAM (bubblyPressureFoam -writeMatrix) against the schwarz2lvl
generator (matgen bubbles3d): same pattern, values equal up to round-off. The last line is
"OK: ..." or "DIFFERENT: ..." (exit code 0 / 1). Standard library only.
"""
import sys


def read_matrix(path):
    entries = {}
    with open(path) as f:
        header = f.readline()
        if "coordinate" not in header:
            raise ValueError(path + ": not a coordinate Matrix Market file")
        symmetric = "symmetric" in header
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        m, n, _ = (int(x) for x in line.split())
        for line in f:
            if not line.strip():
                continue
            i, j, v = line.split()
            i, j, v = int(i), int(j), float(v)
            entries[(i, j)] = entries.get((i, j), 0.0) + v
            if symmetric and i != j:
                entries[(j, i)] = entries.get((j, i), 0.0) + v
    return m, n, entries


def read_vector(path):
    with open(path) as f:
        f.readline()
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        n = int(line.split()[0])
        v = [float(x) for x in f.read().split()]
    if len(v) != n:
        raise ValueError(path + ": %d values, %d expected" % (len(v), n))
    return v


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    a, b = sys.argv[1], sys.argv[2]
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-12
    ma, na, ea = read_matrix(a + ".mtx")
    mb, nb, eb = read_matrix(b + ".mtx")
    # entries below 1e-300 (exact zeros stored by one side only) do not count as pattern differences
    keys_a = {k for k, v in ea.items() if abs(v) > 1e-300}
    keys_b = {k for k, v in eb.items() if abs(v) > 1e-300}
    scale = max(abs(v) for v in ea.values())
    diff = max(abs(ea.get(k, 0.0) - eb.get(k, 0.0)) for k in keys_a | keys_b) / scale
    ra, rb = read_vector(a + "_rhs.mtx"), read_vector(b + "_rhs.mtx")
    rscale = max(max(abs(x) for x in ra), 1e-300)
    rdiff = max(abs(x - y) for x, y in zip(ra, rb)) / rscale if len(ra) == len(rb) else float("inf")
    print("%s: %dx%d, %d nonzeros; %s: %dx%d, %d nonzeros" % (a, ma, na, len(keys_a), b, mb, nb, len(keys_b)))
    print("pattern: %d entries only in the first, %d only in the second" % (len(keys_a - keys_b), len(keys_b - keys_a)))
    ok = (ma, na) == (mb, nb) and keys_a == keys_b and diff <= tol and rdiff <= tol
    print("%s: matrices equal to %.1e, right-hand sides to %.1e (relative to the largest entry; tolerance %.0e)"
          % ("OK" if ok else "DIFFERENT", diff, rdiff, tol))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
