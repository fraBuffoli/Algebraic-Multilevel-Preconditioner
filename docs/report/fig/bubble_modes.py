#!/usr/bin/env python3
"""Figure 1 of the report (fig/bubble_modes.pdf): near-kernel modes of the bubbly pressure matrix.

Usage (from the report folder, with the matgen binary of the repository):
    python3 fig/bubble_modes.py /path/to/Algebraic-Multilevel-Preconditioner/bin/matgen

Writes two 2D bubble matrices with matgen (128 x 128 cells, 5 bubbles of radius 0.07, seed 2,
density ratio 1000 and 1), computes the 10 smallest eigenpairs of A u = lambda diag(A) u with SciPy
(shift-invert Lanczos) and plots them. Requires numpy, scipy, matplotlib.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = 128  # cells per direction
GEN = ["bubbles2d", str(M), str(M), "nb=5", "radius=0.07", "seed=2"]


def smallest(A, k=10):
    """k smallest eigenpairs of A u = lambda diag(A) u (A symmetric positive definite)."""
    vals, vecs = sla.eigsh(A, k=k, M=sp.diags(A.diagonal()), sigma=0, which="LM")
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    matgen = os.path.abspath(sys.argv[1])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bubble_modes.pdf")
    with tempfile.TemporaryDirectory() as tmp:
        for name, rho in (("bub", "1000"), ("hom", "1")):
            subprocess.run([matgen] + GEN + ["rho=" + rho, "out=" + name], cwd=tmp, check=True)
        A = sio.mmread(os.path.join(tmp, "bub.mtx")).tocsc()
        H = sio.mmread(os.path.join(tmp, "hom.mtx")).tocsc()
        b = sio.mmread(os.path.join(tmp, "bub_rhs.mtx")).ravel()
    alpha = b * M * M  # the generated right-hand side is (cell volume) x alpha
    vals, vecs = smallest(A)
    hv, _ = smallest(H)
    print("density ratio 1000:", " ".join("%.3e" % v for v in vals))
    print("density ratio 1:   ", " ".join("%.3e" % v for v in hv))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8,
                         "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(6.6, 3.45))
    gs = fig.add_gridspec(2, 4, hspace=0.40, wspace=0.30, left=0.015, right=0.99, top=0.93, bottom=0.03)
    X = np.linspace(0, 1, M)

    def field(ax, f, title, cmap, sym):
        if sym:
            v = abs(f).max()
            ax.imshow(f.reshape(M, M), origin="lower", cmap=cmap, vmin=-v, vmax=v, extent=[0, 1, 0, 1])
        else:
            ax.imshow(f.reshape(M, M), origin="lower", cmap=cmap, vmin=0, vmax=1, extent=[0, 1, 0, 1])
        ax.contour(X, X, alpha.reshape(M, M), levels=[0.5], colors="k", linewidths=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

    field(fig.add_subplot(gs[0, 0]), alpha, r"gas fraction $\alpha$", "Blues", False)
    ax = fig.add_subplot(gs[0, 1:3])
    k = np.arange(1, 11)
    ax.semilogy(k, vals, "o-", color="#2a6fdb", ms=3.5, lw=1, label=r"density ratio $10^3$")
    ax.semilogy(k, hv, "s--", color="#e8642c", ms=3, lw=1, label="density ratio 1")
    ax.axvline(5.5, color="0.6", lw=0.6, ls=":")
    ax.set_xticks(k)
    ax.set_xlabel(r"$k$", labelpad=1)
    ax.set_title(r"10 smallest eigenvalues $\lambda_k$ of $A\,u=\lambda\,\mathrm{diag}(A)\,u$")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3, lw=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    field(fig.add_subplot(gs[0, 3]), vecs[:, 5], r"$u_6$  ($\lambda_6$ = %.1e)" % vals[5], "RdBu_r", True)
    for i in range(4):
        field(fig.add_subplot(gs[1, i]), vecs[:, i],
              r"$u_{%d}$  ($\lambda_{%d}$ = %.1e)" % (i + 1, i + 1, vals[i]), "RdBu_r", True)
    fig.savefig(out)
    print("written", out)


if __name__ == "__main__":
    main()
