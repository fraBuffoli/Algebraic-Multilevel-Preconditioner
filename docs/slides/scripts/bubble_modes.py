#!/usr/bin/env python3
"""Slide version of Figure 1 of the report (fig/bubble_modes.pdf): near-kernel modes of the bubbly
pressure matrix.

Same matrices, eigenpairs and colours as the report script fig/bubble_modes.py. Differences: the
five bubble modes are highlighted in the spectrum (shaded band + label), the gas fraction is written
to a separate file (fig/bubble_alpha.pdf, shown on the previous slide) and the spectrum takes its
place, the figures are drawn at the size they have on the slides, the panel titles are shorter and
the sign of every eigenvector is fixed (largest entry negative, as in the report figure).

Usage (from the slides folder, with the matgen binary of the repository):
    python3 scripts/bubble_modes.py /path/to/Algebraic-Multilevel-Preconditioner/bin/matgen

Writes fig/bubble_modes.pdf and fig/bubble_alpha.pdf. Requires numpy, scipy, matplotlib.
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
GEN = ["bubbles2d", str(M), str(M), "nb=5", "radius=0.07", "seed=2"]  # as in the report
WIDTH_IN = 3.86  # width on the slide (0.7 \textwidth)
ALPHA_IN = 1.75  # gas-fraction panel (fig/bubble_alpha.pdf)


def smallest(A, k=10):
    """k smallest eigenpairs of A u = lambda diag(A) u (A symmetric positive definite)."""
    vals, vecs = sla.eigsh(A, k=k, M=sp.diags(A.diagonal()), sigma=0, which="LM",
                           v0=np.ones(A.shape[0]))  # deterministic start
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    # the sign of an eigenvector is arbitrary: largest entry made negative, the colours of the report figure
    vecs *= -np.sign(vecs[np.abs(vecs).argmax(axis=0), np.arange(k)])
    return vals, vecs


def sci(x):
    m, e = ("%.1e" % x).split("e")
    return r"%s\cdot10^{%d}" % (m, int(e))


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    matgen = os.path.abspath(sys.argv[1])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fig", "bubble_modes.pdf")
    with tempfile.TemporaryDirectory() as tmp:
        for name, rho in (("bub", "1000"), ("hom", "1")):
            subprocess.run([matgen] + GEN + ["rho=" + rho, "out=" + name], cwd=tmp, check=True,
                           stdout=subprocess.DEVNULL)
        A = sio.mmread(os.path.join(tmp, "bub.mtx")).tocsc()
        H = sio.mmread(os.path.join(tmp, "hom.mtx")).tocsc()
        b = np.asarray(sio.mmread(os.path.join(tmp, "bub_rhs.mtx"))).ravel()
    alpha = b * M * M  # the generated right-hand side is (cell volume) x alpha
    vals, vecs = smallest(A)
    hv, _ = smallest(H)
    print("density ratio 1000:", " ".join("%.3e" % v for v in vals))
    print("density ratio 1:   ", " ".join("%.3e" % v for v in hv))
    print("lambda_6 / lambda_k, k=1..5:", " ".join("%.0f" % (vals[5] / v) for v in vals[:5]))

    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 6.5,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "legend.fontsize": 6})
    fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * 3.45 / 6.6))  # aspect of the report figure
    gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.30, left=0.015, right=0.99, top=0.93, bottom=0.03)
    X = np.linspace(0, 1, M)

    def field(ax, f, title, cmap, sym):
        if sym:
            v = abs(f).max()
            ax.imshow(f.reshape(M, M), origin="lower", cmap=cmap, vmin=-v, vmax=v, extent=[0, 1, 0, 1])
        else:
            ax.imshow(f.reshape(M, M), origin="lower", cmap=cmap, vmin=0, vmax=1, extent=[0, 1, 0, 1])
        ax.contour(X, X, alpha.reshape(M, M), levels=[0.5], colors="k", linewidths=0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, pad=2.5)

    # the gas fraction is drawn on its own (fig/bubble_alpha.pdf): the spectrum takes its place
    ax = fig.add_subplot(gs[0, 0:3])
    x0, y0, w, h = ax.get_position().bounds
    ax.set_position([x0 + 0.065, y0, w - 0.065, h])  # room for the tick labels
    k = np.arange(1, 11)
    # highlight of the five bubble modes (not in the report)
    ax.axvspan(0.5, 5.5, color="#2a6fdb", alpha=0.10, lw=0)
    ax.text(3.3, 3.2e-5, "5 bubbles, 5 modes", ha="center", va="center", fontsize=6, color="#1d4f9e")
    ax.semilogy(k, vals, "o-", color="#2a6fdb", ms=2.8, lw=0.9, label=r"density ratio $10^3$")
    ax.semilogy(k, hv, "s--", color="#e8642c", ms=2.4, lw=0.9, label="density ratio 1")
    ax.axvline(5.5, color="0.6", lw=0.5, ls=":")
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(k)
    ax.set_xlabel(r"$k$", labelpad=0)
    ax.set_title(r"10 smallest $\lambda_k$ of $A\,u=\lambda\,\mathrm{diag}(A)\,u$", pad=2.5)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.05, -0.03), fontsize=5.6,
              handlelength=1.3, handletextpad=0.4, borderaxespad=0.0, labelspacing=0.3)
    ax.grid(alpha=0.3, lw=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    field(fig.add_subplot(gs[0, 3]), vecs[:, 5], r"$u_6$, $\lambda_6=%s$" % sci(vals[5]), "RdBu_r", True)
    for i in range(4):
        field(fig.add_subplot(gs[1, i]), vecs[:, i],
              r"$u_{%d}$, $\lambda_{%d}=%s$" % (i + 1, i + 1, sci(vals[i])), "RdBu_r", True)
    fig.savefig(out, transparent=True)
    print("written", os.path.normpath(out))

    # gas fraction alone, same style as the first panel of the report figure
    fig = plt.figure(figsize=(ALPHA_IN, ALPHA_IN))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    field(ax, alpha, "", "Blues", False)
    ax.plot([0, 1], [1, 1], color="#e8642c", lw=2.0, clip_on=False, solid_capstyle="butt")
    out_a = os.path.join(os.path.dirname(out), "bubble_alpha.pdf")
    fig.savefig(out_a, transparent=True)
    print("written", os.path.normpath(out_a))


if __name__ == "__main__":
    main()
