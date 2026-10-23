/**
 * @file test_bubbles.cpp
 * @brief Bubble field shared with the OpenFOAM application bubblyPressureFoam (bubble_field.hpp).
 *
 *  - BubbleField::alpha(i, j, k) (one cell, used by OpenFOAM at the cell centres) equals
 *    BubbleField::fill (whole grid, used by the generator), 3D and 2D, also across the periodic wrap;
 *  - cellOf maps every cell centre to its cell;
 *  - the bubbles3d generator: right-hand side = cell volume x alpha of the same field, symmetric
 *    matrix, zero row sums except on the top layer (p = 0 there), face coefficients = arithmetic
 *    mean of beta = 1/rho x area / distance (what OpenFOAM assembles with linear interpolation and
 *    an orthogonal snGrad on the uniform mesh).
 * Sequential checks (every rank does the same).
 */
#include "bubble_field.hpp"
#include "matrix_generators.hpp"
#include "test_utils.hpp"

#include <cmath>
#include <iostream>

using namespace schwarz2lvl;

namespace {

void checkField(int dim, int n, int nb, double R, double time)
{
    utils::BubbleFieldParams p;
    p.dim = dim;
    p.nx = n;
    p.ny = n + 1;
    p.nz = (dim == 3) ? n + 2 : 1;
    p.nbubbles = nb;
    p.radius = R;
    p.cfl = 1.0;
    p.time = time;
    const utils::BubbleField f(p);
    std::vector<double> all;
    f.fill(all);
    double diff = 0.0, amax = 0.0;
    bool cells_ok = true;
    const int nz = (dim == 3) ? p.nz : 1;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < p.ny; ++j)
            for (int i = 0; i < p.nx; ++i) {
                const double a = all[static_cast<std::size_t>(i + p.nx * (j + p.ny * k))];
                diff = std::max(diff, std::abs(a - f.alpha(i, j, k)));
                amax = std::max(amax, a);
                int ii, jj, kk;
                f.cellOf((i + 0.5) / p.nx, (j + 0.5) / p.ny, (dim == 3) ? (k + 0.5) / p.nz : 0.5, ii, jj, kk);
                cells_ok = cells_ok && ii == i && jj == j && kk == k;
            }
    test::check(diff < 1e-14 && amax > 0.9 && amax <= 1.0 && cells_ok,
                "BubbleField " + std::to_string(dim) + "D, " + std::to_string(nb) + " bubbles, t = " +
                    std::to_string(static_cast<int>(time)) + ": alpha(i,j,k) = fill (max diff " + test::sci(diff) +
                    "), max alpha " + test::sci(amax) + ", cellOf of the centres");
}

void checkGenerator(int n, double time)
{
    utils::GeneratorOptions o;
    o.nx = o.ny = o.nz = n;
    o.nbubbles = 12;
    o.radius = 0.12;
    o.cfl = 1.0;
    o.time = time;
    o.rho_ratio = 1000.0;
    const utils::GeneratedProblem g = utils::MatrixGenerator::generate("bubbles3d", o);
    utils::BubbleFieldParams p;
    p.nx = p.ny = p.nz = n;
    p.nbubbles = o.nbubbles;
    p.radius = o.radius;
    p.cfl = o.cfl;
    p.time = o.time;
    std::vector<double> alpha;
    utils::BubbleField(p).fill(alpha);
    const double h = 1.0 / n, vol = h * h * h;
    auto beta = [&](core::GlobalIndex c) { return 1.0 / (alpha[static_cast<std::size_t>(c)] / o.rho_ratio + (1.0 - alpha[static_cast<std::size_t>(c)])); };
    double rhs_err = 0.0, coef_err = 0.0, sym_err = 0.0, rowsum = 0.0;
    const core::SpMat A = test::toEigen(g.A);
    const core::SpMat At = A.transpose();
    sym_err = (A - At).norm() / A.norm();
    for (core::GlobalIndex r = 0; r < g.A.nrows; ++r) {
        rhs_err = std::max(rhs_err, std::abs(g.rhs[static_cast<std::size_t>(r)] - vol * alpha[static_cast<std::size_t>(r)]));
        double s = 0.0, d = 0.0;
        for (auto q = g.A.row_ptr[static_cast<std::size_t>(r)]; q < g.A.row_ptr[static_cast<std::size_t>(r) + 1]; ++q) {
            const core::GlobalIndex c = g.A.col[static_cast<std::size_t>(q)];
            const double v = g.A.val[static_cast<std::size_t>(q)];
            s += v;
            if (c == r) d = v;
            if (c != r) coef_err = std::max(coef_err, std::abs(-v - 0.5 * (beta(r) + beta(c)) * h) / (0.5 * (beta(r) + beta(c)) * h));
        }
        const bool top = r / (static_cast<core::GlobalIndex>(n) * n) == n - 1;
        if (!top) rowsum = std::max(rowsum, std::abs(s) / d);
    }
    test::check(rhs_err < 1e-18 && coef_err < 1e-13 && sym_err < 1e-15 && rowsum < 1e-12,
                "bubbles3d " + std::to_string(n) + "^3, t = " + std::to_string(static_cast<int>(time)) +
                    ": rhs = vol alpha (err " + test::sci(rhs_err) + "), coefficients = mean beta area/distance (rel. err " +
                    test::sci(coef_err) + "), symmetric, zero row sums below the top layer");
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    try {
        checkField(3, 20, 15, 0.1, 0.0);
        checkField(3, 20, 15, 0.1, 17.0); // bubbles crossing the top re-enter at the bottom
        checkField(3, 13, 1, 0.3, 5.0);
        checkField(2, 40, 10, 0.08, 9.0);
        checkGenerator(16, 0.0);
        checkGenerator(16, 11.0);
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int f = test::failures();
    if (core::mpiRank(MPI_COMM_WORLD) == 0) std::cout << (f ? "FAILED" : "ALL PASSED") << std::endl;
    MPI_Finalize();
    return f ? 1 : 0;
}
