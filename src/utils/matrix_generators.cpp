/**
 * @file matrix_generators.cpp
 * @brief Implementation of MatrixGenerator (finite-volume assembly).
 */
#include "matrix_generators.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace schwarz2lvl::utils {

namespace {

/// @brief Distance from point p to the segment [a, b] in the plane.
double segmentDistance(double px, double py, double ax, double ay, double bx, double by)
{
    const double vx = bx - ax, vy = by - ay;
    const double t = std::clamp(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy), 0.0, 1.0);
    const double dx = px - (ax + t * vx), dy = py - (ay + t * vy);
    return std::sqrt(dx * dx + dy * dy);
}

/// @brief Primitive of (2s - 1): s^2 - s.
double P1(double s) { return s * s - s; }

} // namespace

double MatrixGenerator::kappa(double x, double y)
{
    // region with kappa = 2
    if (x >= 0.10 && x <= 0.45 && y >= 0.12 && y <= 0.30) return 2.0;
    // two low-diffusivity channels
    if (segmentDistance(x, y, 0.15, 0.62, 0.75, 0.70) < 0.04) return 6e-2;
    if (segmentDistance(x, y, 0.55, 0.62, 0.90, 0.35) < 0.03) return 6e-2;
    // periodic lattice of low-diffusivity square inclusions
    const double px = std::fmod(x, 0.2) - 0.1, py = std::fmod(y, 0.2) - 0.1;
    if (std::abs(px) < 0.035 && std::abs(py) < 0.035) return 6e-2;
    return 1.0;
}

GeneratedProblem MatrixGenerator::generate(const std::string& name, int nx, int ny, int nz, double nu, int ncomp)
{
    if (name.size() < 3) throw std::invalid_argument("unknown generator '" + name + "'");
    const std::string suffix = name.substr(name.size() - 2);
    const std::string kind = name.substr(0, name.size() - 2);
    if ((suffix != "2d" && suffix != "3d") || (kind != "laplace" && kind != "convdiff" && kind != "coupled"))
        throw std::invalid_argument("unknown generator '" + name + "'");
    if (nx < 1 || ny < 1 || nz < 1) throw std::invalid_argument("generator: grid sizes must be positive");

    const int dim = (suffix == "3d") ? 3 : 2;
    if (dim == 2) nz = 1;
    const bool convection = (kind != "laplace");
    const bool heterogeneous = (kind != "laplace");
    const int nc = (kind == "coupled") ? std::max(1, ncomp) : 1;
    const double nuBase = (kind == "laplace") ? 1.0 : nu;

    const double hx = 1.0 / nx, hy = 1.0 / ny, hz = (dim == 3) ? 1.0 / nz : 1.0;
    const core::GlobalIndex ncell = static_cast<core::GlobalIndex>(nx) * ny * nz;
    const core::GlobalIndex n = ncell * nc;

    auto cid = [&](int i, int j, int k) { return static_cast<core::GlobalIndex>(i) + nx * (static_cast<core::GlobalIndex>(j) + static_cast<core::GlobalIndex>(ny) * k); };
    auto dof = [&](core::GlobalIndex cell, int c) { return cell * nc + c; };
    auto nuOf = [&](int c) { return nuBase * (1.0 + c); };
    auto kappaCell = [&](int i, int j) { return heterogeneous ? kappa((i + 0.5) * hx, (j + 0.5) * hy) : 1.0; };

    std::vector<core::GlobalIndex> rows, cols;
    std::vector<core::Scalar> vals;
    const std::size_t est = static_cast<std::size_t>(n) * (2 * dim + 1 + (nc > 1 ? 1 : 0));
    rows.reserve(est);
    cols.reserve(est);
    vals.reserve(est);
    std::vector<core::Scalar> rhs(static_cast<std::size_t>(n), 0.0);
    auto add = [&](core::GlobalIndex r, core::GlobalIndex c, core::Scalar v) {
        rows.push_back(r);
        cols.push_back(c);
        vals.push_back(v);
    };

    // Interior face between owner P and neighbour N. geom = kappa_f * area / dist,
    // F = convective flux from P to N (exact face integral of V.n).
    auto addFace = [&](core::GlobalIndex P, core::GlobalIndex N, double geom, double F) {
        for (int c = 0; c < nc; ++c) {
            const double D = nuOf(c) * geom;
            const core::GlobalIndex p = dof(P, c), q = dof(N, c);
            add(p, p, D + std::max(F, 0.0));
            add(p, q, -D - std::max(-F, 0.0));
            add(q, q, D + std::max(-F, 0.0));
            add(q, p, -D - std::max(F, 0.0));
        }
    };
    // Dirichlet boundary face of cell P with value ub (V.n = 0 on the boundary).
    auto addDirichlet = [&](core::GlobalIndex P, double geom, double ub) {
        for (int c = 0; c < nc; ++c) {
            const double D = nuOf(c) * geom;
            add(dof(P, c), dof(P, c), D);
            rhs[static_cast<std::size_t>(dof(P, c))] += D * ub;
        }
    };

    const double gammaCoupling = nuBase * ((dim == 2) ? hy / hx : hy * hz / hx);

    for (int k = 0; k < nz; ++k) {
        const double z0 = k * hz, z1 = (k + 1) * hz;
        for (int j = 0; j < ny; ++j) {
            const double y0 = j * hy, y1 = (j + 1) * hy;
            for (int i = 0; i < nx; ++i) {
                const double x0 = i * hx, x1 = (i + 1) * hx;
                const core::GlobalIndex P = cid(i, j, k);
                const double kP = kappaCell(i, j);

                // +x face
                if (i + 1 < nx) {
                    const double kN = kappaCell(i + 1, j), kf = 2.0 * kP * kN / (kP + kN);
                    const double area = hy * hz, xf = x1;
                    double F = 0.0;
                    if (convection)
                        F = (dim == 2) ? xf * (1.0 - xf) * (P1(y1) - P1(y0))
                                       : 2.0 * xf * (1.0 - xf) * (P1(y1) - P1(y0)) * 0.5 * (z1 * z1 - z0 * z0);
                    addFace(P, cid(i + 1, j, k), kf * area / hx, F);
                }
                // +y face
                if (j + 1 < ny) {
                    const double kN = kappaCell(i, j + 1), kf = 2.0 * kP * kN / (kP + kN);
                    const double area = hx * hz, yf = y1;
                    double F = 0.0;
                    if (convection) F = -yf * (1.0 - yf) * (P1(x1) - P1(x0)) * ((dim == 2) ? 1.0 : (z1 - z0));
                    addFace(P, cid(i, j + 1, k), kf * area / hy, F);
                }
                // +z face
                if (dim == 3 && k + 1 < nz) {
                    const double area = hx * hy, zf = z1;
                    double F = 0.0;
                    if (convection) F = -zf * (1.0 - zf) * (P1(x1) - P1(x0)) * (P1(y1) - P1(y0));
                    addFace(P, cid(i, j, k + 1), kP * area / hz, F);
                }
                // Dirichlet boundaries: y = 0 (u=0), y = 1 (u=1) in 2D; z = 0 (u=1), z = 1 (u=0) in 3D.
                if (dim == 2) {
                    if (j == 0) addDirichlet(P, kP * hx / (0.5 * hy), 0.0);
                    if (j == ny - 1) addDirichlet(P, kP * hx / (0.5 * hy), 1.0);
                } else {
                    if (k == 0) addDirichlet(P, kP * hx * hy / (0.5 * hz), 1.0);
                    if (k == nz - 1) addDirichlet(P, kP * hx * hy / (0.5 * hz), 0.0);
                }
                // Nonsymmetric cyclic coupling between the components of a cell.
                if (nc > 1)
                    for (int c = 0; c < nc; ++c) {
                        add(dof(P, c), dof(P, c), gammaCoupling);
                        add(dof(P, c), dof(P, (c + 1) % nc), -gammaCoupling);
                    }
            }
        }
    }

    GeneratedProblem g;
    g.A = part::CsrMatrix::fromCoo(n, n, rows, cols, vals);
    g.A.symmetric = (kind == "laplace");
    g.rhs = std::move(rhs);
    g.block_size = nc;
    std::ostringstream d;
    d << name << " " << nx << "x" << ny;
    if (dim == 3) d << "x" << nz;
    if (kind != "laplace") d << " nu=" << nu;
    if (nc > 1) d << " ncomp=" << nc;
    d << " (n=" << n << ", nnz=" << g.A.nnz() << ")";
    g.description = d.str();
    return g;
}

} // namespace schwarz2lvl::utils
