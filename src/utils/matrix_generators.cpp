/**
 * @file matrix_generators.cpp
 * @brief Implementation of MatrixGenerator (finite-volume assembly).
 */
#include "matrix_generators.hpp"

#include <algorithm>
#include <cmath>
#include <random>
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
    GeneratorOptions o;
    o.nx = nx;
    o.ny = ny;
    o.nz = nz;
    o.nu = nu;
    o.ncomp = ncomp;
    return generate(name, o);
}

GeneratedProblem MatrixGenerator::generate(const std::string& name, const GeneratorOptions& opt)
{
    if (name.size() < 3) throw std::invalid_argument("unknown generator '" + name + "'");
    const std::string suffix = name.substr(name.size() - 2);
    const std::string kind = name.substr(0, name.size() - 2);
    if ((suffix != "2d" && suffix != "3d") ||
        (kind != "laplace" && kind != "convdiff" && kind != "coupled" && kind != "bubbles"))
        throw std::invalid_argument("unknown generator '" + name + "'");
    int nx = opt.nx, ny = opt.ny, nz = opt.nz;
    const double nu = opt.nu;
    const int ncomp = opt.ncomp;
    if (nx < 1 || ny < 1 || nz < 1) throw std::invalid_argument("generator: grid sizes must be positive");
    if (kind == "bubbles") return bubbles(suffix == "3d" ? 3 : 2, opt);

    const int dim = (suffix == "3d") ? 3 : 2;
    if (dim == 2) nz = 1;
    const bool convection = (kind != "laplace");
    const bool heterogeneous = (kind != "laplace");
    const int nc = (kind == "coupled") ? std::max(1, ncomp) : 1;
    const double nuBase = (kind == "laplace") ? 1.0 : nu;
    // Time-dependent convection (sequences of matrices with the same pattern): factor 1 at time 0.
    const double pi = 3.14159265358979323846;
    const double convFactor = 1.0 + opt.pulse * std::sin(2.0 * pi * opt.time / std::max(1, opt.pulse_period));

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
        F *= convFactor;
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
    if (opt.pulse != 0.0) d << " pulse=" << opt.pulse << " t=" << opt.time;
    d << " (n=" << n << ", nnz=" << g.A.nnz() << ")";
    g.description = d.str();
    return g;
}

GeneratedProblem MatrixGenerator::bubbles(int dim, const GeneratorOptions& opt)
{
    const int nx = opt.nx, ny = opt.ny, nz = (dim == 3) ? opt.nz : 1;
    if (opt.rho_ratio <= 0.0 || opt.radius <= 0.0 || opt.nbubbles < 0)
        throw std::invalid_argument("bubbles: rho_ratio and radius must be > 0, nbubbles >= 0");
    const double hx = 1.0 / nx, hy = 1.0 / ny, hz = (dim == 3) ? 1.0 / nz : 1.0;
    const int nv = (dim == 3) ? nz : ny;             // cells along the vertical (gravity) direction
    const double hv = 1.0 / nv;
    const core::GlobalIndex ncell = static_cast<core::GlobalIndex>(nx) * ny * nz;
    auto cid = [&](int i, int j, int k) {
        return static_cast<core::GlobalIndex>(i) + nx * (static_cast<core::GlobalIndex>(j) + static_cast<core::GlobalIndex>(ny) * k);
    };

    // ---- phase fraction alpha (gas = 1): union of smoothed spheres/discs.
    const double pi = 3.14159265358979323846;
    const double R = opt.radius;
    const double w = std::max(hx, std::max(hy, (dim == 3) ? hz : hy)); // interface half-thickness ~ one cell
    const double reach = R + 4.0 * w;                                  // beyond: alpha = 0 to round-off
    std::vector<double> alpha(static_cast<std::size_t>(ncell), 0.0);
    std::mt19937_64 gen(opt.seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const double rise = opt.cfl * hv * opt.time;
    for (int b = 0; b < opt.nbubbles; ++b) {
        // Initial centre, wobble phase (drawn in the same order for every time: same bubbles).
        const double x0 = R + (1.0 - 2.0 * R) * u01(gen), y0 = R + (1.0 - 2.0 * R) * u01(gen), z0 = u01(gen);
        const double phase = 2.0 * pi * u01(gen);
        const double wob = 0.5 * R * std::sin(2.0 * pi * opt.time / 40.0 + phase);
        double c[3];
        if (dim == 3) {
            c[0] = std::clamp(x0 + wob, R, 1.0 - R);
            c[1] = y0;
            c[2] = z0 + rise;
        } else {
            c[0] = std::clamp(x0 + wob, R, 1.0 - R);
            c[1] = z0 + rise;                       // vertical = y in 2D
            c[2] = 0.5;
        }
        const int vdir = (dim == 3) ? 2 : 1;
        c[vdir] -= std::floor(c[vdir]);             // periodic in the vertical direction
        const double h[3] = {hx, hy, hz};
        const int n[3] = {nx, ny, nz};
        for (int image = -1; image <= 1; ++image) { // copies above/below for the periodic wrap
            double cc[3] = {c[0], c[1], c[2]};
            cc[vdir] += image;
            int lo[3], hi[3];
            bool empty = false;
            for (int d = 0; d < dim; ++d) {
                lo[d] = std::max(0, static_cast<int>(std::floor((cc[d] - reach) / h[d])));
                hi[d] = std::min(n[d] - 1, static_cast<int>(std::floor((cc[d] + reach) / h[d])));
                empty = empty || lo[d] > hi[d];
            }
            if (dim == 2) lo[2] = hi[2] = 0;
            if (empty) continue;
            for (int k = lo[2]; k <= hi[2]; ++k)
                for (int j = lo[1]; j <= hi[1]; ++j)
                    for (int i = lo[0]; i <= hi[0]; ++i) {
                        const double dx = (i + 0.5) * hx - cc[0], dy = (j + 0.5) * hy - cc[1];
                        const double dz = (dim == 3) ? (k + 0.5) * hz - cc[2] : 0.0;
                        const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                        const double a = 0.5 * (1.0 - std::tanh((dist - R) / w));
                        double& al = alpha[static_cast<std::size_t>(cid(i, j, k))];
                        al = std::max(al, a);
                    }
        }
    }

    // ---- beta = 1/rho per cell, rho = alpha rho_g + (1 - alpha) rho_l, rho_l = 1.
    const double rho_g = 1.0 / opt.rho_ratio;
    std::vector<double> beta(static_cast<std::size_t>(ncell));
    for (std::size_t q = 0; q < beta.size(); ++q) beta[q] = 1.0 / (alpha[q] * rho_g + (1.0 - alpha[q]));

    // ---- CSR assembly. The columns of a 5/7-point stencil are already sorted in the cell
    //      numbering (k-1, j-1, i-1, P, i+1, j+1, k+1), so the CSR is written directly.
    GeneratedProblem g;
    part::CsrMatrix& A = g.A;
    A.nrows = A.ncols = ncell;
    A.row_ptr.assign(static_cast<std::size_t>(ncell) + 1, 0);
    A.col.reserve(static_cast<std::size_t>(ncell) * (2 * dim + 1));
    A.val.reserve(A.col.capacity());
    g.rhs.assign(static_cast<std::size_t>(ncell), 0.0);
    const double vol = hx * hy * hz;
    const double gx = hy * hz / hx, gy = hx * hz / hy, gz = hx * hy / hz; // area / distance per direction
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const core::GlobalIndex P = cid(i, j, k);
                const double bP = beta[static_cast<std::size_t>(P)];
                double diag = 0.0;
                std::size_t diag_pos = 0;
                auto face = [&](core::GlobalIndex N, double geom) {
                    const double coef = 0.5 * (bP + beta[static_cast<std::size_t>(N)]) * geom;
                    A.col.push_back(N);
                    A.val.push_back(-coef);
                    diag += coef;
                };
                if (dim == 3 && k > 0) face(cid(i, j, k - 1), gz);
                if (j > 0) face(cid(i, j - 1, k), gy);
                if (i > 0) face(cid(i - 1, j, k), gx);
                diag_pos = A.col.size();
                A.col.push_back(P);
                A.val.push_back(0.0);
                if (i + 1 < nx) face(cid(i + 1, j, k), gx);
                if (j + 1 < ny) face(cid(i, j + 1, k), gy);
                if (dim == 3 && k + 1 < nz) face(cid(i, j, k + 1), gz);
                // p = 0 on the top face (outlet / free atmosphere), zero flux elsewhere.
                const bool top = (dim == 3) ? (k == nz - 1) : (j == ny - 1);
                if (top) diag += bP * ((dim == 3) ? hx * hy / (0.5 * hz) : hx * hz / (0.5 * hy));
                A.val[diag_pos] = diag;
                A.row_ptr[static_cast<std::size_t>(P) + 1] = static_cast<core::GlobalIndex>(A.col.size());
                g.rhs[static_cast<std::size_t>(P)] = vol * alpha[static_cast<std::size_t>(P)]; // buoyancy-like source
            }
    A.symmetric = true;
    g.block_size = 1;
    std::ostringstream d;
    d << "bubbles" << dim << "d " << nx << "x" << ny;
    if (dim == 3) d << "x" << nz;
    d << " rho_ratio=" << opt.rho_ratio << " nb=" << opt.nbubbles << " R=" << opt.radius << " cfl=" << opt.cfl
      << " t=" << opt.time << " (n=" << ncell << ", nnz=" << A.nnz() << ")";
    g.description = d.str();
    return g;
}

} // namespace schwarz2lvl::utils
