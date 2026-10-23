/**
 * @file bubble_field.hpp
 * @brief Phase fraction of the bubbles generators: header-only, C++14, standard library only.
 *
 * Shared by utils::MatrixGenerator::bubbles and by the OpenFOAM application
 * `openfoam/bubblyPressureFoam` (compiled with the OpenFOAM toolchain, C++14): both evaluate the
 * same bubbles with the same arithmetic, so the pressure matrix assembled by OpenFOAM equals the
 * generated one up to round-off, on any decomposition.
 */
#ifndef SCHWARZ2LVL_UTILS_BUBBLE_FIELD_HPP
#define SCHWARZ2LVL_UTILS_BUBBLE_FIELD_HPP

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace schwarz2lvl {
namespace utils {

/// @brief Parameters of the bubbles (unit square/cube, uniform grid of nx x ny x nz cells).
struct BubbleFieldParams {
    int dim = 3;                    ///< 2 or 3 (vertical direction: y in 2D, z in 3D).
    int nx = 64, ny = 64, nz = 64;  ///< Cells per direction (nz ignored in 2D).
    int nbubbles = 50;              ///< Number of bubbles.
    double radius = 0.06;           ///< Radius.
    double cfl = 0.5;               ///< Rise per step, in cells.
    double time = 0.0;              ///< Pseudo-time [steps].
    unsigned seed = 2024;           ///< Seed of the initial positions.
};

/**
 * @class BubbleField
 * @brief Gas fraction alpha of rising bubbles: union of smoothed spheres/discs.
 *
 * Every bubble starts at a random position (seed), rises by `cfl` cells per step (periodic in the
 * vertical direction: a bubble leaving the top re-enters at the bottom) and wobbles sideways.
 * alpha = 0.5 (1 - tanh((d - R) / w)) with d the distance of the cell centre from the bubble
 * centre and w one cell (interface about two cells thick, as a VoF interface); the maximum over
 * the bubbles; alpha = 0 outside the index box of half-width R + 4w around each bubble.
 */
class BubbleField {
public:
    explicit BubbleField(const BubbleFieldParams& p) : p_(p)
    {
        if (p.dim != 2 && p.dim != 3) throw std::invalid_argument("BubbleField: dim must be 2 or 3");
        if (p.nx < 1 || p.ny < 1 || (p.dim == 3 && p.nz < 1)) throw std::invalid_argument("BubbleField: bad grid");
        if (p.radius <= 0.0 || p.nbubbles < 0) throw std::invalid_argument("BubbleField: radius > 0, nbubbles >= 0");
        n_[0] = p.nx;
        n_[1] = p.ny;
        n_[2] = (p.dim == 3) ? p.nz : 1;
        h_[0] = 1.0 / p.nx;
        h_[1] = 1.0 / p.ny;
        h_[2] = (p.dim == 3) ? 1.0 / p.nz : 1.0;
        const int nv = (p.dim == 3) ? p.nz : p.ny; // cells along the vertical direction
        const double hv = 1.0 / nv;
        const double pi = 3.14159265358979323846;
        const double R = p.radius;
        w_ = std::max(h_[0], std::max(h_[1], (p.dim == 3) ? h_[2] : h_[1]));
        const double reach = R + 4.0 * w_;
        std::mt19937_64 gen(p.seed);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        const double rise = p.cfl * hv * p.time;
        const int vdir = (p.dim == 3) ? 2 : 1;
        for (int b = 0; b < p.nbubbles; ++b) {
            // Initial centre and wobble phase (drawn in the same order for every time: same bubbles).
            const double x0 = R + (1.0 - 2.0 * R) * u01(gen), y0 = R + (1.0 - 2.0 * R) * u01(gen), z0 = u01(gen);
            const double phase = 2.0 * pi * u01(gen);
            const double wob = 0.5 * R * std::sin(2.0 * pi * p.time / 40.0 + phase);
            const double xw = x0 + wob;
            const double xc = (xw < R) ? R : ((1.0 - R < xw) ? 1.0 - R : xw); // clamp to [R, 1 - R]
            double c[3];
            if (p.dim == 3) {
                c[0] = xc;
                c[1] = y0;
                c[2] = z0 + rise;
            } else {
                c[0] = xc;
                c[1] = z0 + rise; // vertical = y in 2D
                c[2] = 0.5;
            }
            c[vdir] -= std::floor(c[vdir]); // periodic in the vertical direction
            for (int image = -1; image <= 1; ++image) { // copies above/below for the periodic wrap
                Image im;
                im.c[0] = c[0];
                im.c[1] = c[1];
                im.c[2] = c[2];
                im.c[vdir] += image;
                bool empty = false;
                for (int d = 0; d < p.dim; ++d) {
                    im.lo[d] = std::max(0, static_cast<int>(std::floor((im.c[d] - reach) / h_[d])));
                    im.hi[d] = std::min(n_[d] - 1, static_cast<int>(std::floor((im.c[d] + reach) / h_[d])));
                    empty = empty || im.lo[d] > im.hi[d];
                }
                if (p.dim == 2) im.lo[2] = im.hi[2] = 0;
                if (!empty) images_.push_back(im);
            }
        }
    }

    /// @brief alpha of cell (i, j, k) (k = 0 in 2D).
    double alpha(int i, int j, int k) const
    {
        double al = 0.0;
        for (const Image& im : images_)
            if (i >= im.lo[0] && i <= im.hi[0] && j >= im.lo[1] && j <= im.hi[1] && k >= im.lo[2] && k <= im.hi[2])
                al = std::max(al, profile(im, i, j, k));
        return al;
    }

    /// @brief alpha of all cells, numbered i + nx (j + ny k) (one pass over the boxes: fast).
    void fill(std::vector<double>& alpha) const
    {
        alpha.assign(static_cast<std::size_t>(n_[0]) * static_cast<std::size_t>(n_[1]) * static_cast<std::size_t>(n_[2]), 0.0);
        for (const Image& im : images_)
            for (int k = im.lo[2]; k <= im.hi[2]; ++k)
                for (int j = im.lo[1]; j <= im.hi[1]; ++j)
                    for (int i = im.lo[0]; i <= im.hi[0]; ++i) {
                        double& al = alpha[static_cast<std::size_t>(i) +
                                           static_cast<std::size_t>(n_[0]) *
                                               (static_cast<std::size_t>(j) + static_cast<std::size_t>(n_[1]) * static_cast<std::size_t>(k))];
                        al = std::max(al, profile(im, i, j, k));
                    }
    }

    /// @brief Index of the cell containing the point (x, y, z) of the unit square/cube.
    void cellOf(double x, double y, double z, int& i, int& j, int& k) const
    {
        const double q[3] = {x, y, z};
        int* out[3] = {&i, &j, &k};
        for (int d = 0; d < 3; ++d) {
            int c = static_cast<int>(std::floor(q[d] / h_[d]));
            *out[d] = (d < p_.dim) ? std::min(n_[d] - 1, std::max(0, c)) : 0;
        }
    }

    const BubbleFieldParams& params() const { return p_; }

private:
    struct Image {
        double c[3];      ///< Centre.
        int lo[3], hi[3]; ///< Index box of the cells within R + 4w (per direction).
    };
    BubbleFieldParams p_;
    int n_[3];
    double h_[3];
    double w_ = 0.0;              ///< Interface half-thickness: one cell.
    std::vector<Image> images_;   ///< Bubbles and their periodic copies with a non-empty box.

    double profile(const Image& im, int i, int j, int k) const
    {
        const double dx = (i + 0.5) * h_[0] - im.c[0], dy = (j + 0.5) * h_[1] - im.c[1];
        const double dz = (p_.dim == 3) ? (k + 0.5) * h_[2] - im.c[2] : 0.0;
        const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        return 0.5 * (1.0 - std::tanh((dist - p_.radius) / w_));
    }
};

} // namespace utils
} // namespace schwarz2lvl

#endif // SCHWARZ2LVL_UTILS_BUBBLE_FIELD_HPP
