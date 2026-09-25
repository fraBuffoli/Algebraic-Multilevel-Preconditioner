/**
 * @file matrix_generators.hpp
 * @brief Built-in finite-volume test problems (rank 0), inspired by Section 4.2
 *        of the paper and by OpenFOAM discretizations.
 */
#ifndef SCHWARZ2LVL_UTILS_MATRIX_GENERATORS_HPP
#define SCHWARZ2LVL_UTILS_MATRIX_GENERATORS_HPP

#include "csr_matrix.hpp"

#include <string>
#include <vector>

namespace schwarz2lvl::utils {

/// @brief Output of a generator: matrix, right-hand side and cell block size.
struct GeneratedProblem {
    part::CsrMatrix A;              ///< System matrix.
    std::vector<core::Scalar> rhs;  ///< Right-hand side from the boundary conditions.
    int block_size = 1;       ///< Unknowns per cell.
    std::string description;  ///< Human readable description.
};

/**
 * @class MatrixGenerator
 * @brief Cell-centred finite-volume generators on the unit square/cube.
 *
 * All problems discretize the steady convection-diffusion equation (4.1)
 * of the paper,
 * \f[ \nabla\cdot(V u) - \nu \nabla\cdot(\kappa \nabla u) = 0, \f]
 * with first-order upwind convection (OpenFOAM `Gauss upwind`) and a
 * two-point diffusion flux with harmonic face diffusivity. The velocity
 * fields are those of Section 4.2; face fluxes are integrated *exactly*, so the
 * discrete divergence of V vanishes and every row not touching a Dirichlet
 * boundary has zero row sum (the lumped local matrices A~_ii of floating
 * subdomains are then exactly singular, which exercises the kernel handling
 * of (3.2)). The diffusivity kappa is a heterogeneous field loosely inspired
 * by Fig. 3(b) (background 1, inclusions 6e-2, one region 2).
 *
 * Boundary conditions: u = 1 and u = 0 on two opposite faces (y in 2D, z in 3D),
 * zero flux elsewhere (V.n = 0 on the whole boundary).
 *
 * Available names:
 *  - `laplace2d`, `laplace3d`   : V = 0, kappa = 1, nu = 1 (SPD);
 *  - `convdiff2d`, `convdiff3d` : Eq. (4.1) with the given nu (nonsymmetric);
 *  - `coupled2d`, `coupled3d`   : `ncomp` convection-diffusion equations per cell
 *    (viscosity nu*(1+c) for component c) with a nonsymmetric cyclic coupling
 *    between components, unknowns interleaved cell by cell as in OpenFOAM
 *    coupled solvers (block size = ncomp).
 */
class MatrixGenerator {
public:
    /**
     * @brief Generates a problem.
     * @param name  Generator name (see class description).
     * @param nx    Cells along x.
     * @param ny    Cells along y.
     * @param nz    Cells along z (3D only).
     * @param nu    Viscosity.
     * @param ncomp Unknowns per cell (coupled generators only).
     * @return The generated problem.
     * @throws std::invalid_argument for unknown names.
     */
    static GeneratedProblem generate(const std::string& name, int nx, int ny, int nz, double nu, int ncomp);

    /**
     * @brief Diffusivity field kappa(x, y) (z-independent).
     * @param x Abscissa in [0, 1].
     * @param y Ordinate in [0, 1].
     */
    static double kappa(double x, double y);
};

} // namespace schwarz2lvl::utils

#endif // SCHWARZ2LVL_UTILS_MATRIX_GENERATORS_HPP
