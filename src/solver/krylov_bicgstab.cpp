#include "krylov_bicgstab.hpp"
#include <mpi.h>
#include <iostream>
#include <cmath>

namespace schwarz2lvl {

bool KrylovBicgstab::solve(const SparseMatrixWrapper& A,
                           const VectorType& b,
                           VectorType& x,
                           const Preconditioner& prec) const {
    
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    const auto& local_matrix = A.getMatrix();
    const Eigen::Index n_local = local_matrix.rows();

    // Calcolo parallelo della norma iniziale del termine noto
    double local_b_norm_sq = b.squaredNorm();
    double global_b_norm_sq = 0.0;
    MPI_Allreduce(&local_b_norm_sq, &global_b_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double b_norm = std::sqrt(global_b_norm_sq);

    if (b_norm == 0.0) {
        x.setZero(n_local);
        return true;
    }

    if (my_rank == 0) {
        std::cout << "Starting PARALLEL Preconditioned BiCGStab, Target Tolerance: " << tolerance_ << std::endl;
    }

    // Allocazione dei vettori di lavoro locali
    VectorType r = b - local_matrix * x;
    VectorType r_tilde = r; // Ombra del residuo iniziale
    VectorType p = r;
    VectorType v(n_local), h(n_local), s(n_local), t(n_local), y(n_local), z(n_local);

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    v.setZero(); h.setZero();

    int iter = 0;
    double rel_res = 1.0;

    while (iter < max_iter_) {
        iter++;

        // 1. Calcolo parallelo di rho = (r_tilde, r)
        double local_rho = r.dot(r_tilde);
        double global_rho = 0.0;
        MPI_Allreduce(&local_rho, &global_rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (std::abs(global_rho) < 1e-15) {
            if (my_rank == 0) std::cerr << "BiCGStab Breakdown: rho near zero." << std::endl;
            return false;
        }

        if (iter > 1) {
            double beta = (global_rho / rho_old) * (alpha / omega);
            p = r + beta * (p - omega * v);
        }

        // 2. Primo step di Precondizionamento Destro: y = M^-1 * p
        y.setZero();
        {
            ScopedTimer timer("solver.preconditioner_apply");
            prec.apply(p, y);
        }

        // 3. Primo prodotto Matrice-Vettore: v = A * y
        {
            ScopedTimer timer("solver.matrix_vector_product");
            v = local_matrix * y;
        }

        // 4. Calcolo parallelo di alpha = rho / (r_tilde, v)
        double local_dot_rv = v.dot(r_tilde);
        double global_dot_rv = 0.0;
        MPI_Allreduce(&local_dot_rv, &global_dot_rv, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        alpha = global_rho / global_dot_rv;

        // Vettore intermedio
        s = r - alpha * v;

        // Verifica convergenza intermedia
        double local_s_norm_sq = s.squaredNorm();
        double global_s_norm_sq = 0.0;
        MPI_Allreduce(&local_s_norm_sq, &global_s_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        rel_res = std::sqrt(global_s_norm_sq) / b_norm;
        if (rel_res <= tolerance_) {
            x += alpha * y;
            if (my_rank == 0) std::cout << "BiCGStab Converged (early) at iter: " << iter << " -> Res: " << rel_res << std::endl;
            return true;
        }

        // 5. Secondo step di Precondizionamento Destro: z = M^-1 * s
        z.setZero();
        {
            ScopedTimer timer("solver.preconditioner_apply");
            prec.apply(s, z);
        }

        // 6. Secondo prodotto Matrice-Vettore: t = A * z
        {
            ScopedTimer timer("solver.matrix_vector_product");
            t = local_matrix * z;
        }

        // 7. Calcolo parallelo di omega = (t, s) / (t, t)
        double local_dot_ts = t.dot(s);
        double local_dot_tt = t.dot(t);
        double global_dot_ts = 0.0, global_dot_tt = 0.0;
        MPI_Allreduce(&local_dot_ts, &global_dot_ts, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_dot_tt, &global_dot_tt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        omega = global_dot_ts / global_dot_tt;

        // Aggiornamento della soluzione e del residuo reali
        x += alpha * y + omega * z;
        r = s - omega * t;

        // Controllo convergenza finale dell'iterazione
        double local_r_norm_sq = r.squaredNorm();
        double global_r_norm_sq = 0.0;
        MPI_Allreduce(&local_r_norm_sq, &global_r_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        rel_res = std::sqrt(global_r_norm_sq) / b_norm;

        if (my_rank == 0) {
            std::cout << "  Iteration: " << iter << " -> Relative Residual = " << rel_res << std::endl;
        }

        if (rel_res <= tolerance_) {
            if (my_rank == 0) std::cout << "BiCGStab Successfully Converged! Total Iterations: " << iter << std::endl;
            return true;
        }

        if (std::abs(omega) < 1e-15) {
            if (my_rank == 0) std::cerr << "BiCGStab Breakdown: omega near zero." << std::endl;
            return false;
        }

        rho_old = global_rho;
    }

    if (my_rank == 0) std::cerr << "BiCGStab Warning: Reached max iterations without convergence." << std::endl;
    return false;
}

} // namespace schwarz2lvl
