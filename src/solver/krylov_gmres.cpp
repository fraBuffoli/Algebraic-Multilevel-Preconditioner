#include "krylov_gmres.hpp"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

namespace schwarz2lvl {

bool KrylovGmres::solve(const SparseMatrixWrapper& A,
                        const VectorType& b,
                        VectorType& x,
                        const Preconditioner& prec) const {
    
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    const auto& local_matrix = A.getMatrix();
    const Eigen::Index n_local = local_matrix.rows();
    
    // 1. Calcolo parallelo della norma del termine noto ||b||
    double local_b_norm_sq = b.squaredNorm();
    double global_b_norm_sq = 0.0;
    MPI_Allreduce(&local_b_norm_sq, &global_b_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double b_norm = std::sqrt(global_b_norm_sq);

    if (b_norm == 0.0) {
        x.setZero(n_local);
        last_iterations_ = 0;
        last_residual_   = 0.0;
        if (verbose_ && my_rank == 0) {
            std::cout << "GMRES: Right-hand side is zero vector. Solution is zero." << std::endl;
        }
        return true;
    }

    if (verbose_ && my_rank == 0) {
        std::cout << "Starting PARALLEL GMRES with Restart(" << restart_dim_ << "), Target Tolerance: " << tolerance_ << std::endl;
    }

    VectorType r(n_local);
    VectorType v_tmp(n_local);
    VectorType w(n_local);

    int total_iters = 0;
    double rel_res = 1.0;
    
    while (total_iters < max_iter_ && rel_res > tolerance_) {
        // r = b - A * x (Prodotto locale, le comunicazioni di bordo avvengono qui dentro)
        r = b - local_matrix * x;
        
        double local_r_norm_sq = r.squaredNorm();
        double global_r_norm_sq = 0.0;
        MPI_Allreduce(&local_r_norm_sq, &global_r_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double r_norm = std::sqrt(global_r_norm_sq);
        rel_res = r_norm / b_norm;

        if (verbose_ && total_iters == 0 && my_rank == 0) {
            std::cout << "  Iteration: " << total_iters << " -> Relative Residual = " << rel_res << std::endl;
        }

        if (rel_res <= tolerance_) {
            last_iterations_ = total_iters;
            last_residual_   = rel_res;
            return true;
        }

        int m = std::min(restart_dim_, max_iter_ - total_iters);

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m + 1, m);
        std::vector<VectorType> V(m + 1, VectorType::Zero(n_local));
        Eigen::VectorXd g = Eigen::VectorXd::Zero(m + 1);
        g(0) = r_norm;

        V[0] = r / r_norm;

        std::vector<double> cs(m, 0.0);
        std::vector<double> sn(m, 0.0);

        int k = 0;
        for (k = 0; k < m; ++k) {
            total_iters++;

            // Step A: Precondizionamento Parallelo Distribuito
            v_tmp.setZero(n_local);
            {
                ScopedTimer p_timer("solver.preconditioner_apply");
                prec.apply(V[k], v_tmp);
            }

            // Step B: Prodotto Matrice-Vettore Parallelo
            {
                ScopedTimer mv_timer("solver.matrix_vector_product");
                w = local_matrix * v_tmp;
            }

            // Step C: Ortogonalizzazione di Gram-Schmidt PARALLELA
            for (int i = 0; i <= k; ++i) {
                double local_dot = w.dot(V[i]);
                double global_dot = 0.0;
                // Sincronizziamo il prodotto scalare su tutto il cluster
                MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                
                H(i, k) = global_dot;
                w -= H(i, k) * V[i];
            }
            
            double local_w_norm_sq = w.squaredNorm();
            double global_w_norm_sq = 0.0;
            MPI_Allreduce(&local_w_norm_sq, &global_w_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            H(k + 1, k) = std::sqrt(global_w_norm_sq);

            if (H(k + 1, k) < 1e-15) {
                H(k + 1, k) = 0.0;
            } else {
                V[k + 1] = w / H(k + 1, k);
            }

            // Step D & E: Rotazioni di Givens (Locali, lavorano su scalari già sincronizzati)
            for (int i = 0; i < k; ++i) {
                double temp = cs[i] * H(i, k) + sn[i] * H(i + 1, k);
                H(i + 1, k) = -sn[i] * H(i, k) + cs[i] * H(i + 1, k);
                H(i, k) = temp;
            }

            if (H(k + 1, k) == 0.0) {
                cs[k] = 1.0; sn[k] = 0.0;
            } else {
                if (std::abs(H(k, k)) > std::abs(H(k + 1, k))) {
                    double t = H(k + 1, k) / H(k, k);
                    cs[k] = 1.0 / std::sqrt(1.0 + t * t);
                    sn[k] = cs[k] * t;
                } else {
                    double t = H(k, k) / H(k + 1, k);
                    sn[k] = 1.0 / std::sqrt(1.0 + t * t);
                    cs[k] = sn[k] * t;
                }
            }

            H(k, k) = cs[k] * H(k, k) + sn[k] * H(k + 1, k);
            H(k + 1, k) = 0.0; 

            double g_temp = cs[k] * g(k);
            g(k + 1) = -sn[k] * g(k);
            g(k) = g_temp;

            rel_res = std::abs(g(k + 1)) / b_norm;
            if (verbose_ && my_rank == 0) {
                std::cout << "  Iteration: " << total_iters << " -> Relative Residual = " << rel_res << std::endl;
            }

            if (rel_res <= tolerance_) {
                k++; break;
            }
        }

        // Risoluzione del piccolo sistema triangolare (identico su tutti i rank)
        Eigen::VectorXd y = Eigen::VectorXd::Zero(k);
        for (int i = k - 1; i >= 0; --i) {
            double sum = 0.0;
            for (int j = i + 1; j < k; ++j) {
                sum += H(i, j) * y(j);
            }
            y(i) = (g(i) - sum) / H(i, i);
        }

        v_tmp.setZero(n_local);
        for (int i = 0; i < k; ++i) {
            v_tmp += V[i] * y(i);
        }

        w.setZero(n_local); 
        {
            ScopedTimer p_timer("solver.preconditioner_apply");
            prec.apply(v_tmp, w);
        }
        x += w;
    }

    last_iterations_ = total_iters;
    last_residual_   = rel_res;

    if (rel_res <= tolerance_) {
        if (verbose_ && my_rank == 0) {
            std::cout << "GMRES Successfully Converged! Total Iterations: "
                      << total_iters << std::endl;
        }
        return true;
    }
    return false;
}

} // namespace schwarz2lvl
