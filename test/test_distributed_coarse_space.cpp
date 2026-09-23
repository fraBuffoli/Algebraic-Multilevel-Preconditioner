// Verifica DistributedCoarseSpace (Step 6) su tre fronti, contro la
// pipeline esistente (matrice globale riletta da ogni rank SOLO qui):
//
//  1) Lumping locale: LocalBlockSplitting::applyLumping(LocalMatrix, ...)
//     contro la versione con matrice globale. Bound: la somma esterna di k
//     termini ha errore <= gamma_k * sum|a|, due ordini -> 2 gamma_k.
//
//  2) A_00 assemblata: contro R_0 A R_0^T costruita come in CoarseSpace,
//     entrata per entrata, con il bound di Higham per il triplo prodotto:
//        |dA00_pq| <= 2 gamma_K (|R_0| |A| |R_0|^T)_pq,  K = k_A + n_max + 1
//     (k_A = max nnz per riga di A, n_max = max n_i: lunghezze delle due
//     somme annidate). Controllo di sensibilita': azzerando nella A_00 nuova
//     i blocchi "angolo" (sottodomini accoppiati ma NON vicini di halo) il
//     criterio deve bocciarla -- e il test conta quanti angoli esistono,
//     cosi' si vede che il caso e' davvero presente e coperto.
//
//  3) apply: contro CoarseSpace::apply (vettori lunghi n, Allreduce).
//     Qui le due A_00 differiscono per arrotondamento, quindi le soluzioni
//     grossolane differiscono di ~ kappa(A_00) * u (teoria perturbativa):
//     criterio  ||dz|| / ||z|| <= 1e3 * kappa_2(A_00) * u.  Un errore nel
//     flusso dati (restrizione, Allgatherv, prolungamento) da' invece O(1).
//
// La STESSA Z_i (calcolata una volta) alimenta entrambe le versioni, cosi'
// il confronto isola la logica del coarse space dall'eigensolver.
#include "matrix_market_io.hpp"
#include "global_matrix.hpp"
#include "restriction_operator.hpp"
#include "partition_of_unity.hpp"
#include "local_block_splitting.hpp"
#include "local_eigensolver.hpp"
#include "coarse_space.hpp"
#include "domain_decomposer.hpp"
#include "matrix_distributor.hpp"
#include "halo_exchange.hpp"
#include "distributed_spmv.hpp"
#include "distributed_coarse_space.hpp"
#include <Eigen/SVD>
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>

namespace {
double gammaK(double k) {
    const double u = std::numeric_limits<double>::epsilon() / 2.0;
    return k * u / (1.0 - k * u);
}
// max_pq |X - Y| / (2 gamma_K B) sulle righe [row0, row0+nrows).
double boundRatio(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, const Eigen::MatrixXd& B,
                  double gamma, Eigen::Index row0, Eigen::Index nrows) {
    double worst = 0.0;
    for (Eigen::Index p = row0; p < row0 + nrows; ++p)
        for (Eigen::Index q = 0; q < X.cols(); ++q) {
            const double err = std::abs(X(p, q) - Y(p, q));
            if (err == 0.0) continue;
            const double bound = 2.0 * gamma * B(p, q);
            worst = std::max(worst, bound > 0.0 ? err / bound : std::numeric_limits<double>::infinity());
        }
    return worst;
}
} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int my_rank = 0, num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    const std::string matrix_path = argc > 1 ? argv[1] : "matrices/nos5.mtx";
    const double tau = argc > 2 ? std::atof(argv[2]) : schwarz2lvl::kDefaultTau;

    // ================= Percorso NUOVO =====================================
    schwarz2lvl::DomainDecomposer decomposer;
    if (my_rank == 0) decomposer.decompose(matrix_path, num_ranks);
    schwarz2lvl::SubdomainPackage pkg =
        schwarz2lvl::MatrixDistributor::distribute(my_rank == 0 ? &decomposer : nullptr);

    std::vector<int> partition_map;
    int num_rows = 0;
    if (my_rank == 0) {
        partition_map = decomposer.partitionMap();
        num_rows = static_cast<int>(decomposer.numGlobalRows());
    }
    MPI_Bcast(&num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank != 0) partition_map.resize(num_rows);
    MPI_Bcast(partition_map.data(), num_rows, MPI_INT, 0, MPI_COMM_WORLD);

    schwarz2lvl::HaloExchange halo;
    halo.setup(partition_map, pkg.topology);
    schwarz2lvl::DistributedSpMV spmv;
    spmv.setup(pkg.local_matrix, halo);

    const schwarz2lvl::MatrixType A_ii = pkg.local_matrix.extractLocalSquareBlock();
    schwarz2lvl::MatrixType A_tilde = A_ii;
    schwarz2lvl::LocalBlockSplitting lumper;
    lumper.applyLumping(pkg.local_matrix, A_tilde);                 // NUOVO overload, solo dati locali

    schwarz2lvl::PartitionOfUnity D_i;
    D_i.setWeights(std::vector<double>(pkg.pou_weights.data(), pkg.pou_weights.data() + pkg.pou_weights.size()));
    schwarz2lvl::LocalEigensolver eigensolver(tau);
    eigensolver.computeEigenpairs(A_ii, A_tilde, D_i);
    const Eigen::MatrixXd& Z_i = eigensolver.getLocalBase();

    schwarz2lvl::DistributedCoarseSpace coarse_new;
    coarse_new.setup(halo, spmv, pkg.pou_weights, Z_i);

    // ================= Riferimento ========================================
    schwarz2lvl::SparseMatrixWrapper A_ref;
    schwarz2lvl::MatrixMarketIO::readMatrix(matrix_path, A_ref);
    const schwarz2lvl::MatrixType& A = A_ref.getMatrix();
    int k_A = 0;
    for (int i = 0; i < num_rows; ++i) k_A = std::max(k_A, static_cast<int>(A.innerVector(i).nonZeros()));

    // ---- 1) lumping -----------------------------------------------------
    schwarz2lvl::MatrixType A_tilde_ref = A_ii;
    lumper.applyLumping(A_ref, A_tilde_ref, pkg.topology);         // vecchio overload, matrice globale
    double ratio_lump = 0.0;
    {
        const auto& map = pkg.local_matrix.indexMap();
        for (Eigen::Index r = map.numInterior(); r < map.size(); ++r) {
            double scale = std::abs(A_ii.coeff(r, r));
            int k = 0;
            for (schwarz2lvl::MatrixType::InnerIterator it(pkg.local_matrix.raw(), r); it; ++it, ++k)
                if (map.globalToLocal(static_cast<int>(it.col())) == -1) scale += std::abs(it.value());
            const double err = std::abs(A_tilde.coeff(r, r) - A_tilde_ref.coeff(r, r));
            if (err > 0.0) ratio_lump = std::max(ratio_lump, err / (2.0 * gammaK(k + 1) * scale));
        }
        // fuori diagonale e righe interior devono essere identiche (nessuna modifica)
        const schwarz2lvl::MatrixType diff = A_tilde - A_tilde_ref;
        for (int kk = 0; kk < diff.outerSize(); ++kk)
            for (schwarz2lvl::MatrixType::InnerIterator it(diff, kk); it; ++it)
                if (it.row() != it.col() && it.value() != 0.0) ratio_lump = std::numeric_limits<double>::infinity();
    }

    // ---- raccolta di Z, indici estesi e pesi di tutti (come main.cpp) --
    std::vector<Eigen::MatrixXd> all_Z(num_ranks);
    std::vector<schwarz2lvl::RestrictionOperator> all_R(num_ranks);
    std::vector<schwarz2lvl::PartitionOfUnity> all_D(num_ranks);
    int n_max = 0;
    for (int r = 0; r < num_ranks; ++r) {
        int rows_Z = 0, cols_Z = 0;
        if (my_rank == r) { all_Z[r] = Z_i; rows_Z = static_cast<int>(Z_i.rows()); cols_Z = static_cast<int>(Z_i.cols()); }
        MPI_Bcast(&rows_Z, 1, MPI_INT, r, MPI_COMM_WORLD);
        MPI_Bcast(&cols_Z, 1, MPI_INT, r, MPI_COMM_WORLD);
        if (my_rank != r) all_Z[r].resize(rows_Z, cols_Z);
        MPI_Bcast(all_Z[r].data(), rows_Z * cols_Z, MPI_DOUBLE, r, MPI_COMM_WORLD);

        std::vector<int> idx(rows_Z);
        std::vector<double> w(rows_Z);
        if (my_rank == r) {
            idx = pkg.topology.getGlobalIndices();
            for (int j = 0; j < rows_Z; ++j) w[j] = pkg.pou_weights(j);
        }
        MPI_Bcast(idx.data(), rows_Z, MPI_INT, r, MPI_COMM_WORLD);
        MPI_Bcast(w.data(), rows_Z, MPI_DOUBLE, r, MPI_COMM_WORLD);
        all_R[r] = schwarz2lvl::RestrictionOperator(idx);
        all_D[r].setWeights(w);
        n_max = std::max(n_max, rows_Z);
    }

    // ---- 2) A_00 di riferimento e bound ---------------------------------
    const Eigen::Index m0 = coarse_new.coarseSize();
    Eigen::MatrixXd A00_ref, B;
    double ratio_A00 = 0.0, ratio_bug = 0.0;
    int corner_pairs = 0;
    bool sensitivity_ok = true;
    double kappa = 0.0;
    if (m0 > 0) {
        std::vector<Eigen::Triplet<double>> t, t_abs;
        Eigen::Index row0 = 0;
        for (int r = 0; r < num_ranks; ++r) {
            const auto& idx = all_R[r].getGlobalIndices();
            const auto& w = all_D[r].getWeights();
            for (Eigen::Index a = 0; a < all_Z[r].cols(); ++a)
                for (Eigen::Index j = 0; j < all_Z[r].rows(); ++j) {
                    const double v = all_Z[r](j, a) * w(j);
                    if (v == 0.0) continue;
                    t.emplace_back(row0 + a, idx[j], v);
                    t_abs.emplace_back(row0 + a, idx[j], std::abs(v));
                }
            row0 += all_Z[r].cols();
        }
        Eigen::SparseMatrix<double> R0(m0, num_rows), R0abs(m0, num_rows);
        R0.setFromTriplets(t.begin(), t.end());
        R0abs.setFromTriplets(t_abs.begin(), t_abs.end());
        const Eigen::SparseMatrix<double> Aabs = Eigen::SparseMatrix<double>(A).cwiseAbs();
        A00_ref = Eigen::MatrixXd(R0 * (Eigen::SparseMatrix<double>(A) * Eigen::SparseMatrix<double>(R0.transpose())));
        B = Eigen::MatrixXd(R0abs * (Aabs * Eigen::SparseMatrix<double>(R0abs.transpose())));

        const double gamma = gammaK(static_cast<double>(k_A + n_max + 1));
        ratio_A00 = boundRatio(coarse_new.coarseMatrix(), A00_ref, B, gamma, 0, m0);

        // Angoli: sottodomini j != me, NON vicini di halo, con blocco accoppiato
        // (B > 0) alle mie righe. Sensibilita': azzerarli deve far fallire il bound.
        const auto& nbrs = halo.getNeighborRanks();
        const Eigen::Index off_i = coarse_new.myCoarseOffset();
        const Eigen::Index m_i = coarse_new.myNumModes();
        const auto& mpr = coarse_new.modesPerRank();
        Eigen::MatrixXd A00_bug = coarse_new.coarseMatrix();
        Eigen::Index off_j = 0;
        for (int j = 0; j < num_ranks; ++j) {
            const bool is_nbr = std::find(nbrs.begin(), nbrs.end(), j) != nbrs.end();
            if (j != my_rank && !is_nbr && m_i > 0 && mpr[j] > 0 &&
                B.block(off_i, off_j, m_i, mpr[j]).maxCoeff() > 0.0) {
                ++corner_pairs;
                A00_bug.block(off_i, off_j, m_i, mpr[j]).setZero();
            }
            off_j += mpr[j];
        }
        if (corner_pairs > 0) {
            ratio_bug = boundRatio(A00_bug, A00_ref, B, gamma, off_i, m_i);
            sensitivity_ok = ratio_bug > 1.0;
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A00_ref);
        kappa = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
    }

    // ---- 3) apply ---------------------------------------------------------
    schwarz2lvl::CoarseSpace coarse_old;
    coarse_old.setup(A_ref, all_Z, all_R, all_D);

    schwarz2lvl::VectorType r_global(num_rows);
    for (int i = 0; i < num_rows; ++i) r_global(i) = std::sin(0.013 * i) + 0.002 * i;
    schwarz2lvl::VectorType z_ref = schwarz2lvl::VectorType::Zero(num_rows);
    coarse_old.apply(r_global, z_ref);                               // collettiva

    const auto& my_interior = pkg.topology.getInteriorIndices();
    const Eigen::Index n_int = halo.numInterior();
    schwarz2lvl::VectorType r_owned(n_int), z_owned;
    for (Eigen::Index j = 0; j < n_int; ++j) r_owned(j) = r_global(my_interior[j]);
    coarse_new.apply(r_owned, z_owned);                              // collettiva

    double loc[2] = {0.0, 0.0}, glob[2];
    for (Eigen::Index j = 0; j < n_int; ++j) {
        const double d = z_owned(j) - z_ref(my_interior[j]);
        loc[0] += d * d;
        loc[1] += z_ref(my_interior[j]) * z_ref(my_interior[j]);
    }
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double rel_apply = glob[1] > 0.0 ? std::sqrt(glob[0] / glob[1]) : std::sqrt(glob[0]);
    const double apply_tol = 1e3 * kappa * std::numeric_limits<double>::epsilon() / 2.0;

    const bool ok = ratio_lump <= 1.0 && ratio_A00 <= 1.0 && sensitivity_ok &&
                    coarse_old.coarseSize() == m0 && (m0 == 0 || rel_apply <= apply_tol);

    if (my_rank == 0) {
        std::cout << "m0=" << m0 << "  kappa(A00)=" << std::scientific << kappa
                  << "  rel_diff_apply=" << rel_apply << "  (soglia " << apply_tol << ")" << std::endl;
    }
    for (int r = 0; r < num_ranks; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == my_rank) {
            std::cout << "[rank " << my_rank << "] m_i=" << coarse_new.myNumModes()
                      << " lump(err/bound)=" << std::scientific << ratio_lump
                      << " A00(err/bound)=" << ratio_A00
                      << " angoli=" << corner_pairs
                      << " bug_angoli_rilevabile=" << sensitivity_ok
                      << ((ok) ? "  >> OK" : "  >> MISMATCH") << std::endl;
            std::cout.flush();
        }
    }
    MPI_Finalize();
    return ok ? 0 : 1;
}