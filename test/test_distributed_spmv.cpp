// Verifica DistributedSpMV confrontando y_owned = (A x)|_interior, calcolato
// in modo distribuito (blocco locale + halo exchange), contro A * x_global
// (matrice intera, riletta da ogni rank SOLO per costruire il riferimento).
//
// Criterio di accettazione: bound componente per componente di Higham per
// un prodotto scalare di k termini in aritmetica floating point. Per la
// riga i con k_i non-zeri, OGNI ordine di somma soddisfa
//     |fl(y_i) - y_i| <= gamma_{k_i} * (|A| |x|)_i,   gamma_k = k u / (1 - k u)
// quindi due ordini diversi (colonne globali crescenti nel riferimento,
// colonne LOCALI -- interior prima, poi boundary -- in DistributedSpMV)
// differiscono al piu' di 2 gamma_{k_i} (|A| |x|)_i. La scala corretta e'
// (|A||x|)_i, NON |y_i|: con cancellazione (y_i piccolo, addendi grandi)
// l'errore di arrotondamento resta proporzionale agli addendi.
//
// Controllo di sensibilita': il test verifica anche che lo stesso criterio
// BOCCI la versione con halo dimenticata (boundary di x lasciato a zero),
// cosi' una tolleranza troppo larga non puo' mascherare un bug reale.
#include "matrix_market_io.hpp"
#include "global_matrix.hpp"
#include "domain_decomposer.hpp"
#include "matrix_distributor.hpp"
#include "halo_exchange.hpp"
#include "distributed_spmv.hpp"
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <limits>

namespace {
// Rapporto massimo errore / bound sulle righe interior di questo rank:
// <= 1 significa "differenza spiegata interamente dall'arrotondamento".
double maxBoundRatio(const schwarz2lvl::VectorType& y_test,
                     const schwarz2lvl::VectorType& y_ref,
                     const schwarz2lvl::VectorType& abs_Ax,
                     const std::vector<int>& row_nnz,
                     const std::vector<int>& my_interior) {
    const double u = std::numeric_limits<double>::epsilon() / 2.0;   // unit roundoff 2^-53
    double worst = 0.0;
    for (size_t j = 0; j < my_interior.size(); ++j) {
        const int gi = my_interior[j];
        const double k = static_cast<double>(row_nnz[gi]);
        const double gamma_k = k * u / (1.0 - k * u);
        const double bound = 2.0 * gamma_k * abs_Ax(gi);
        const double err = std::abs(y_test(static_cast<Eigen::Index>(j)) - y_ref(gi));
        if (err == 0.0) continue;
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

    // --- Percorso NUOVO --------------------------------------------------
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

    // --- Vettore di test globale, valori distinti e non banali -----------
    schwarz2lvl::VectorType x_global(num_rows);
    for (int i = 0; i < num_rows; ++i) x_global(i) = std::sin(0.017 * i) + 0.001 * i;

    const Eigen::Index num_interior = halo.numInterior();
    const auto& my_interior = pkg.topology.getInteriorIndices();
    schwarz2lvl::VectorType x_owned(num_interior);
    for (Eigen::Index j = 0; j < num_interior; ++j) x_owned(j) = x_global(my_interior[j]);

    schwarz2lvl::VectorType y_owned;
    spmv.apply(x_owned, y_owned);

    // --- Riferimento: rilettura indipendente, matrice ancora intera -----
    schwarz2lvl::SparseMatrixWrapper A_ref;
    schwarz2lvl::MatrixMarketIO::readMatrix(matrix_path, A_ref);
    const schwarz2lvl::MatrixType& A = A_ref.getMatrix();
    const schwarz2lvl::VectorType y_global_ref = A * x_global;
    const schwarz2lvl::VectorType abs_Ax = A.cwiseAbs() * x_global.cwiseAbs();
    std::vector<int> row_nnz(num_rows);
    for (int i = 0; i < num_rows; ++i) row_nnz[i] = static_cast<int>(A.innerVector(i).nonZeros());

    const double ratio = maxBoundRatio(y_owned, y_global_ref, abs_Ax, row_nnz, my_interior);
    double max_err = 0.0;
    for (Eigen::Index j = 0; j < num_interior; ++j)
        max_err = std::max(max_err, std::abs(y_owned(j) - y_global_ref(my_interior[j])));
    const bool ok = ratio <= 1.0;

    // --- Sensibilita': halo "dimenticata" (boundary di x a zero) ---------
    // Stessa matrice che usa DistributedSpMV (righe interior, colonne
    // locali), ma senza exchange: deve essere BOCCIATA dal criterio.
    const Eigen::Index n_boundary = pkg.local_matrix.rows() - num_interior;
    bool sensitivity_ok = true;   // banalmente vero se non ho boundary (es. N=1)
    if (n_boundary > 0) {
        const schwarz2lvl::MatrixType A_int = pkg.local_matrix.extractLocalSquareBlock().topRows(num_interior);
        schwarz2lvl::VectorType x_no_halo = schwarz2lvl::VectorType::Zero(pkg.local_matrix.rows());
        x_no_halo.head(num_interior) = x_owned;
        const schwarz2lvl::VectorType y_bug = A_int * x_no_halo;
        sensitivity_ok = maxBoundRatio(y_bug, y_global_ref, abs_Ax, row_nnz, my_interior) > 1.0;
    }

    for (int r = 0; r < num_ranks; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == my_rank) {
            std::cout << "[rank " << my_rank << "] num_interior=" << num_interior
                      << " max_err_abs=" << std::scientific << max_err
                      << " max(err/bound)=" << ratio
                      << " bug_rilevabile=" << sensitivity_ok
                      << ((ok && sensitivity_ok) ? "  >> OK" : "  >> MISMATCH") << std::endl;
            std::cout.flush();
        }
    }
    MPI_Finalize();
    return (ok && sensitivity_ok) ? 0 : 1;
}