// Verifica DomainDecomposer confrontando, per OGNI rank r in [0, N), il
// pacchetto che produce contro la pipeline esistente e gia' validata
// (SubdomainTopology + RestrictionOperator + PartitionOfUnity +
// OneLevelPreconditioner), su due fronti indipendenti:
//   1) A_ii: pkg.local_matrix.extractLocalSquareBlock() vs OneLevelPreconditioner::getLocalMatrix()
//   2) pesi POU: pkg.pou_weights (dal singolo passaggio globale) vs
//      PartitionOfUnity::computeWeights (algoritmo per-rank gia' validato)
// Serve un solo processo MPI (np=1): DomainDecomposer gira comunque solo
// su rank 0, e qui vogliamo validare TUTTI gli N rank in un colpo solo,
// senza lanciare N processi reali (quello e' compito di Step 3,
// MatrixDistributor, non di questo test).
#include "matrix_market_io.hpp"
#include "global_matrix.hpp"
#include "subdomain_topology.hpp"
#include "restriction_operator.hpp"
#include "partition_of_unity.hpp"
#include "one_level_preconditioner.hpp"
#include "domain_decomposer.hpp"
#include <mpi.h>
#include <iostream>
#include <cmath>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int my_rank = 0, num_procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (num_procs != 1) {
        if (my_rank == 0) std::cerr << "test_domain_decomposer va lanciato con -np 1." << std::endl;
        MPI_Finalize();
        return 1;
    }

    const std::string matrix_path = argc > 1 ? argv[1] : "matrices/nos5.mtx";
    const int N = argc > 2 ? std::atoi(argv[2]) : 10;

    schwarz2lvl::DomainDecomposer decomposer;
    decomposer.decompose(matrix_path, N);

    schwarz2lvl::SparseMatrixWrapper A_ref;
    schwarz2lvl::MatrixMarketIO::readMatrix(matrix_path, A_ref);
    const std::vector<int>& partition_map = decomposer.partitionMap();

    int mismatches = 0;
    for (int r = 0; r < N; ++r) {
        schwarz2lvl::SubdomainTopology topo_ref;
        topo_ref.computeTopology(A_ref, partition_map, r);
        schwarz2lvl::RestrictionOperator R_ref(topo_ref.getGlobalIndices());
        schwarz2lvl::PartitionOfUnity D_ref;
        D_ref.computeWeights(A_ref, partition_map, topo_ref);
        schwarz2lvl::OneLevelPreconditioner prec_ref;
        prec_ref.setup(A_ref, topo_ref, R_ref, D_ref);
        const schwarz2lvl::MatrixType& A_ii_ref = prec_ref.getLocalMatrix();

        const auto& pkg = decomposer.packageForRank(r);
        schwarz2lvl::MatrixType A_ii_new = pkg.local_matrix.extractLocalSquareBlock();

        bool same_shape = (A_ii_ref.rows() == A_ii_new.rows()) &&
                           (A_ii_ref.cols() == A_ii_new.cols()) &&
                           (A_ii_ref.nonZeros() == A_ii_new.nonZeros());
        double max_diff_Aii = 0.0;
        if (same_shape) {
            schwarz2lvl::MatrixType diff = A_ii_ref - A_ii_new;
            for (int k = 0; k < diff.outerSize(); ++k)
                for (schwarz2lvl::MatrixType::InnerIterator it(diff, k); it; ++it)
                    max_diff_Aii = std::max(max_diff_Aii, std::abs(it.value()));
        }

        bool same_size_pou = (D_ref.getWeights().size() == pkg.pou_weights.size());
        double max_diff_pou = 0.0;
        if (same_size_pou) {
            max_diff_pou = (D_ref.getWeights() - pkg.pou_weights).cwiseAbs().maxCoeff();
        }

        bool ok = same_shape && max_diff_Aii == 0.0 && same_size_pou && max_diff_pou < 1e-14;
        if (!ok) mismatches++;

        std::cout << "rank " << r << ": n_i=" << pkg.local_matrix.rows()
                  << " same_shape=" << same_shape
                  << " max_diff_Aii=" << std::scientific << max_diff_Aii
                  << " max_diff_pou=" << max_diff_pou
                  << (ok ? "  >> OK" : "  >> MISMATCH") << std::endl;
    }

    std::cout << "N=" << N << "  mismatches=" << mismatches
              << (mismatches == 0 ? "  ALL OK" : "  FAILURES FOUND") << std::endl;

    MPI_Finalize();
    return mismatches == 0 ? 0 : 1;
}