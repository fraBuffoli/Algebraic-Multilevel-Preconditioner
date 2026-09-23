#include "matrix_distributor.hpp"
#include "local_index_map.hpp"
#include <mpi.h>
#include <stdexcept>
#include <vector>

namespace schwarz2lvl {

// tags MPI for packaging the subdomain data. Each tag is unique to avoid message collisions.
constexpr int kTagHeader = 200;
constexpr int kTagInterior = 201;
constexpr int kTagBoundary = 202;
constexpr int kTagTripletRows = 203;
constexpr int kTagTripletCols = 204;
constexpr int kTagTripletVals = 205;
constexpr int kTagPouWeights = 206;

void sendPackage(const SubdomainPackage& pkg, int dest_rank) {
    const auto& interior = pkg.topology.getInteriorIndices();
    const auto& boundary = pkg.topology.getBoundaryIndices();
    const MatrixType& mat = pkg.local_matrix.raw();

    std::vector<int> rows, cols;
    std::vector<double> vals;
    rows.reserve(static_cast<size_t>(mat.nonZeros()));
    cols.reserve(static_cast<size_t>(mat.nonZeros()));
    vals.reserve(static_cast<size_t>(mat.nonZeros()));
    for (Eigen::Index local_row = 0; local_row < mat.rows(); ++local_row) {
        for (MatrixType::InnerIterator it(mat, local_row); it; ++it) {
            rows.push_back(static_cast<int>(local_row));
            cols.push_back(static_cast<int>(it.col()));
            vals.push_back(it.value());
        }
    }

    const int header[4] = {
        static_cast<int>(interior.size()),
        static_cast<int>(boundary.size()),
        static_cast<int>(rows.size()),
        static_cast<int>(pkg.local_matrix.globalCols())
    };

    MPI_Send(header, 4, MPI_INT, dest_rank, kTagHeader, MPI_COMM_WORLD);
    MPI_Send(interior.data(), header[0], MPI_INT, dest_rank, kTagInterior, MPI_COMM_WORLD);
    MPI_Send(boundary.data(), header[1], MPI_INT, dest_rank, kTagBoundary, MPI_COMM_WORLD);
    MPI_Send(rows.data(), header[2], MPI_INT, dest_rank, kTagTripletRows, MPI_COMM_WORLD);
    MPI_Send(cols.data(), header[2], MPI_INT, dest_rank, kTagTripletCols, MPI_COMM_WORLD);
    MPI_Send(vals.data(), header[2], MPI_DOUBLE, dest_rank, kTagTripletVals, MPI_COMM_WORLD);
    MPI_Send(pkg.pou_weights.data(), header[0] + header[1], MPI_DOUBLE, dest_rank, kTagPouWeights, MPI_COMM_WORLD);
}

SubdomainPackage recvPackage(int source_rank) {
    int header[4];
    MPI_Recv(header, 4, MPI_INT, source_rank, kTagHeader, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    const int num_interior = header[0];
    const int num_boundary = header[1];
    const int nnz = header[2];
    const int n_global = header[3];

    std::vector<int> interior(num_interior), boundary(num_boundary);
    MPI_Recv(interior.data(), num_interior, MPI_INT, source_rank, kTagInterior, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(boundary.data(), num_boundary, MPI_INT, source_rank, kTagBoundary, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<int> rows(nnz), cols(nnz);
    std::vector<double> vals(nnz);
    MPI_Recv(rows.data(), nnz, MPI_INT, source_rank, kTagTripletRows, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(cols.data(), nnz, MPI_INT, source_rank, kTagTripletCols, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(vals.data(), nnz, MPI_DOUBLE, source_rank, kTagTripletVals, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    SubdomainPackage pkg;
    pkg.topology.setIndices(interior, boundary);

    LocalIndexMap index_map;
    index_map.build(pkg.topology.getGlobalIndices(), static_cast<Eigen::Index>(num_interior));

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<size_t>(nnz));
    for (int k = 0; k < nnz; ++k) triplets.emplace_back(rows[k], cols[k], vals[k]);
    pkg.local_matrix.build(index_map, n_global, triplets);

    pkg.pou_weights.resize(num_interior + num_boundary);
    MPI_Recv(pkg.pou_weights.data(), num_interior + num_boundary, MPI_DOUBLE,
             source_rank, kTagPouWeights, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    return pkg;
}

SubdomainPackage MatrixDistributor::distribute(const DomainDecomposer* decomposer) {
    int my_rank = 0, num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (my_rank == 0) {
        if (decomposer == nullptr) {
            throw std::runtime_error("MatrixDistributor::distribute Error: rank 0 called "
                                      "without a decomposed DomainDecomposer.");
        }
        for (int r = 1; r < num_ranks; ++r) {
            sendPackage(decomposer->packageForRank(r), r);
        }
        return decomposer->packageForRank(0);
    }

    return recvPackage(0);
}

} // namespace schwarz2lvl