#ifndef MATRIX_DISTRIBUTOR_HPP
#define MATRIX_DISTRIBUTOR_HPP

#include "domain_decomposer.hpp"

namespace schwarz2lvl {

/**
 * @class MatrixDistributor
 * @brief Distributes the subdomain topology from rank 0 to all other ranks.
 */
class MatrixDistributor {
public:
    /**
     * @brief Distributes the subdomain topology from rank 0 to all other ranks.
     * @param decomposer Pointer to the DomainDecomposer instance on rank 0.
     * @return A SubdomainPackage containing the local topology and matrix for the calling rank.
     */
    static SubdomainPackage distribute(const DomainDecomposer* decomposer);
};

} // namespace schwarz2lvl
#endif // MATRIX_DISTRIBUTOR_HPP