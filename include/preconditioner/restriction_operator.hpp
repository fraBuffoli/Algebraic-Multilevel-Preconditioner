#ifndef RESTRICTION_OPERATOR_HPP
#define RESTRICTION_OPERATOR_HPP

#include "global_matrix.hpp"
#include <vector>

namespace schwarz2lvl {

/**
 * @class RestrictionOperator
 * @brief Manages the algebraic restriction (R_i) and prolongation (R_i^T) for a single subdomain.
 * 
 * This class maps global vector indices to local subdomain indices. It avoids 
 * explicit matrix-vector multiplications by using direct index-based gathering and scattering.
 */    
class RestrictionOperator {
public:
    /**
     * @brief Default constructor creating an uninitialized operator.
     */
    RestrictionOperator() = default;

    /**
     * @brief Constructs the operator with a specific set of global indices.
     * @param global_indices Ordered list of global indices belonging to this subdomain.
     */
    explicit RestrictionOperator(const std::vector<int>& global_indices) 
        : global_indices_(global_indices) {};

    /**
     * @brief Updates the global indices mapping for this subdomain.
     * @param global_indices Ordered list of global indices.
     */
    void setIndices(const std::vector<int>& global_indices){
        global_indices_ = global_indices;
    };

    /**
     * @brief Returns the local size of the subdomain (n_i).
     * @return Number of local unknowns.
     */
    Eigen::Index localSize() const {
        return static_cast<Eigen::Index>(global_indices_.size());       
    };

    /**
     * @brief Restricts a global vector to the local subdomain: v_local = R_i * v_global.
     * 
     * Extracts values from the global vector at the positions specified by the internal index map.
     * 
     * @param g_vec Input global vector (size n).
     * @param l_vec Output local vector (resized to n_i).
     */
    void apply(const VectorType& g_vec, VectorType& l_vec) const;

    /**
     * @brief Prolongs a local vector into a global vector: v_global = v_global + (R_i^T * v_local).
     * 
     * Scatters and accumulates local subdomain corrections into the global vector using accumulation (+=).
     * 
     * @param l_vec Input local vector (size n_i).
     * @param g_vec Output global vector (size n) where local values are added.
     */
    void applyTranspose(const VectorType& l_vec, VectorType& g_vec) const;

    /**
     * @brief Direct read access to the underlying global indices list.
     * @return Const reference to the std::vector<int> of global indices.
     */
    const std::vector<int>& getGlobalIndices() const {
        return global_indices_;
    };

private:
    std::vector<int> global_indices_; ///< Map where local index 'j' corresponds to global index 'global_indices_[j]'
};

} // namespace schwarz2lvl

#endif // RESTRICTION_OPERATOR_HPP
