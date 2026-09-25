/**
 * @file csr_matrix.cpp
 * @brief Implementation of CsrMatrix.
 */
#include "csr_matrix.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace schwarz2lvl::part{

CsrMatrix CsrMatrix::fromCoo(core::GlobalIndex nrows, core::GlobalIndex ncols, const std::vector<core::GlobalIndex>& rows,
                             const std::vector<core::GlobalIndex>& cols, const std::vector<core::Scalar>& vals){
    if (rows.size() != cols.size() || rows.size() != vals.size())
        throw std::invalid_argument("CsrMatrix::fromCoo: inconsistent coordinates arrays");
    
    CsrMatrix A;
    A.nrows = nrows;
    A.ncols = ncols;
    const std::size_t nz = vals.size();

    std::vector<core::GlobalIndex> ptr(static_cast<std::size_t>(nrows)+1,0);
    for (std::size_t k = 0; k<nz; ++k){
        if(rows[k]<0 || rows[k]>=nrows || cols[k]<0 || cols[k] >=ncols)
            throw std::out_of_range("CsrMatrix::FromCoo: index out of range");
        ++ptr[static_cast<std::size_t>(rows[k])+1];
    }
    std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());
    std::vector<core::GlobalIndex> c(nz);
    std::vector<core::Scalar> v(nz);
    {
        std::vector<std::int64_t> next(ptr.begin(), ptr.end() - 1);
        for (std::size_t k = 0; k < nz; ++k) {
            const auto p = next[static_cast<std::size_t>(rows[k])]++;
            c[static_cast<std::size_t>(p)] = cols[k];
            v[static_cast<std::size_t>(p)] = vals[k];
        }
    }

    A.row_ptr.assign(static_cast<std::size_t>(nrows) + 1, 0);
    A.col.reserve(nz);
    A.val.reserve(nz);
    std::vector<std::size_t> perm;
    for (core::GlobalIndex i = 0; i < nrows; ++i) {
        const auto b = static_cast<std::size_t>(ptr[static_cast<std::size_t>(i)]);
        const auto e = static_cast<std::size_t>(ptr[static_cast<std::size_t>(i) + 1]);
        perm.resize(e - b);
        std::iota(perm.begin(), perm.end(), b);
        std::sort(perm.begin(), perm.end(), [&](std::size_t x, std::size_t y) { return c[x] < c[y]; });
        core::GlobalIndex last = -1;
        for (std::size_t q : perm) {
            if (c[q] == last) {
                A.val.back() += v[q];
            } else {
                A.col.push_back(c[q]);
                A.val.push_back(v[q]);
                last = c[q];
            }
        }
        A.row_ptr[static_cast<std::size_t>(i) + 1] = static_cast<std::int64_t>(A.col.size());
    }
    return A;
}

void CsrMatrix::multiply(const std::vector<core::Scalar>& x, std::vector<core::Scalar>& y) const
{
    y.assign(static_cast<std::size_t>(nrows), 0.0);
    for (core::GlobalIndex i = 0; i < nrows; ++i) {
        core::Scalar s = 0.0;
        for (auto k = row_ptr[static_cast<std::size_t>(i)]; k < row_ptr[static_cast<std::size_t>(i) + 1]; ++k)
            s += val[static_cast<std::size_t>(k)] * x[static_cast<std::size_t>(col[static_cast<std::size_t>(k)])];
        y[static_cast<std::size_t>(i)] = s;
    }
}

} // namespace schwarz2lvl::part