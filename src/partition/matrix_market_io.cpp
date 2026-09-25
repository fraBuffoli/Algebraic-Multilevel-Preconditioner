/**
 * @file matrix_market_io.cpp
 * @brief Implementation of MatrixMarketIO.
 *
 * The whole file is read into memory and parsed with strtoll/strtod, which
 * is an order of magnitude faster than iostream-based parsing for the
 * multi-GB files produced by large CFD cases.
 */
#include "matrix_market_io.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace schwarz2lvl::part {

namespace {

/// @brief Reads a whole file into a string.
std::string slurp(const std::string& path)
{
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open file '" + path + "'");
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string buf(static_cast<std::size_t>(size), '\0');
    const std::size_t got = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    if (got != buf.size()) throw std::runtime_error("short read on '" + path + "'");
    return buf;
}

/// @brief Lower-case copy of a string.
std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

/// @brief Parsed Matrix Market banner.
struct Banner {
    std::string format;   ///< coordinate | array
    std::string field;    ///< real | integer | pattern | complex
    std::string symmetry; ///< general | symmetric | skew-symmetric | hermitian
};

/**
 * @brief Parses the banner and skips comments.
 * @param buf File content.
 * @param pos In: 0. Out: position of the size line.
 */
Banner parseBanner(const std::string& buf, std::size_t& pos)
{
    const std::size_t eol = buf.find('\n');
    std::istringstream is(buf.substr(0, eol));
    std::string mm, object;
    Banner b;
    is >> mm >> object >> b.format >> b.field >> b.symmetry;
    if (lower(mm) != "%%matrixmarket" || lower(object) != "matrix")
        throw std::runtime_error("not a Matrix Market matrix file");
    b.format = lower(b.format);
    b.field = lower(b.field);
    b.symmetry = lower(b.symmetry);
    if (b.field == "complex" || b.symmetry == "hermitian")
        throw std::runtime_error("complex Matrix Market files are not supported (real arithmetic only)");
    pos = (eol == std::string::npos) ? buf.size() : eol + 1;
    // skip comment and empty lines
    while (pos < buf.size() && (buf[pos] == '%' || buf[pos] == '\n' || buf[pos] == '\r')) {
        const std::size_t e = buf.find('\n', pos);
        pos = (e == std::string::npos) ? buf.size() : e + 1;
    }
    return b;
}

} // namespace

CsrMatrix MatrixMarketIO::readMatrix(const std::string& path)
{
    const std::string buf = slurp(path);
    std::size_t pos = 0;
    const Banner b = parseBanner(buf, pos);
    if (b.format != "coordinate") throw std::runtime_error("readMatrix: only coordinate format is supported");

    const char* p = buf.c_str() + pos;
    char* end = nullptr;
    const long long m = std::strtoll(p, &end, 10);
    p = end;
    const long long n = std::strtoll(p, &end, 10);
    p = end;
    const long long nz = std::strtoll(p, &end, 10);
    p = end;
    if (m <= 0 || n <= 0 || nz < 0) throw std::runtime_error("readMatrix: invalid size line");

    const bool pattern = (b.field == "pattern");
    const bool sym = (b.symmetry == "symmetric");
    const bool skew = (b.symmetry == "skew-symmetric");

    std::vector<core::GlobalIndex> rows, cols;
    std::vector<core::Scalar> vals;
    const std::size_t cap = static_cast<std::size_t>(nz) * ((sym || skew) ? 2 : 1);
    rows.reserve(cap);
    cols.reserve(cap);
    vals.reserve(cap);
    for (long long k = 0; k < nz; ++k) {
        const long long i = std::strtoll(p, &end, 10);
        if (end == p) throw std::runtime_error("readMatrix: unexpected end of file");
        p = end;
        const long long j = std::strtoll(p, &end, 10);
        p = end;
        double v = 1.0;
        if (!pattern) {
            v = std::strtod(p, &end);
            p = end;
        }
        rows.push_back(i - 1);
        cols.push_back(j - 1);
        vals.push_back(v);
        if ((sym || skew) && i != j) {
            rows.push_back(j - 1);
            cols.push_back(i - 1);
            vals.push_back(skew ? -v : v);
        }
    }
    CsrMatrix A = CsrMatrix::fromCoo(m, n, rows, cols, vals);
    A.symmetric = sym;
    return A;
}

std::vector<core::Scalar> MatrixMarketIO::readVector(const std::string& path)
{
    const std::string buf = slurp(path);
    std::size_t pos = 0;
    const Banner b = parseBanner(buf, pos);
    const char* p = buf.c_str() + pos;
    char* end = nullptr;
    const long long m = std::strtoll(p, &end, 10);
    p = end;
    const long long n = std::strtoll(p, &end, 10);
    p = end;
    if (b.format == "array") {
        if (n != 1) throw std::runtime_error("readVector: array must have one column");
        std::vector<core::Scalar> v(static_cast<std::size_t>(m));
        for (long long i = 0; i < m; ++i) {
            v[static_cast<std::size_t>(i)] = std::strtod(p, &end);
            if (end == p) throw std::runtime_error("readVector: unexpected end of file");
            p = end;
        }
        return v;
    }
    const long long nz = std::strtoll(p, &end, 10);
    p = end;
    std::vector<core::Scalar> v(static_cast<std::size_t>(m), 0.0);
    for (long long k = 0; k < nz; ++k) {
        const long long i = std::strtoll(p, &end, 10);
        p = end;
        std::strtoll(p, &end, 10);
        p = end;
        v[static_cast<std::size_t>(i - 1)] += std::strtod(p, &end);
        p = end;
    }
    return v;
}

void MatrixMarketIO::writeMatrix(const std::string& path, const CsrMatrix& A)
{
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write '" + path + "'");
    std::fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
    std::fprintf(f, "%lld %lld %lld\n", static_cast<long long>(A.nrows), static_cast<long long>(A.ncols),
                 static_cast<long long>(A.nnz()));
    for (core::GlobalIndex i = 0; i < A.nrows; ++i)
        for (auto k = A.row_ptr[static_cast<std::size_t>(i)]; k < A.row_ptr[static_cast<std::size_t>(i) + 1]; ++k)
            std::fprintf(f, "%lld %lld %.17g\n", static_cast<long long>(i + 1),
                         static_cast<long long>(A.col[static_cast<std::size_t>(k)] + 1),
                         A.val[static_cast<std::size_t>(k)]);
    std::fclose(f);
}

void MatrixMarketIO::writeVector(const std::string& path, const std::vector<core::Scalar>& v)
{
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write '" + path + "'");
    std::fprintf(f, "%%%%MatrixMarket matrix array real general\n%zu 1\n", v.size());
    for (core::Scalar x : v) std::fprintf(f, "%.17g\n", x);
    std::fclose(f);
}

} // namespace schwarz2lvl::part
