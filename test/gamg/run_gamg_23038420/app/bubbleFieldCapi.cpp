/*
 * C interface to schwarz2lvl::utils::BubbleField (see bubbleFieldCapi.h). Compiled with the
 * compiler and flags of schwarz2lvl, e.g.
 *     mpicxx -std=c++17 -O3 -march=skylake-avx512 -DNDEBUG -fPIC -I../../include/utils -c bubbleFieldCapi.cpp
 * before `wmake` (Make/options links bubbleFieldCapi.o).
 */
#include "bubbleFieldCapi.h"

#include "bubble_field.hpp"

using schwarz2lvl::utils::BubbleField;
using schwarz2lvl::utils::BubbleFieldParams;

extern "C" {

void* s2lBubbleFieldNew(int dim, int nx, int ny, int nz, int nbubbles, double radius, double cfl, double time,
                        unsigned seed)
{
    try {
        BubbleFieldParams p;
        p.dim = dim;
        p.nx = nx;
        p.ny = ny;
        p.nz = nz;
        p.nbubbles = nbubbles;
        p.radius = radius;
        p.cfl = cfl;
        p.time = time;
        p.seed = seed;
        return new BubbleField(p);
    } catch (...) {
        return nullptr;
    }
}

void s2lBubbleFieldAlpha(const void* field, int n, const int* i, const int* j, const int* k, double* alpha)
{
    const BubbleField& f = *static_cast<const BubbleField*>(field);
    for (int c = 0; c < n; ++c) alpha[c] = f.alpha(i[c], j[c], k[c]);
}

void s2lBubbleFieldCellOf(const void* field, double x, double y, double z, int* i, int* j, int* k)
{
    static_cast<const BubbleField*>(field)->cellOf(x, y, z, *i, *j, *k);
}

void s2lBubbleFieldDelete(void* field) { delete static_cast<BubbleField*>(field); }

} // extern "C"
