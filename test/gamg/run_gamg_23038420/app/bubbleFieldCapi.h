/*---------------------------------------------------------------------------*\
    Bubble field of schwarz2lvl (include/utils/bubble_field.hpp) behind a C interface.

    bubbleFieldCapi.cpp is compiled with the compiler and flags of schwarz2lvl (not with the
    OpenFOAM toolchain) and linked into bubblyPressureFoam, so that OpenFOAM evaluates the bubbles
    with the same machine code as the generator. Compiling bubble_field.hpp with the OpenFOAM
    toolchain of G100 (icpx oneAPI 2021, -O3 -xCORE-AVX512) moved two of 150 bubbles.
\*---------------------------------------------------------------------------*/
#ifndef BUBBLE_FIELD_CAPI_H
#define BUBBLE_FIELD_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

/* New field (unit cube, nx x ny x nz cells, pseudo-time = step); NULL on invalid parameters. */
void* s2lBubbleFieldNew(int dim, int nx, int ny, int nz, int nbubbles, double radius, double cfl,
                        double time, unsigned seed);
/* alpha[c] = gas fraction of cell (i[c], j[c], k[c]), c = 0..n-1. */
void s2lBubbleFieldAlpha(const void* field, int n, const int* i, const int* j, const int* k, double* alpha);
/* Cell (i, j, k) containing the point (x, y, z). */
void s2lBubbleFieldCellOf(const void* field, double x, double y, double z, int* i, int* j, int* k);
void s2lBubbleFieldDelete(void* field);

#ifdef __cplusplus
}
#endif

#endif
