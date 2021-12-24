#include "ZOnePermanentCalculatorDFE.h"


#ifndef CPYTHON
#include <tbb/tbb.h>
#endif


namespace pic {

#define ROWCOL(m, r, c) ((m[r] & (1<<c)) != 0)

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void ZOnePermanentCalculatorDFE(std::vector<uint64_t>& matrix_mtx, std::vector<uint64_t>& perm, int isSim, int isGray, int isRows, int useGlynn, int useDual)
{
    size_t rows = matrix_mtx.size();
    size_t cols = rows;
    if (rows == 0) { perm[0] = 1; return; }
    else if (rows == 1) { perm[0] = matrix_mtx[0]; return; }
    else if (rows == 2) { perm[0] = ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 1) + ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 0); return; } //ad + bc
    else if (rows == 3 && useDual) {
      perm[0] = ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 2) +
                ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 0) +
                ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 1) +
                ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 0) +
                ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 2) +
                ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 1);
     return; } //aei + bfg + cdh + ceg + bdi + afh
    while (matrix_mtx.size() < MAX_SIM_DIM) {
      matrix_mtx.push_back(1UL << matrix_mtx.size());
    }
    calcPermanentZOneDFE( (const uint64_t*)matrix_mtx.data(), rows, cols, perm.data(), isSim, isGray, isRows, useGlynn, useDual);

    return;
}




}
