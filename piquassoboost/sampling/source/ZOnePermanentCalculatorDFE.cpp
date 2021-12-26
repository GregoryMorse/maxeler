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
    else if (rows == 3 && !isRows && (useDual || useGlynn)) {
      perm[0] = ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 2) +
                ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 0) +
                ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 1) +
                ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 0) +
                ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 2) +
                ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 1);
     return; } //aei + bfg + cdh + ceg + bdi + afh
    else if (rows == 4 && useDual && !isRows && useGlynn) { //list(itertools.permutations([0,1,2,3], 4))
      perm[0] = ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 2) * ROWCOL(matrix_mtx, 3, 3) +
                ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 3) * ROWCOL(matrix_mtx, 3, 2) +
                ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 1) * ROWCOL(matrix_mtx, 3, 3) +
                ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 3) * ROWCOL(matrix_mtx, 3, 1) +
                ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 3) * ROWCOL(matrix_mtx, 2, 1) * ROWCOL(matrix_mtx, 3, 2) +
                ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 3) * ROWCOL(matrix_mtx, 2, 2) * ROWCOL(matrix_mtx, 3, 1) +
                ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 2) * ROWCOL(matrix_mtx, 3, 3) +
                ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 3) * ROWCOL(matrix_mtx, 3, 2) +
                ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 0) * ROWCOL(matrix_mtx, 3, 3) +
                ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 3) * ROWCOL(matrix_mtx, 3, 0) +
                ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 3) * ROWCOL(matrix_mtx, 2, 0) * ROWCOL(matrix_mtx, 3, 2) +
                ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 3) * ROWCOL(matrix_mtx, 2, 2) * ROWCOL(matrix_mtx, 3, 0) +
                ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 1) * ROWCOL(matrix_mtx, 3, 3) +
                ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 3) * ROWCOL(matrix_mtx, 3, 1) +
                ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 0) * ROWCOL(matrix_mtx, 3, 3) +
                ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 3) * ROWCOL(matrix_mtx, 3, 0) +
                ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 3) * ROWCOL(matrix_mtx, 2, 0) * ROWCOL(matrix_mtx, 3, 1) +
                ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 3) * ROWCOL(matrix_mtx, 2, 1) * ROWCOL(matrix_mtx, 3, 0) +
                ROWCOL(matrix_mtx, 0, 3) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 1) * ROWCOL(matrix_mtx, 3, 2) +
                ROWCOL(matrix_mtx, 0, 3) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 2) * ROWCOL(matrix_mtx, 3, 1) +
                ROWCOL(matrix_mtx, 0, 3) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 0) * ROWCOL(matrix_mtx, 3, 2) +
                ROWCOL(matrix_mtx, 0, 3) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 2) * ROWCOL(matrix_mtx, 3, 0) +
                ROWCOL(matrix_mtx, 0, 3) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 0) * ROWCOL(matrix_mtx, 3, 1) +
                ROWCOL(matrix_mtx, 0, 3) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 1) * ROWCOL(matrix_mtx, 3, 0);
      return; }
    size_t max_dim = useDual ? MAX_FPGA_DIM : MAX_SINGLE_FPGA_DIM;
    while (matrix_mtx.size() < max_dim) {
      matrix_mtx.push_back(1UL << matrix_mtx.size());
    }
    if (isSim) calcPermanentZOneSIM( (const uint64_t*)matrix_mtx.data(), rows, cols, perm.data());
    //else calcPermanentZOneDFE( (const uint64_t*)matrix_mtx.data(), rows, cols, perm.data());

    return;
}




}
