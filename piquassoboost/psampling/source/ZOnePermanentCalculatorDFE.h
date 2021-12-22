
#ifndef ZOnePermanentCalculatorDFE_H
#define ZOnePermanentCalculatorDFE_H

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif

// the maximal dimension of matrix to be ported to FPGA for permanent calculation
#define MAX_FPGA_DIM 40
#define MAX_SINGLE_FPGA_DIM 28
#define MAX_SIM_DIM 20
//#define PermanentZOneRowsGrayDFE_MTXSIZE (20)

extern "C"
{
    void calcPermanentZOneDFE(const uint64_t* mtx_data, const uint64_t rows, const uint64_t cols, uint64_t* perm);
    void calcPermanentZOneGrayDFE(const uint64_t* mtx_data, const uint64_t rows, const uint64_t cols, uint64_t* perm);
    void calcPermanentZOneRowsDFE(const uint64_t* mtx_data, const uint64_t rows, const uint64_t cols, uint64_t* perm);
    void calcPermanentZOneRowsGrayDFE(const uint64_t* mtx_data, const uint64_t rows, const uint64_t cols, uint64_t* perm);
    void initialize_ZOne_DFE();
    void releive_ZOne_DFE();
}



namespace pic {

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void ZOnePermanentCalculatorDFE(std::vector<uint64_t>& matrix_mtx, std::vector<uint64_t>& perm, int isSim, int isGray, int isRows);


}



#endif
