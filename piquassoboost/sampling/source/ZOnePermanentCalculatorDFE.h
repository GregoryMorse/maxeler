
#ifndef ZOnePermanentCalculatorDFE_H
#define ZOnePermanentCalculatorDFE_H

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif

// the maximal dimension of matrix to be ported to FPGA for permanent calculation
#define MAX_FPGA_DIM 24
#define MAX_SINGLE_FPGA_DIM 20
//#define PermanentZOneRowsGrayDFE_MTXSIZE (20)

extern "C"
{
    void calcPermanentZOneSIM(const uint64_t* mtx_data, const uint64_t rows, const uint64_t cols, uint64_t* perm);
    void initialize_ZOne_SIM(int isGray, int isRows, int useGlynn, int useDual);
    void releive_ZOne_SIM();
    //void calcPermanentZOneDFE(const uint64_t* mtx_data, const uint64_t rows, const uint64_t cols, uint64_t* perm);
    //void initialize_ZOne_DFE(int isGray, int isRows, int useGlynn, int useDual);
    //void releive_ZOne_DFE();
}



namespace pic {

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void ZOnePermanentCalculatorDFE(std::vector<uint64_t>& matrix_mtx, std::vector<uint64_t>& perm, int isSim, int isGray, int isRows, int useGlynn, int useDual);


}



#endif
