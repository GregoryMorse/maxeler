
#ifndef ZOnePermanentCalculatorDFE_H
#define ZOnePermanentCalculatorDFE_H

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif

// the maximal dimension of matrix to be ported to FPGA for permanent calculation
#define ZONE_MAX_FPGA_DIM 3*8
#define ZONE_MAX_SINGLE_FPGA_DIM 5*4
//#define PermanentZOneRowsGrayDFE_MTXSIZE (20)

typedef void(*CALCPERMDFE)(const uint64_t*, const uint64_t, const uint64_t, uint64_t*);
typedef void(*INITPERMDFE)(int,int,int,int);
typedef void(*FREEPERMDFE)(void);
extern "C" CALCPERMDFE calcPermanentZOneDFE; 
extern "C" INITPERMDFE initialize_ZOne_DFE; 
extern "C" FREEPERMDFE releive_ZOne_DFE; 

namespace pic {

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void ZOnePermanentCalculatorDFE(std::vector<uint64_t>& matrix_mtx, std::vector<uint64_t>& perm, int isSim, int isGray, int isRows, int useGlynn, int useDual);


}



#endif
