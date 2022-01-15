
#ifndef GlynnPermanentCalculatorDFE_H
#define GlynnPermanentCalculatorDFE_H

#include "matrix.h"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif

// the maximal dimension of matrix to be ported to FPGA for permanent calculation
#define MAX_FPGA_DIM 8*2
#define MAX_SINGLE_FPGA_DIM 4*4

typedef void(*CALCPERMGLYNNDFE)(const pic::Complex16*, const double*, const uint64_t, const uint64_t, pic::Complex16*);
typedef void(*INITPERMGLYNNDFE)(int);
typedef void(*FREEPERMGLYNNDFE)(void);
extern "C" CALCPERMGLYNNDFE calcPermanentGlynnDFE; 
extern "C" INITPERMGLYNNDFE initialize_DFE; 
extern "C" FREEPERMGLYNNDFE releive_DFE; 

namespace pic {

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void GlynnPermanentCalculator_DFE(matrix& matrix_mtx, Complex16& perm, int useDual);

}



#endif
