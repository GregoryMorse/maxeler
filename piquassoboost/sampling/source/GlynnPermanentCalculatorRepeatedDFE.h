
#ifndef GlynnPermanentCalculatorRepeatedDFE_H
#define GlynnPermanentCalculatorRepeatedDFE_H

#include "matrix.h"
#include "GlynnPermanentCalculatorDFE.h"
#include "PicState.h"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif

// the maximal dimension of matrix to be ported to FPGA for permanent calculation
#define MAX_FPGA_DIM 8*5
#define MAX_SINGLE_FPGA_DIM 4*10
#define BASEKERNPOW2 2

namespace pic {

void
GlynnPermanentCalculatorRepeatedMulti_DFE(matrix& matrix_init, PicState_int64& input_state,
    PicState_int64& output_state, Complex16& perm, int useDual);

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void GlynnPermanentCalculatorRepeated_DFE(matrix& matrix_mtx, PicState_int64& input_state,
    PicState_int64& output_state, Complex16& perm, int useDual);

}

typedef void(*CALCPERMGLYNNREPDFE)(const pic::ComplexFix16**, const long double*, const uint64_t, const uint64_t, const unsigned char*,
  const uint8_t*, const uint8_t, const uint8_t, const uint64_t*, const uint64_t, const uint8_t, pic::Complex16*);
typedef void(*INITPERMGLYNNREPDFE)(void);
typedef void(*FREEPERMGLYNNREPDFE)(void);
extern "C" CALCPERMGLYNNREPDFE calcPermanentGlynnRepDFE; 
extern "C" INITPERMGLYNNREPDFE initializeRep_DFE; 
extern "C" FREEPERMGLYNNREPDFE releiveRep_DFE; 


#endif
