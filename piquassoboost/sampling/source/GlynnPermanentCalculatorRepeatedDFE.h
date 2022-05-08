
#ifndef GlynnPermanentCalculatorRepeatedDFE_H
#define GlynnPermanentCalculatorRepeatedDFE_H

#include "matrix.h"
#include "GlynnPermanentCalculatorDFE.h"
#include "PicState.h"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif

namespace pic {

void
GlynnPermanentCalculatorRepeatedMulti_DFE(matrix& matrix_init, PicState_int64& input_state,
    PicState_int64& output_state, Complex16& perm, int useDual);

void
GlynnPermanentCalculatorRepeatedMultiInputBatch_DFE(matrix& matrix_init, std::vector<std::vector<PicState_int64>>& input_states,
    std::vector<PicState_int64>& output_states, std::vector<std::vector<Complex16>>& perm, int useDual);

void
GlynnPermanentCalculatorRepeatedMultiOutputBatch_DFE(matrix& matrix_init, std::vector<PicState_int64>& input_states,
    std::vector<std::vector<PicState_int64>>& output_states, std::vector<std::vector<Complex16>>& perm, int useDual);

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void GlynnPermanentCalculatorRepeated_DFE(matrix& matrix_mtx, PicState_int64& input_state,
    PicState_int64& output_state, Complex16& perm, int useDual, int useFloat);

void
GlynnPermanentCalculatorRepeatedInputBatch_DFE(matrix& matrix_init, std::vector<std::vector<PicState_int64>>& input_states,
    std::vector<PicState_int64>& output_states, std::vector<std::vector<Complex16>>& perm, int useDual, int useFloat);

void
GlynnPermanentCalculatorRepeatedOutputBatch_DFE(matrix& matrix_init, std::vector<PicState_int64>& input_states,
    std::vector<std::vector<PicState_int64>>& output_states, std::vector<std::vector<Complex16>>& perm, int useDual, int useFloat);


}


#endif
