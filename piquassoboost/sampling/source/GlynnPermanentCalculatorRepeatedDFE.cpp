#include "GlynnPermanentCalculatorRepeatedDFE.h"
#include "GlynnPermanentCalculatorRepeated.h"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif
#include <vector>

CALCPERMGLYNNREPDFE calcPermanentGlynnRepDFE = NULL;
INITPERMGLYNNREPDFE initializeRep_DFE = NULL;
FREEPERMGLYNNREPDFE releiveRep_DFE = NULL;

#define ROWCOL(m, r, c) ToComplex32(m[ r*m.stride + c])

namespace pic {

inline Complex16 ToComplex16(Complex32 v) {
  return Complex16((double)v.real(), (double)v.imag());
}

inline Complex32 ToComplex32(Complex16 v) {
  return Complex32((long double)v.real(), (long double)v.imag());
}

uint64_t binomial_gcode(uint64_t bc, int parity, uint64_t n, uint64_t k)
{
  return parity ? bc*k/(n-k+1) : bc*(n-k)/(k+1);
}

void input_to_bincoeff_indices(PicState_int64& input_state, int useDual, std::vector<uint8_t> & rowchange_indices, std::vector<uint64_t> & mplicity, uint8_t & onerows, uint64_t & changecount, uint8_t & mulsum)
{
  std::vector<uint8_t> mrows;
  for (size_t i = 0; i < input_state.size(); i++) {
    if (input_state[i] == 1) rowchange_indices.push_back(i);
    else if (input_state[i] > 1) mrows.push_back(i);
  }
  sort(mrows.begin(), mrows.end(), [&input_state](size_t i, size_t j) { return input_state[i] < input_state[j]; }); 
  while (rowchange_indices.size() < 1+2+(useDual ? 1 : 0)) { //Glynn anchor row, plus 2/3 anchor rows needed for binary Gray code in kernel plus one more for the sum up kernel to tick
    rowchange_indices.push_back(mrows[0]);
    input_state[mrows[0]]--;
    if (mrows[0] == 1) {
      rowchange_indices.push_back(mrows[0]);
      mrows.erase(mrows.begin());
    }
  }
  onerows = rowchange_indices.size(), mulsum = 0, changecount = 0;
  if (mrows.size() == 0) { mplicity.push_back(1); return; }
  std::vector<uint64_t> curmp, inp;
  for (size_t i = 0; i < mrows.size(); i++) {
    rowchange_indices.push_back(mrows[i]);
    curmp.push_back(input_state[mrows[i]]);
    inp.push_back(input_state[mrows[i]]);
    mulsum += input_state[mrows[i]];
  }
  int parity = 0;
  uint64_t gcodeidx = 0, cur_multiplicity = 1;
  while (true) {
    mplicity.push_back(cur_multiplicity);
    parity = !parity;
    for (size_t i = curmp.size()-1; ; i--) {
      bool curdir = gcodeidx & (1 << i);
      if ((!curdir && curmp[i] != inp[i]) || (curdir && curmp[i] != -inp[i])) {
        cur_multiplicity = binomial_gcode(cur_multiplicity, curdir, inp[i], (curmp[i] + inp[i]) / 2);
        curmp[i] = curdir ? curmp[i] - 2 : curmp[i] + 2;
        rowchange_indices.push_back(rowchange_indices[onerows+i] | (curdir ? 0x80 : 0)); //high bit indicates subtraction
        changecount++;
        gcodeidx ^= (1 << curmp.size()) - (1 << (i+1));        
        break;
      } else if (i == 0) return;
    }
  } 
}

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void
GlynnPermanentCalculatorRepeated_DFE(matrix& matrix_mtx, PicState_int64& input_state,
    PicState_int64& output_state, Complex16& perm, int useDual)
{
    int64_t photons = 0;
    for (size_t i = 0; i < input_state.size(); i++) {
        photons += input_state[i];
    }
    if (photons < 1+BASEKERNPOW2 || (photons < 1+1+BASEKERNPOW2 && useDual)) { //compute with other method
      GlynnPermanentCalculatorRepeated gpc;
      perm = gpc.calculate(matrix_mtx, input_state, output_state);
      return;
    }
    // calulate the maximal sum of the columns to normalize the matrix
    matrix_base<Complex32> colSumMax( matrix_mtx.cols, 1);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex32) );
    for (size_t idx=0; idx<matrix_mtx.rows; idx++) {
        for( size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
            Complex32 value1 = colSumMax[jdx] + matrix_mtx[ idx*matrix_mtx.stride + jdx];
            Complex32 value2 = colSumMax[jdx] - matrix_mtx[ idx*matrix_mtx.stride + jdx];
            if ( std::abs( value1 ) < std::abs( value2 ) ) {
                colSumMax[jdx] = value2;
            }
            else {
                colSumMax[jdx] = value1;
            }

        }

    }

    // calculate the renormalization coefficients
    matrix_base<long double> renormalize_data(matrix_mtx.cols, 1);
    for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
        renormalize_data[jdx] = std::abs(colSumMax[jdx]); 
        //printf("%d %.21Lf\n", jdx, renormalize_data[jdx]);
    }

    // renormalize the input matrix and convert to fixed point maximizing precision via long doubles
    // SLR and DFE input matrix with 1.0 filling on top row, 0 elsewhere 
    const size_t max_dim = useDual ? MAX_FPGA_DIM : MAX_SINGLE_FPGA_DIM;
    const size_t rows = matrix_mtx.rows;
    const size_t max_fpga_cols = max_dim >> BASEKERNPOW2;
    const size_t numinits = 1 << BASEKERNPOW2;
    const size_t actualinits = (matrix_mtx.cols + 9) / 10;
    matrix_base<ComplexFix16> mtxfix[numinits] = {};
    const long double fixpow = 1L << 62;
    for (size_t i = 0; i < actualinits; i++) {
      mtxfix[i] = matrix_base<ComplexFix16>(rows, max_fpga_cols);
      size_t basecol = max_fpga_cols * i;
      size_t lastcol = matrix_mtx.cols<=basecol ? 0 : std::min(max_fpga_cols, matrix_mtx.cols-basecol);
      for (size_t idx=0; idx < rows; idx++) {
        size_t offset = idx * matrix_mtx.stride + basecol;
        size_t offset_small = idx*mtxfix[i].stride;
        for (size_t jdx = 0; jdx < lastcol; jdx++) {
          mtxfix[i][offset_small+jdx].real = llrint((long double)matrix_mtx[offset+jdx].real() * fixpow / renormalize_data[basecol+jdx]);
          mtxfix[i][offset_small+jdx].imag = llrint((long double)matrix_mtx[offset+jdx].imag() * fixpow / renormalize_data[basecol+jdx]);
          //printf("%d %d %d %llX %llX\n", i, idx, jdx, mtxfix[i][offset_small+jdx].real, mtxfix[i][offset_small+jdx].imag); 
        }
        memset(&mtxfix[i][offset_small+lastcol], 0, sizeof(ComplexFix16)*(max_fpga_cols-lastcol));
      }
      for (size_t jdx = lastcol; jdx < max_fpga_cols; jdx++) mtxfix[i][jdx].real = fixpow; 
    }

    //note: stride must equal number of columns, or this will not work as the C call expects contiguous data
    ComplexFix16* mtx_fix_data[numinits];
    //assert(mtxfix[i].stride == mtxfix[i].cols);
    for (size_t i = 0; i < numinits; i++) mtx_fix_data[i] = mtxfix[i].get_data();
    std::vector<unsigned char> colIndices; colIndices.reserve(max_dim);
    for (size_t i = 0; i < input_state.size(); i++) {
      for (size_t j = input_state[i]; j != 0; j--) {
        colIndices.push_back(i);
      }
    }
    for (size_t i = (colIndices.size() % 16 == 0) ? 0 : (16 - colIndices.size() % 16); i != 0; i--) colIndices.push_back(0); //round up to nearest 16 bytes to allow streaming
    std::vector<uint8_t> rowchange_indices;
    std::vector<uint64_t> mplicity;
    uint8_t onerows, mulsum; uint64_t changecount;
    input_to_bincoeff_indices(output_state, useDual, rowchange_indices, mplicity, onerows, changecount, mulsum); 
    for (size_t i = (rowchange_indices.size() % 16 == 0) ? 0 : (16 - rowchange_indices.size() % 16); i != 0; i--) rowchange_indices.push_back(0); //round up to nearest 16 bytes to allow streaming
    calcPermanentGlynnRepDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, colIndices.data(),
      rowchange_indices.data(), mplicity.data(), onerows, changecount, mulsum, &perm);


    return;
}

}
