#include "GlynnPermanentCalculatorRepeatedDFE.h"
#include "GlynnPermanentCalculatorRepeated.h"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif
#include <vector>
#include "common_functionalities.h"

typedef void(*CALCPERMGLYNNREPDFE)(const pic::ComplexFix16**, const long double*, const uint64_t, const uint64_t, const unsigned char*,
  const uint8_t*, const uint8_t, const uint8_t, const uint64_t*, const uint64_t, const uint8_t, pic::Complex16*);
typedef int(*INITPERMGLYNNREPDFE)(int, size_t*, size_t*);
typedef void(*FREEPERMGLYNNREPDFE)(void);

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

matrix transpose_reorder_rows_cols(matrix& matrix_mtx, std::vector<uint8_t> & rowchange_indices, std::vector<uint8_t> & colIndices, int transpose)
{
    matrix matrix_rows(rowchange_indices.size(), colIndices.size());
    if (transpose) {
        for (size_t i = 0; i < colIndices.size(); i++) {
            size_t offset = colIndices[i]*matrix_mtx.stride;
            for (size_t j = 0; j < rowchange_indices.size(); j++) {
                matrix_rows[j*matrix_rows.stride+i] = matrix_mtx[offset+rowchange_indices[j]];
            }
        }
    } else {
        for (size_t i = 0; i < rowchange_indices.size(); i++) {
            size_t offset = rowchange_indices[i]*matrix_mtx.stride;
            size_t newoffset = i*matrix_rows.stride;
            for (size_t j = 0; j < colIndices.size(); j++) {
                matrix_rows[newoffset+j] = matrix_mtx[offset+colIndices[j]];
            }
        }
    }
    return matrix_rows;
}

inline void symmetricQuadrantNormalize(Complex32* sums, Complex16 val) {
    Complex32 value1 = sums[0] + val;
    Complex32 value2 = sums[0] - val;
    Complex32 value3 = sums[1] + val;
    Complex32 value4 = sums[1] - val;
    int symQuad1 = (value1.real() < 0) == (value1.imag() < 0);                  
    int symQuad2 = (value2.real() < 0) == (value2.imag() < 0);
    int symQuad3 = (value3.real() < 0) == (value3.imag() < 0);
    int symQuad4 = (value4.real() < 0) == (value4.imag() < 0);
    if (symQuad1 == symQuad2) { 
        sums[symQuad1] = std::norm(value1) > std::norm(value2) ? value1 : value2;
        if (symQuad3 == symQuad2) {
            sums[symQuad3] = std::norm(value3) > std::norm(sums[symQuad3]) ? value3 : sums[symQuad3];
            if (symQuad4 == symQuad3) {
                sums[symQuad4] = std::norm(value4) > std::norm(sums[symQuad4]) ? value4 : sums[symQuad4];
            } else sums[symQuad4] = value4;
        } else {
            sums[symQuad3] = value3;
            sums[symQuad4] = std::norm(value4) > std::norm(sums[symQuad4]) ? value4 : sums[symQuad4];
        }
    } else {
        sums[symQuad1] = value1;
        sums[symQuad2] = value2;
        sums[symQuad3] = std::norm(value3) > std::norm(sums[symQuad3]) ? value3 : sums[symQuad3];  
        sums[symQuad4] = std::norm(value4) > std::norm(sums[symQuad4]) ? value4 : sums[symQuad4];
    }
}

typedef void(*CALCPERMGLYNNDFE)(const pic::ComplexFix16**, const long double*, const uint64_t, const uint64_t, const uint64_t, pic::Complex16*);
extern "C" CALCPERMGLYNNDFE calcPermanentGlynnDFE;
extern "C" CALCPERMGLYNNDFE calcPermanentGlynnDFEF;

void
GlynnPermanentCalculatorRepeatedMulti_DFE(matrix& matrix_init, PicState_int64& input_state,
    PicState_int64& output_state, Complex16& perm, int useDual)
{
    lock_lib();
    int useFloat = 0;
    if (!useFloat) init_dfe_lib(DFE_MAIN, useDual);
    else if (useFloat) init_dfe_lib(DFE_FLOAT, useDual);    
    size_t photons = 0;
    uint64_t t1 = 1, t2 = 1;
    for (size_t i = 0; i < input_state.size(); i++) {
        photons += input_state[i];
        t1 *= (input_state[i]+1); t2 *= (output_state[i]+1);
    }
    if (!((!useFloat && calcPermanentGlynnDFE) || (useFloat && calcPermanentGlynnDFEF)) ||
        photons < 1+dfe_basekernpow2) { //compute with other method
      GlynnPermanentCalculatorRepeated gpc;
      perm = gpc.calculate(matrix_init, input_state, output_state);
      unlock_lib();
      return;
    }
    int transpose = t1 < t2; //transpose if needed to reduce complexity on rows direction
    const size_t max_dim = dfe_mtx_size;
    //convert multiplicities of rows and columns to indices
    std::vector<unsigned char> colIndices; colIndices.reserve(max_dim);
    for (size_t i = 0; i < output_state.size(); i++) {
      for (size_t j = transpose ? output_state[i] : input_state[i]; j != 0; j--) {
        colIndices.push_back(i);
      }
    }  
    PicState_int64 adj_input_state = transpose ? input_state.copy() : output_state.copy();  
    std::vector<uint8_t> mrows;
    std::vector<uint8_t> row_indices;
    for (size_t i = 0; i < adj_input_state.size(); i++) {
        if (adj_input_state[i] == 1) row_indices.push_back(i);
        else if (adj_input_state[i] > 1) mrows.push_back(i);
    }
    //sort multiplicity >=2 row indices since we need anchor rows, and complexity reduction greatest by using smallest multiplicities
    sort(mrows.begin(), mrows.end(), [&adj_input_state](size_t i, size_t j) { return adj_input_state[i] < adj_input_state[j]; }); 
    //while (row_indices.size() < 1+dfe_basekernpow2) { //Glynn anchor row, plus 2/3 anchor rows needed for binary Gray code in kernel
    while (row_indices.size() < 1) { //Glynn anchor row, prevent streaming more than 256MB of data
        row_indices.push_back(mrows[0]);
        if (--adj_input_state[mrows[0]] == 1) {
          row_indices.push_back(mrows[0]);
          mrows.erase(mrows.begin());
        }
    }
    //construct multiplicity Gray code counters
    uint8_t mulsum = 0, onerows = row_indices.size();
    std::vector<uint64_t> curmp, inp;
    uint64_t totalPerms = 1;
    for (size_t i = 0; i < mrows.size(); i++) {
        //for (size_t j = 0; j < adj_input_state[mrows[i]]; j++)
        row_indices.push_back(mrows[i]);
        curmp.push_back(adj_input_state[mrows[i]]);
        inp.push_back(adj_input_state[mrows[i]]);
        mulsum += adj_input_state[mrows[i]];
        totalPerms *= (adj_input_state[mrows[i]] + 1);
    }
    if (onerows < 1+dfe_basekernpow2) { //compute with other method
      GlynnPermanentCalculatorRepeated gpc;
      perm = gpc.calculate(matrix_init, input_state, output_state);
      unlock_lib();
      return;
    }
    if (mrows.size() == 0) {
        matrix matrix_rows = transpose_reorder_rows_cols(matrix_init, row_indices, colIndices, transpose);
        GlynnPermanentCalculator_DFE(matrix_rows, perm, useDual, useFloat);
        unlock_lib();
        return;
    }

    // calulate the maximal sum of the columns to normalize the matrix
    matrix_base<Complex32> colSumMax( photons, 2);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex32) );
    for (size_t i=0; i<row_indices.size(); i++) {
        //size_t offset = (transpose ? colIndices[i] : row_indices[i]) * matrix_init.stride;
        for (int64_t idx = 0; idx < (i < onerows ? 1 : adj_input_state[row_indices[i]]); idx++) {
            for( size_t jdx=0; jdx<photons; jdx++) {
                size_t offset = transpose ? colIndices[jdx]*matrix_init.stride+row_indices[i] : row_indices[i]*matrix_init.stride+colIndices[jdx];
                symmetricQuadrantNormalize(&colSumMax[2*jdx], matrix_init[offset]);
            }
        }
    }

    // calculate the renormalization coefficients
    matrix_base<long double> renormalize_data(1, photons);
    for (size_t jdx=0; jdx<photons; jdx++ ) {
        renormalize_data[jdx] = std::abs(std::norm(colSumMax[2*jdx]) > std::norm(colSumMax[2*jdx+1]) ? colSumMax[2*jdx] : colSumMax[2*jdx+1]);
    }
    
    // renormalize the input matrix and convert to fixed point maximizing precision via long doubles
    // SLR and DFE input matrix with 1.0 filling on top row, 0 elsewhere 
    const size_t rows = row_indices.size();
    const size_t numinits = 4;
    const size_t max_fpga_cols = max_dim / numinits;
    const size_t actualinits = (photons + max_fpga_cols-1) / max_fpga_cols;
    matrix_base<ComplexFix16> mtxprefix[numinits] = {};
    const long double fixpow = 1ULL << 62;
    for (size_t i = 0; i < actualinits; i++) {
      mtxprefix[i] = matrix_base<ComplexFix16>(rows, max_fpga_cols);
      size_t basecol = max_fpga_cols * i;
      size_t lastcol = photons<=basecol ? 0 : std::min(max_fpga_cols, photons-basecol);
      for (size_t idx=0; idx < rows; idx++) {
        size_t offset_small = idx*mtxprefix[i].stride;
        for (size_t jdx = 0; jdx < lastcol; jdx++) {
          size_t offset = transpose ? colIndices[basecol+jdx]*matrix_init.stride+row_indices[idx] : row_indices[idx]*matrix_init.stride+colIndices[basecol+jdx];
          mtxprefix[i][offset_small+jdx].real = llrint((long double)matrix_init[offset].real() * fixpow / renormalize_data[basecol+jdx]);
          mtxprefix[i][offset_small+jdx].imag = llrint((long double)matrix_init[offset].imag() * fixpow / renormalize_data[basecol+jdx]);
          if (idx >= onerows) { //start with all positive Gray codes, so sum everything onto the adjust row
              for (int64_t j = 0; j < adj_input_state[row_indices[idx]]; j++) {
                  mtxprefix[i][jdx].real += mtxprefix[i][offset_small+jdx].real;
                  mtxprefix[i][jdx].imag += mtxprefix[i][offset_small+jdx].imag;
              }
          }
        }
        memset(&mtxprefix[i][offset_small+lastcol], 0, sizeof(ComplexFix16)*(max_fpga_cols-lastcol));
      }
      for (size_t jdx = lastcol; jdx < max_fpga_cols; jdx++) mtxprefix[i][jdx].real = fixpow;
    }
    
    matrix_base<ComplexFix16> mtxfix[numinits] = {};
    for (size_t i = 0; i < actualinits; i++)
        mtxfix[i] = matrix_base<ComplexFix16>(onerows * totalPerms, max_fpga_cols);
  
    Complex32 res;
    uint64_t gcodeidx = 0, cur_multiplicity = 1, skipidx = (1ULL << curmp.size())-1; //gcodeidx is direction bit vector, skipidx set to not skip all indexes - technically "not skip index"
    std::vector<uint64_t> mplicity; //binomial coefficient multipliers
    size_t bytesPerMatrix = onerows*max_fpga_cols*sizeof(uint64_t)*2;
    size_t maxmatrices = (1ULL << 28) / bytesPerMatrix;
    size_t permBase = 0;
    ComplexFix16* mtx_fix_data[numinits];
    matrix_base<long double> renormalize_data_all(maxmatrices, photons);            
    std::vector<Complex16> perms;
    perms.resize(totalPerms);
    while (true) {
        size_t offset_small = (mplicity.size()-permBase)*onerows*max_fpga_cols;
        for (size_t i = 0; i < actualinits; i++) {
            memcpy(mtxfix[i].get_data()+offset_small, mtxprefix[i].get_data(), sizeof(ComplexFix16) * onerows * max_fpga_cols);
        }
        mplicity.push_back(cur_multiplicity);
        if (skipidx == 0 || (mplicity.size() - permBase) == maxmatrices) { //all multiplicities completed
            //GlynnPermanentCalculatorBatch_DFE(matrices, perms, useDual, false);
            //note: stride must equal number of columns, or this will not work as the C call expects contiguous data
            size_t numPerms = std::min(maxmatrices, totalPerms - permBase);
            //assert(mtxfix[i].stride == mtxfix[i].cols);
            for (size_t i = 0; i < maxmatrices; i++) memcpy(renormalize_data_all.get_data()+photons*i, renormalize_data.get_data(), photons * sizeof(long double));
            for (size_t j = 0; j < numinits; j++) mtx_fix_data[j] = mtxfix[j].get_data();
            if (useFloat)
                calcPermanentGlynnDFEF( (const ComplexFix16**)mtx_fix_data, renormalize_data_all.get_data(), onerows, photons, numPerms, perms.data()+permBase);
            else
                calcPermanentGlynnDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data_all.get_data(), onerows, photons, numPerms, perms.data()+permBase);
            if (skipidx == 0) {
                for (size_t i = 0; i < perms.size(); i++) {
                    if (i & 1) res -= ToComplex32(perms[i]) * (long double)mplicity[i]; 
                    else res += ToComplex32(perms[i]) * (long double)mplicity[i];
                }
                perm = ToComplex16(res / (long double)(1ULL << mulsum)); //2**mulsum is the effective number of permanents or sum of all multiplicities
                unlock_lib();
                return;
            }
            permBase += numPerms;
        }
        size_t i = __builtin_ctzll(skipidx); //count of trailing zeros to compute next change index
        bool curdir = (gcodeidx & (1ULL << i)) == 0;
        cur_multiplicity = binomial_gcode(cur_multiplicity, curdir, inp[i], (curmp[i] + inp[i]) / 2);
        curmp[i] = curdir ? curmp[i] - 2 : curmp[i] + 2;
        size_t offset = (onerows+i)*max_fpga_cols;
        //add or subtract to the adjustment row
        for (size_t idx = 0; idx < actualinits; idx++) {
            for (size_t j = 0; j < max_fpga_cols; j++) { //Gray code adjustment by adding or subtracting 2 times the appropriate row, caring for overflow since we are (64, -62) fixed point cannot multiply by 2 which requires 65 bits
                if (curdir) {
                    mtxprefix[idx][j].real = (mtxprefix[idx][j].real - mtxprefix[idx][offset+j].real) - mtxprefix[idx][offset+j].real;
                    mtxprefix[idx][j].imag = (mtxprefix[idx][j].imag - mtxprefix[idx][offset+j].imag) - mtxprefix[idx][offset+j].imag;
                } else {
                    mtxprefix[idx][j].real = (mtxprefix[idx][j].real + mtxprefix[idx][offset+j].real) + mtxprefix[idx][offset+j].real;
                    mtxprefix[idx][j].imag = (mtxprefix[idx][j].imag + mtxprefix[idx][offset+j].imag) + mtxprefix[idx][offset+j].imag;
                }
            }            
        }
        if ((!curdir && curmp[i] == inp[i]) || (curdir && curmp[i] == -inp[i])) skipidx ^= ((1ULL << (i+1)) - 1); //set all skipping before and including current index
        else skipidx ^= ((1ULL << i) - 1); //flip all skipping which come before current index
        gcodeidx ^= (1ULL << i) - 1; //flip all directions which come before current index
    }
}

void
GlynnPermanentCalculatorRepeatedMultiBatch_DFE(matrix& matrix_init, std::vector<PicState_int64>& input_states,
    std::vector<std::vector<PicState_int64>>& output_states, std::vector<std::vector<Complex16>>& perm, int useDual)
{
    for (size_t i = 0; i < input_states.size(); i++) {
        perm[i].resize(output_states[i].size());
        for (size_t j = 0; j < output_states[i].size(); j++) {
            GlynnPermanentCalculatorRepeatedMulti_DFE(matrix_init, input_states[i], output_states[i][j], perm[i][j], useDual); 
        }
    }
}


matrix transpose_reorder_rows(matrix& matrix_mtx, std::vector<uint8_t> & rowchange_indices, int transpose)
{
    matrix matrix_rows(rowchange_indices.size(), transpose ? matrix_mtx.rows : matrix_mtx.cols);
    if (transpose) {
        for (size_t i = 0; i < matrix_mtx.rows; i++) {
            size_t offset = i*matrix_mtx.stride;
            for (size_t j = 0; j < rowchange_indices.size(); j++) {
                matrix_rows[j*matrix_rows.stride+i] = matrix_mtx[offset+rowchange_indices[j]];
            }
        }
    } else {
        for (size_t i = 0; i < rowchange_indices.size(); i++) {
            size_t offset = rowchange_indices[i]*matrix_mtx.stride;
            size_t newoffset = i*matrix_rows.stride;
            for (size_t j = 0; j < matrix_mtx.cols; j++) {
                matrix_rows[newoffset+j] = matrix_mtx[offset+j];
            }
        }
    }
    return matrix_rows;
}
void location_to_counter(std::vector<uint64_t>& count, std::vector<uint64_t>& inp, uint64_t loc)
{
    for (size_t i = 0; i < inp.size(); i++) {
        count.push_back(loc % inp[i]);
        loc = loc / inp[i]; 
    }
}
void counter_to_gcode(std::vector<uint64_t>& gcode, std::vector<uint64_t>& counterChain, std::vector<uint64_t>& inp)
{
    gcode = counterChain;
    int parity = 0;
    for (size_t j = inp.size()-1; j != ~0ULL; j--) {
        if (parity) gcode[j] += inp[j];
        if (((counterChain[j] & 1) != 0) && (((inp[j] & 1) != 0) || (((inp[j] & 1) == 0) && (counterChain[j] < inp[j]))))
            parity = !parity;
    }
}
uint64_t divide_gray_code(std::vector<uint64_t>& inp, std::vector<uint64_t>& mplicity, std::vector<uint8_t> initDirections, uint8_t loopLength)
{
    uint64_t total = 1;
    for (size_t i = 0; i < inp.size(); i++) total *= inp[i];
    uint64_t segment = total / loopLength, rem = total % loopLength;
    uint64_t cursum = 0;
    initDirections.resize(loopLength * inp.size());
    for (size_t i = 0; i < loopLength; i++) {
        std::vector<uint64_t> loc, gcode;
        location_to_counter(loc, inp, cursum);
        counter_to_gcode(gcode, loc, inp);
        uint64_t bincoeff = 1;
        for (size_t j = 0; j < gcode.size(); j++) {
            bool curdir =  gcode[j] < inp[j];
            uint64_t curval = curdir ? inp[j]-1-gcode[j] : gcode[j]-inp[j];
            bincoeff *= binomialCoeff(inp[j], curval);
            int64_t curmp = (curval << 1) - inp[j];
            uint64_t k = 0;
            for (; k < (curmp < 0 ? -curmp : curmp); k++) { //expand Gray code into a bit vector, staggered by loopLength
                initDirections[k*loopLength+i] = curval < 0 ? 1 : 0;
            }
            for (; k < inp[j]; k+=2) { //remaining pairs which sum to 0
                initDirections[k*loopLength+i] = 1; initDirections[k*loopLength+i] = 0;
            }
        }
        mplicity.push_back(bincoeff);
        cursum += segment + ((i < rem) ? 1 : 0);
    }
    return total;
}

matrix input_to_bincoeff_indices(matrix& matrix_mtx, PicState_int64& input_state, int useDual, std::vector<uint8_t> & rowchange_indices, std::vector<uint64_t> & mplicity, uint8_t & onerows, uint64_t & changecount, uint8_t & mulsum, int transpose)
{
  std::vector<uint8_t> mrows;
  std::vector<uint8_t> row_indices;
  for (size_t i = 0; i < input_state.size(); i++) {
    if (input_state[i] == 1) row_indices.push_back(i);
    else if (input_state[i] > 1) mrows.push_back(i);
  }
  sort(mrows.begin(), mrows.end(), [&input_state](size_t i, size_t j) { return input_state[i] < input_state[j]; }); 
  while (row_indices.size() < 1+dfe_basekernpow2) { //Glynn anchor row, plus 2/3 anchor rows needed for binary Gray code in kernel
    row_indices.push_back(mrows[0]);
    if (--input_state[mrows[0]] == 1) {
      row_indices.push_back(mrows[0]);
      mrows.erase(mrows.begin());
    }
  }
  onerows = row_indices.size(), mulsum = 0, changecount = 0;
  std::vector<uint64_t> curmp, inp;
  for (size_t i = 0; i < mrows.size(); i++) {
    row_indices.push_back(mrows[i]);
    curmp.push_back(0); //curmp.push_back(input_state[mrows[i]]);
    inp.push_back(input_state[mrows[i]]+1);
    mulsum += input_state[mrows[i]];
  }
  matrix matrix_rows = transpose_reorder_rows(matrix_mtx, row_indices, transpose);
  for (size_t i = 0; i < row_indices.size(); i++) {
      for (size_t j = i < onerows ? 1 : input_state[row_indices[i]]; j != 0; j--) {
        rowchange_indices.push_back(i);
      }
  }
  if (mrows.size() == 0) { mplicity.push_back(1); return matrix_rows; }
  std::vector<uint8_t> k; k.resize(inp.size());
  uint64_t cur_multiplicity = 1;
  while (true) {
      mplicity.push_back(cur_multiplicity);
      size_t j = 0;
      for (size_t i = 0; i < curmp.size(); i++) {
          if (curmp[i] == inp[i]-1) { curmp[i] = 0; j++; }
          else { curmp[i]++; break; }
      }
      if (j == inp.size()) {
          return matrix_rows;
      }
      bool curdir =  k[j] < inp[j];
      cur_multiplicity = binomial_gcode(cur_multiplicity, curdir, inp[j]-1, curdir ? inp[j]-1-k[j] : k[j]-inp[j]);
      //rowchange_indices.push_back((onerows+j) | (curdir ? 0x80 : 0)); //high bit indicates subtraction
      changecount++;
      for (size_t i = 0; i <= j; i++)
          k[i] = (k[i] != (inp[i] << 1)-1) ? k[i] + 1 : 0;
  }
  /*
  uint64_t gcodeidx = 0, cur_multiplicity = 1, skipidx = (1ULL << curmp.size())-1;
  while (true) {
    mplicity.push_back(cur_multiplicity);
    if (skipidx == 0) {
        return matrix_rows;
    }
    size_t i = __builtin_ctzll(skipidx);
    bool curdir = (gcodeidx & (1ULL << i)) == 0;
    cur_multiplicity = binomial_gcode(cur_multiplicity, curdir, inp[i], (curmp[i] + inp[i]) / 2);
    curmp[i] = curdir ? curmp[i] - 2 : curmp[i] + 2;
    rowchange_indices.push_back((onerows+i) | (curdir ? 0x80 : 0)); //high bit indicates subtraction
    changecount++;
    if ((!curdir && curmp[i] == inp[i]) || (curdir && curmp[i] == -inp[i])) skipidx ^= ((1ULL << (i+1)) - 1);
    else skipidx ^= ((1ULL << i) - 1);
    gcodeidx ^= (1ULL << i) - 1;
  }*/
}

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void
GlynnPermanentCalculatorRepeated_DFE(matrix& matrix_init, PicState_int64& input_state,
    PicState_int64& output_state, Complex16& perm, int useDual)
{
    lock_lib();
    init_dfe_lib(DFE_REP, useDual);    
    size_t photons = 0;
    uint64_t t1 = 1, t2 = 1;   
    for (size_t i = 0; i < input_state.size(); i++) {
        photons += input_state[i];
        t1 *= (input_state[i]+1); t2 *= (output_state[i]+1);
    }
    if (!calcPermanentGlynnRepDFE || photons < 1+dfe_basekernpow2) { //compute with other method
      GlynnPermanentCalculatorRepeated gpc;
      perm = gpc.calculate(matrix_init, input_state, output_state);
      unlock_lib();
      return;
    }
    int transpose = t1 < t2; //transpose if needed to reduce complexity on rows direction
    std::vector<uint8_t> rowchange_indices;
    std::vector<uint64_t> mplicity;
    uint8_t onerows, mulsum; uint64_t changecount;
    PicState_int64 adj_input_state = transpose ? input_state.copy() : output_state.copy();
    matrix matrix_mtx = input_to_bincoeff_indices(matrix_init, adj_input_state, useDual, rowchange_indices, mplicity, onerows, changecount, mulsum, transpose); 
    
    // calulate the maximal sum of the columns to normalize the matrix
    matrix_base<Complex32> colSumMax( matrix_mtx.cols, 2);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex32) );
    for (size_t i=0; i<photons; i++) {
        size_t idx = rowchange_indices[i];
        for( size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
            size_t offset = idx*matrix_mtx.stride + jdx;
            symmetricQuadrantNormalize(&colSumMax[2*jdx], matrix_mtx[offset]);
        }
    }

    // calculate the renormalization coefficients
    matrix_base<long double> renormalize_data(matrix_mtx.cols, 1);
    for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
        renormalize_data[jdx] = std::abs(std::norm(colSumMax[2*jdx]) > std::norm(colSumMax[2*jdx+1]) ? colSumMax[2*jdx] : colSumMax[2*jdx+1]);
        //printf("%d %.21Lf\n", jdx, renormalize_data[jdx]);
    }

    // renormalize the input matrix and convert to fixed point maximizing precision via long doubles
    // SLR and DFE input matrix with 1.0 filling on top row, 0 elsewhere 
    const size_t max_dim = dfe_mtx_size;
    const size_t rows = matrix_mtx.rows;
    const size_t numinits = 4;
    const size_t max_fpga_cols = max_dim / numinits;
    const size_t actualinits = (matrix_mtx.cols + max_fpga_cols-1) / max_fpga_cols;
    matrix_base<ComplexFix16> mtxfix[numinits] = {};
    const long double fixpow = 1ULL << 62;
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
    for (size_t i = 0; i < output_state.size(); i++) {
      for (size_t j = transpose ? output_state[i] : input_state[i]; j != 0; j--) {
        colIndices.push_back(i);
      }
    }
    for (size_t i = (colIndices.size() % 16 == 0) ? 0 : (16 - colIndices.size() % 16); i != 0; i--) colIndices.push_back(0); //round up to nearest 16 bytes to allow streaming
    for (size_t i = (rowchange_indices.size() % 16 == 0) ? 0 : (16 - rowchange_indices.size() % 16); i != 0; i--) rowchange_indices.push_back(0); //round up to nearest 16 bytes to allow streaming
    calcPermanentGlynnRepDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, colIndices.data(),
      rowchange_indices.data(), photons, onerows, mplicity.data(), changecount, mulsum, &perm);

    unlock_lib();
    return;
}

void
GlynnPermanentCalculatorRepeatedBatch_DFE(matrix& matrix_init, std::vector<PicState_int64>& input_states,
    std::vector<std::vector<PicState_int64>>& output_states, std::vector<std::vector<Complex16>>& perm, int useDual)
{
    for (size_t i = 0; i < input_states.size(); i++) {
        perm[i].resize(output_states[i].size());
        for (size_t j = 0; j < output_states[i].size(); j++) {
            GlynnPermanentCalculatorRepeated_DFE(matrix_init, input_states[i], output_states[i][j], perm[i][j], useDual); 
        }
    }
}


}
