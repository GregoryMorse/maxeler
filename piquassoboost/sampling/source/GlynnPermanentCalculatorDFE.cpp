#include "GlynnPermanentCalculatorDFE.h"
#include "GlynnPermanentCalculator.h"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif

CALCPERMGLYNNDFE calcPermanentGlynnDFE = NULL;
INITPERMGLYNNDFE initialize_DFE = NULL;
FREEPERMGLYNNDFE releive_DFE = NULL;
CALCPERMGLYNNDFE calcPermanentGlynnDFEF = NULL;
INITPERMGLYNNDFE initialize_DFEF = NULL;
FREEPERMGLYNNDFE releive_DFEF = NULL;

#include <dlfcn.h>
#include <unistd.h>
#include "GlynnPermanentCalculatorRepeatedDFE.h"

#define DFE_LIB_SIM "libPermanentGlynnSIM.so"
#define DFE_LIB_SIMDUAL "libPermanentGlynnDUALSIM.so"
#define DFE_LIB "libPermanentGlynnDFE.so"
#define DFE_LIBDUAL "libPermanentGlynnDUALDFE.so"
#define DFE_LIB_SIMF "libPermanentGlynnSIMF.so"
#define DFE_LIB_SIMFDUAL "libPermanentGlynnDualSIMF.so"
#define DFE_LIBF "libPermanentGlynnDFEF.so"
#define DFE_LIBFDUAL "libPermanentGlynnDualDFEF.so"
#define DFE_REP_LIB_SIM "libPermRepGlynnSIM.so"
#define DFE_REP_LIB_SIMDUAL "libPermRepGlynnDualSIM.so"
#define DFE_REP_LIB "libPermRepGlynnDFE.so"
#define DFE_REP_LIBDUAL "libPermRepGlynnDualDFE.so"

void* handle = NULL;
size_t refcount = 0;


void unload_dfe_lib()
{
    if (handle) {
        if (releive_DFE) {
            releive_DFE();
            initialize_DFE = NULL, releive_DFE = NULL, calcPermanentGlynnDFE = NULL;
        }
        if (releive_DFEF) {
            releive_DFEF();
            initialize_DFEF = NULL, releive_DFEF = NULL, calcPermanentGlynnDFEF = NULL;
        }
        if (releiveRep_DFE) {
            releiveRep_DFE();
            initializeRep_DFE = NULL, releiveRep_DFE = NULL, calcPermanentGlynnRepDFE = NULL;
        }
        dlclose(handle);
        handle = NULL;
    }
}

void init_dfe_lib(int choice, int dual) {
    unload_dfe_lib();
    const char* simLib = NULL, *lib = NULL;
    if (choice == DFE_MAIN) {
        simLib = dual ? DFE_LIB_SIMDUAL : DFE_LIB_SIM;
        lib = dual ? DFE_LIBDUAL : DFE_LIB;
    } else if (choice == DFE_FLOAT) {
        simLib = dual ? DFE_LIB_SIMFDUAL : DFE_LIB_SIMF;
        lib = dual ? DFE_LIBFDUAL : DFE_LIBF;
    } else if (choice == DFE_REP) {
        simLib = dual ? DFE_REP_LIB_SIMDUAL : DFE_REP_LIB_SIM;
        lib = dual ? DFE_REP_LIBDUAL : DFE_REP_LIB;
    }
    handle = dlopen(getenv("SLIC_CONF") ? simLib : lib, RTLD_NOW); //"MAXELEROSDIR"
    if (handle == NULL) {
        char* pwd = getcwd(NULL, 0);
        fprintf(stderr, "%s\n'%s' (in %s mode) failed to load from working directory '%s' use export LD_LIBRARY_PATH\n", dlerror(), getenv("SLIC_CONF") ? simLib : lib, getenv("SLIC_CONF") ? "simulator" : "DFE", pwd);
        free(pwd);
    } else {
      if (choice == DFE_MAIN) {
          calcPermanentGlynnDFE = (CALCPERMGLYNNDFE)dlsym(handle, "calcPermanentGlynnDFE");
          initialize_DFE = (INITPERMGLYNNDFE)dlsym(handle, "initialize_DFE");
          releive_DFE = (FREEPERMGLYNNDFE)dlsym(handle, "releive_DFE");
      } else if (choice == DFE_FLOAT) {
          calcPermanentGlynnDFEF = (CALCPERMGLYNNDFE)dlsym(handle, "calcPermanentGlynnDFEF");
          initialize_DFEF = (INITPERMGLYNNDFE)dlsym(handle, "initialize_DFEF");
          releive_DFEF = (FREEPERMGLYNNDFE)dlsym(handle, "releive_DFEF");
      } else if (choice == DFE_REP) {
          calcPermanentGlynnRepDFE = (CALCPERMGLYNNREPDFE)dlsym(handle, "calcPermanentGlynnRepDFE");
          initializeRep_DFE = (INITPERMGLYNNREPDFE)dlsym(handle, "initializeRep_DFE");
          releiveRep_DFE = (FREEPERMGLYNNREPDFE)dlsym(handle, "releiveRep_DFE");
      }
    }
}

#define ROWCOL(m, r, c) ToComplex32(m[ r*m.stride + c])

namespace pic {

inline Complex16 ToComplex16(Complex32 v) {
  return Complex16((double)v.real(), (double)v.imag());
}

inline Complex32 ToComplex32(Complex16 v) {
  return Complex32((long double)v.real(), (long double)v.imag());
}

inline long long doubleToLLRaw(double d)
{
    double* pd = &d;
    long long* pll = (long long*)pd;
    return *pll;
}

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void
GlynnPermanentCalculator_DFE(matrix& matrix_mtx, Complex16& perm, int useDual, int useFloat)
{
    if (!useFloat && !initialize_DFE) init_dfe_lib(DFE_MAIN);
    else if (useFloat && !initialize_DFEF) init_dfe_lib(DFE_FLOAT);
    if (!useFloat && initialize_DFE) initialize_DFE(useDual);
    else if (useFloat && initialize_DFEF) initialize_DFEF(useDual);

    if (!((!useFloat && calcPermanentGlynnDFE) || (useFloat && calcPermanentGlynnDFEF)) ||
        matrix_mtx.rows < 1+BASEKERNPOW2 || (matrix_mtx.rows < 1+1+BASEKERNPOW2 && useDual)) { //compute with other method
      if (matrix_mtx.rows == 0) perm = Complex16(1.0,0.0);
      else if (matrix_mtx.rows == 1) perm = matrix_mtx[0];
      else if (matrix_mtx.rows == 2) perm = ToComplex16(ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 1) + ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 0));
      else if (matrix_mtx.rows == 3)
        perm = ToComplex16(ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 2) +
               ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 0) +
               ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 1) +
               ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 0) +
               ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 2) +
               ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 1));
      else if (matrix_mtx.rows == 4)
        perm = ToComplex16(ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 2) * ROWCOL(matrix_mtx, 3, 3) +
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
                ROWCOL(matrix_mtx, 0, 3) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 1) * ROWCOL(matrix_mtx, 3, 0));
      else {
        GlynnPermanentCalculator gpc;
        perm = gpc.calculate(matrix_mtx);
      }        
      return;
    }
    matrix_base<long double> renormalize_data(matrix_mtx.cols, 1);
    if (!useFloat) {
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
        for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
            renormalize_data[jdx] = std::abs(colSumMax[jdx]); 
            //printf("%d %.21Lf\n", jdx, renormalize_data[jdx]);
        }
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
    const double fOne = doubleToLLRaw(1.0);
    for (size_t i = 0; i < actualinits; i++) {
      mtxfix[i] = matrix_base<ComplexFix16>(rows, max_fpga_cols);
      size_t basecol = max_fpga_cols * i;
      size_t lastcol = matrix_mtx.cols<=basecol ? 0 : std::min(max_fpga_cols, matrix_mtx.cols-basecol);
      for (size_t idx=0; idx < rows; idx++) {
        size_t offset = idx * matrix_mtx.stride + basecol;
        size_t offset_small = idx*mtxfix[i].stride;
        for (size_t jdx = 0; jdx < lastcol; jdx++) {
          mtxfix[i][offset_small+jdx].real = useFloat ? doubleToLLRaw(matrix_mtx[offset+jdx].real()) : llrint((long double)matrix_mtx[offset+jdx].real() * fixpow / renormalize_data[basecol+jdx]);
          mtxfix[i][offset_small+jdx].imag = useFloat ? doubleToLLRaw(matrix_mtx[offset+jdx].imag()) : llrint((long double)matrix_mtx[offset+jdx].imag() * fixpow / renormalize_data[basecol+jdx]);
          //printf("%d %d %d %llX %llX\n", i, idx, jdx, mtxfix[i][offset_small+jdx].real, mtxfix[i][offset_small+jdx].imag); 
        }
        memset(&mtxfix[i][offset_small+lastcol], 0, sizeof(ComplexFix16)*(max_fpga_cols-lastcol));
      }
      for (size_t jdx = lastcol; jdx < max_fpga_cols; jdx++) mtxfix[i][jdx].real = useFloat ? fOne : fixpow; 
    }

    //note: stride must equal number of columns, or this will not work as the C call expects contiguous data
    ComplexFix16* mtx_fix_data[numinits];
    //assert(mtxfix[i].stride == mtxfix[i].cols);
    //assert(matrix_mtx.rows == matrix_mtx.cols && matrix_mtx.rows <= (dual ? MAX_FPGA_DIM : MAX_SINGLE_FPGA_DIM));
    for (size_t i = 0; i < numinits; i++) mtx_fix_data[i] = mtxfix[i].get_data();
    if (useFloat)
        calcPermanentGlynnDFEF( (const ComplexFix16**)mtx_fix_data, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, &perm);
    else
        calcPermanentGlynnDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, &perm);


    return;
}

}
