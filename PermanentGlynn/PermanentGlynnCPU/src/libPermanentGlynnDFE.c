#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <MaxSLiCInterface.h>

#ifdef MAXELER_SIM
#include "PermanentGlynn_singleSIM.h"
#include "PermanentGlynn_dualSIM.h"
#else
#include "PermanentGlynn_singleDFE.h"
#include "PermanentGlynn_dualDFE.h"
#endif

#define SIZE 8

#define BASEKERNPOW2 3


/// @brief Structure type representing 16 byte complex numbers
typedef struct Complex16 {
  /// the real part of a complex number
  double real;
  /// the imaginary part of a complex number
  double imag;
} Complex16;

typedef struct ComplexFix16 {
  /// the real part of a complex number
  __int64_t real;
  /// the imaginary part of a complex number
  __int64_t imag;
} ComplexFix16;


/// @brief Structure type representing 32 byte complex numbers
typedef struct Complex32 {
  /// the real part of a complex number
  long double real;
  /// the imaginary part of a complex number
  long double imag;
} Complex32;



/**
@brief Calculates the n-th power of 2.
@param n An natural number
@return Returns with the n-th power of 2.
*/
unsigned long long power_of_2(unsigned long long n) {
  if (n == 0) return 1;
  if (n == 1) return 2;

  return 2 * power_of_2(n-1);
}


/// static variable to indicate whether DFE is initialized
typedef void (*RUNFUNC)(max_engine_t*, void*);
static bool initialized = false, useDual;
static max_file_t* mavMaxFile;
static max_engine_t* mavDFE;
static void (*freeFunc)(void);
static RUNFUNC runFunc;
#ifndef MAXELER_SIM
typedef void (*RUNARRAYFUNC)(max_engarray_t*, void**);
static max_engarray_t* array = NULL;
static RUNARRAYFUNC runArrayFunc;
#endif

void releive_DFE();
/**
@brief Interface function to initialize DFE array
*/
void initialize_DFE(int dual)
{

	if (initialized) {
    if (dual==useDual) return;
    else releive_DFE();
  }
  max_file_t* (*initFunc)(void) = NULL;
  useDual = dual;
  if (!useDual) {
#ifdef MAXELER_SIM
    initFunc = PermanentGlynn_singleSIM_init, runFunc = (RUNFUNC)PermanentGlynn_singleSIM_run, freeFunc = PermanentGlynn_singleSIM_free;
#else
    initFunc = PermanentGlynn_singleDFE_init, runFunc = (RUNFUNC)PermanentGlynn_singleDFE_run, freeFunc = PermanentGlynn_singleDFE_free;
#endif  
  } else {
#ifdef MAXELER_SIM
    initFunc = PermanentGlynn_dualSIM_init, runFunc = (RUNFUNC)PermanentGlynn_dualSIM_run, freeFunc = PermanentGlynn_dualSIM_free;
#else
    initFunc = PermanentGlynn_dualDFE_init, runArrayFunc = (RUNARRAYFUNC)PermanentGlynn_dualDFE_run_array, freeFunc = PermanentGlynn_dualDFE_free;
#endif  
  }
	// initialize the max file

#ifdef DEBUG
	printf("Maxfile initialized\n");
#endif
	

  if (!initFunc) return;
  mavMaxFile = initFunc();
#ifdef MAXELER_SIM 
  mavDFE = max_load(mavMaxFile, "local:*");
#else
  if (dual) array = max_load_array(mavMaxFile, 2, "*");
  else mavDFE = max_load(mavMaxFile, "local:*");
#endif
  initialized = true;
#ifdef DEBUG
	printf("Maxfile uploaded to DFE\n");
#endif


}




/**
@brief Interface function to releive DFE array
*/
void releive_DFE()
{

	if (!initialized) return;

#ifdef DEBUG
        printf("Unloading Maxfile\n");
#endif

	// unload the max files from the devices
  initialized = false;
#ifdef MAXELER_SIM
  max_unload(mavDFE);
#else
  if (useDual) max_unload_array(array);
  else max_unload(mavDFE);
#endif
  max_file_free(mavMaxFile);
  freeFunc();  
}

/**
@brief Interface function to calculate the Permanent using Glynns formula on DFE
*/
void calcPermanentGlynnDFE(const ComplexFix16** mtx_data, const long double* renormalize_data, const uint64_t rows, const uint64_t cols, Complex16* perm)
{
    if (!initialized) return;
    
    uint64_t numOfPartialPerms = rows;
    //numOfPartialPerms = max(numOfPartialPerms, BASEKERNPOW2+1+1+(useDual ? 1 : 0));//extra 1 since maxTicks cannot be 1, minimum of 2

    //printf("%lld, %d\n", numOfPartialPerms, rows);

	// variable to store the result
	__int128 res[2];

    union {
#ifdef MAXELER_SIM    
      PermanentGlynn_singleSIM_actions_t glynnRowsGray;
      PermanentGlynn_dualSIM_actions_t dualGlynnRowsGray;
#else
      PermanentGlynn_singleDFE_actions_t glynnRowsGray;
      PermanentGlynn_dualDFE_actions_t dualGlynnRowsGray;
#endif
    } actions
#ifdef MAXELER_SIM
             ;
#else
             , dualactions;
    void *arractions[2] = {&actions, &dualactions};
#endif

    // simulation
    if (!useDual) {
      actions.glynnRowsGray.param_ticksMax = numOfPartialPerms, actions.glynnRowsGray.outstream_res = res;
      actions.glynnRowsGray.instream_InputMtx0 = (__int64_t*)mtx_data[0]; actions.glynnRowsGray.instream_size_InputMtx0 = sizeof(ComplexFix16)*10*rows;
      actions.glynnRowsGray.instream_InputMtx1 = (__int64_t*)mtx_data[1]; actions.glynnRowsGray.instream_size_InputMtx1 = sizeof(ComplexFix16)*10*rows;
      actions.glynnRowsGray.instream_InputMtx2 = (__int64_t*)mtx_data[2]; actions.glynnRowsGray.instream_size_InputMtx2 = sizeof(ComplexFix16)*10*rows;
      actions.glynnRowsGray.instream_InputMtx3 = (__int64_t*)mtx_data[3]; actions.glynnRowsGray.instream_size_InputMtx3 = sizeof(ComplexFix16)*10*rows;
    } else {
      //Simulation of manager I/Os of purpose OTHER_FPGA not yet supported.
#ifdef MAXELER_SIM
      actions.dualGlynnRowsGray.param_ticksMax = numOfPartialPerms, actions.dualGlynnRowsGray.outstream_res = res;
      actions.dualGlynnRowsGray.instream_InputMtx0 = (__int64_t*)mtx_data[0]; actions.dualGlynnRowsGray.instream_size_InputMtx0 = sizeof(ComplexFix16)*10*rows;
      actions.dualGlynnRowsGray.instream_InputMtx1 = (__int64_t*)mtx_data[1]; actions.dualGlynnRowsGray.instream_size_InputMtx1 = sizeof(ComplexFix16)*10*rows;
      actions.dualGlynnRowsGray.instream_InputMtx2 = (__int64_t*)mtx_data[2]; actions.dualGlynnRowsGray.instream_size_InputMtx2 = sizeof(ComplexFix16)*10*rows;
      actions.dualGlynnRowsGray.instream_InputMtx3 = (__int64_t*)mtx_data[3]; actions.dualGlynnRowsGray.instream_size_InputMtx3 = sizeof(ComplexFix16)*10*rows;
      actions.dualGlynnRowsGray.instream_InputMtx4 = (__int64_t*)mtx_data[0]; actions.dualGlynnRowsGray.instream_size_InputMtx4 = sizeof(ComplexFix16)*10*rows;
      actions.dualGlynnRowsGray.instream_InputMtx5 = (__int64_t*)mtx_data[1]; actions.dualGlynnRowsGray.instream_size_InputMtx5 = sizeof(ComplexFix16)*10*rows;
      actions.dualGlynnRowsGray.instream_InputMtx6 = (__int64_t*)mtx_data[2]; actions.dualGlynnRowsGray.instream_size_InputMtx6 = sizeof(ComplexFix16)*10*rows;
      actions.dualGlynnRowsGray.instream_InputMtx7 = (__int64_t*)mtx_data[3]; actions.dualGlynnRowsGray.instream_size_InputMtx7 = sizeof(ComplexFix16)*10*rows;
#else
      actions.dualGlynnRowsGray.param_isLocal = 1, actions.dualGlynnRowsGray.param_ticksMax = numOfPartialPerms, actions.dualGlynnRowsGray.outstream_res = res, actions.dualGlynnRowsGray.outstream_size_res = sizeof(res);
      actions.dualGlynnRowsGray.instream_InputMtx0 = (__int64_t*)mtx_data[0]; actions.dualGlynnRowsGray.instream_size_InputMtx0 = sizeof(ComplexFix16)*10*rows;
      actions.dualGlynnRowsGray.instream_InputMtx1 = (__int64_t*)mtx_data[1]; actions.dualGlynnRowsGray.instream_size_InputMtx1 = sizeof(ComplexFix16)*10*rows;
      actions.dualGlynnRowsGray.instream_InputMtx2 = (__int64_t*)mtx_data[2]; actions.dualGlynnRowsGray.instream_size_InputMtx2 = sizeof(ComplexFix16)*10*rows;
      actions.dualGlynnRowsGray.instream_InputMtx3 = (__int64_t*)mtx_data[3]; actions.dualGlynnRowsGray.instream_size_InputMtx3 = sizeof(ComplexFix16)*10*rows;
      dualactions.dualGlynnRowsGray.param_isLocal = 0, dualactions.dualGlynnRowsGray.param_ticksMax = numOfPartialPerms, dualactions.dualGlynnRowsGray.outstream_res = NULL, dualactions.dualGlynnRowsGray.outstream_size_res = 0;
      dualactions.dualGlynnRowsGray.instream_InputMtx0 = (__int64_t*)mtx_data[0]; dualactions.dualGlynnRowsGray.instream_size_InputMtx0 = sizeof(ComplexFix16)*10*rows;
      dualactions.dualGlynnRowsGray.instream_InputMtx1 = (__int64_t*)mtx_data[1]; dualactions.dualGlynnRowsGray.instream_size_InputMtx1 = sizeof(ComplexFix16)*10*rows;
      dualactions.dualGlynnRowsGray.instream_InputMtx2 = (__int64_t*)mtx_data[2]; dualactions.dualGlynnRowsGray.instream_size_InputMtx2 = sizeof(ComplexFix16)*10*rows;
      dualactions.dualGlynnRowsGray.instream_InputMtx3 = (__int64_t*)mtx_data[3]; dualactions.dualGlynnRowsGray.instream_size_InputMtx3 = sizeof(ComplexFix16)*10*rows;
#endif
    }

#ifdef DEBUG
	printf("Start permanent calulation on DFE\n");
#endif

#ifdef MAXELER_SIM
    runFunc(mavDFE, &actions);
#else
    if (useDual) runArrayFunc(array, arractions);
    else runFunc(mavDFE, &actions);
#endif

#ifdef DEBUG
	printf("Permanent calulation on DFE finished\n");
#endif

    //128-bit fixed point with 124 fractional bits conversion by dividing by 2^124==(2^62)*(2^62) 
    numOfPartialPerms = 1ULL << (numOfPartialPerms-1);
    long double factor = (long double)(1ULL<<62);

    long double real = ((long double)res[0])/factor/factor;
    long double imag = ((long double)res[1])/factor/factor;

    real /= numOfPartialPerms;
    imag /= numOfPartialPerms;

    // renormalize the result according to the normalization of the input matrix
    for (int jdx=0; jdx<cols; jdx++ ) {
        real *= renormalize_data[jdx];
        imag *= renormalize_data[jdx];
    }
    perm->real = real; perm->imag = imag;

    return;
}



