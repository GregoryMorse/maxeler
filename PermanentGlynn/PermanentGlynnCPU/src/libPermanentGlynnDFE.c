#include <stdint.h>
#include <stdlib.h>
#include <MaxSLiCInterface.h>

#ifdef MAXELER_SIM
#include "PermanentGlynn_singleSIM.h"
#include "PermanentGlynn_dualSIM.h"
#else
#include "PermanentGlynn_singleDFE.h"
#include "PermanentGlynn_dualDFE.h"
#endif

#define SIZE 8




/// @brief Structure type representing 16 byte complex numbers
typedef struct Complex16 {
  /// the real part of a complex number
  double real;
  /// the imaginary part of a complex number
  double imag;
} Complex16;


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
    initFunc = PermanentGlynn_dualDFE_init, runFunc = (RUNARRAYFUNC)PermanentGlynn_dualDFE_run_array, freeFunc = PermanentGlynn_dualDFE_free;
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
void calcPermanentGlynnDFE(const Complex16* mtx_data, const double* renormalize_data, const uint64_t rows, const uint64_t cols, Complex16* perm)
{
    if (!initialized) return;
    
    uint64_t numOfPartialPerms = rows;

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
      actions.glynnRowsGray.param_ticksMax = numOfPartialPerms, actions.glynnRowsGray.param_InputMtx = (double*)mtx_data, actions.glynnRowsGray.outstream_res = res;
    } else {
      //Simulation of manager I/Os of purpose OTHER_FPGA not yet supported.
#ifdef MAXELER_SIM
      actions.dualGlynnRowsGray.param_ticksMax = numOfPartialPerms, actions.dualGlynnRowsGray.param_InputMtx = (double*)mtx_data, actions.dualGlynnRowsGray.outstream_res = res;
#else
      actions.dualGlynnRowsGray.param_isLocal = 1, actions.dualGlynnRowsGray.param_ticksMax = numOfPartialPerms, actions.dualGlynnRowsGray.param_InputMtx = (double*)mtx_data, actions.dualGlynnRowsGray.outstream_res = res, actions.dualGlynnRowsGray.outstream_size_res = sizeof(res);
      dualactions.dualGlynnRowsGray.param_isLocal = 0, dualactions.dualGlynnRowsGray.param_ticksMax = numOfPartialPerms, dualactions.dualGlynnRowsGray.param_InputMtx = (double*)mtx_data, dualactions.dualGlynnRowsGray.outstream_res = NULL, dualactions.dualGlynnRowsGray.outstream_size_res = 0;
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
    numOfPartialPerms = 1 << (numOfPartialPerms-1);
    long double factor = (long double)power_of_2(62);

    perm->real = ((long double)res[0])/factor/factor;
    perm->imag = ((long double)res[1])/factor/factor;

    perm->real = perm->real / numOfPartialPerms;
    perm->imag = perm->imag / numOfPartialPerms;

    // renormalize the result according to the normalization of th einput matrix
    for (int jdx=0; jdx<cols; jdx++ ) {
        perm->real = perm->real * renormalize_data[jdx];
        perm->imag = perm->imag * renormalize_data[jdx];
    }





    return;
}



