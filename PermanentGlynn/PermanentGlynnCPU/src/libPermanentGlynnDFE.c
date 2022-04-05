#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <MaxSLiCInterface.h>

#ifdef MAXELER_SIM
#ifdef USE_FLOAT
#ifndef DUAL
#include "PermanentGlynn_singleSIMF.h"
#else
#include "PermanentGlynn_dualSIMF.h"
#endif
#else
#ifndef DUAL
#include "PermanentGlynn_singleSIM.h"
#else
#include "PermanentGlynn_dualSIM.h"
#endif
#endif
#else
#ifdef USE_FLOAT
#ifndef DUAL
#include "PermanentGlynn_singleDFEF.h"
#else
#include "PermanentGlynn_dualDFEF.h"
#endif
#else
#ifndef DUAL
#include "PermanentGlynn_singleDFE.h"
#else
#include "PermanentGlynn_dualDFE.h"
#endif
#endif
#endif

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


/// static variable to indicate whether DFE is initialized
typedef void (*RUNFUNC)(max_engine_t*, void*);
static bool initialized = false;
static max_file_t* mavMaxFile;
static void (*freeFunc)(void);
static max_group_t* group = NULL;
#if defined(DUAL) && !defined(MAXELER_SIM)
//typedef void (*RUNARRAYFUNC)(max_engarray_t*, void**);
//static max_engarray_t* array = NULL;
//static RUNARRAYFUNC runArrayFunc;
typedef max_run_t*(*RUNGROUPFUNC)(max_group_t*, void*);
#else
//static max_engine_t* mavDFE;
//static RUNFUNC runFunc;
typedef void(*RUNGROUPFUNC)(max_group_t*, void*);
#endif
static RUNGROUPFUNC runFunc;

#ifdef USE_FLOAT
void releive_DFEF();
#else
void releive_DFE();
#endif
/**
@brief Interface function to initialize DFE array
*/
#ifdef USE_FLOAT
void initialize_DFEF()
#else
void initialize_DFE()
#endif
{

	if (initialized) return;
  max_file_t* (*initFunc)(void) = NULL;
#ifndef DUAL
#ifdef MAXELER_SIM
#ifdef USE_FLOAT
    initFunc = PermanentGlynn_singleSIMF_init, runFunc = (RUNGROUPFUNC)PermanentGlynn_singleSIMF_run_group, freeFunc = PermanentGlynn_singleSIMF_free;
#else
    initFunc = PermanentGlynn_singleSIM_init, runFunc = (RUNGROUPFUNC)PermanentGlynn_singleSIM_run_group, freeFunc = PermanentGlynn_singleSIM_free;
#endif
#else
#ifdef USE_FLOAT
    initFunc = PermanentGlynn_singleDFEF_init, runFunc = (RUNGROUPFUNC)PermanentGlynn_singleDFEF_run_group, freeFunc = PermanentGlynn_singleDFEF_free;
#else
    initFunc = PermanentGlynn_singleDFE_init, runFunc = (RUNGROUPFUNC)PermanentGlynn_singleDFE_run_group, freeFunc = PermanentGlynn_singleDFE_free;
#endif
#endif  
#else
#ifdef MAXELER_SIM
#ifdef USE_FLOAT
    initFunc = PermanentGlynn_dualSIMF_init, runFunc = (RUNGROUPFUNC)PermanentGlynn_dualSIMF_run_group, freeFunc = PermanentGlynn_dualSIMF_free;
#else
    initFunc = PermanentGlynn_dualSIM_init, runFunc = (RUNGROUPFUNC)PermanentGlynn_dualSIM_run_group, freeFunc = PermanentGlynn_dualSIM_free;
#endif
#else
#ifdef USE_FLOAT
    initFunc = PermanentGlynn_dualDFEF_init, runFunc = (RUNGROUPFUNC)PermanentGlynn_dualDFEF_run_group_nonblock, freeFunc = PermanentGlynn_dualDFEF_free; //runArrayFunc = (RUNARRAYFUNC)PermanentGlynn_dualDFEF_run_array
#else
    initFunc = PermanentGlynn_dualDFE_init, runFunc = (RUNGROUPFUNC)PermanentGlynn_dualDFE_run_group_nonblock, freeFunc = PermanentGlynn_dualDFE_free; //runArrayFunc = (RUNARRAYFUNC)PermanentGlynn_dualDFE_run_array
#endif
#endif  
#endif
	// initialize the max file

#ifdef DEBUG
	printf("Maxfile initialized\n");
#endif
	

  if (!initFunc) return;
  mavMaxFile = initFunc();
  group = max_load_group(mavMaxFile, MAXOS_EXCLUSIVE, "local:*", 2);
#if defined(DUAL) && !defined(MAXELER_SIM)
  //array = max_load_array(mavMaxFile, 2, "*");
#else
  //mavDFE = max_load(mavMaxFile, "local:*");
#endif
  initialized = true;
#ifdef DEBUG
	printf("Maxfile uploaded to DFE\n");
#endif


}




/**
@brief Interface function to releive DFE array
*/
#ifdef USE_FLOAT
void releive_DFEF()
#else
void releive_DFE()
#endif
{

	if (!initialized) return;

#ifdef DEBUG
        printf("Unloading Maxfile\n");
#endif

	// unload the max files from the devices
  initialized = false;
  max_unload_group(group);
#if defined(DUAL) && !defined(MAXELER_SIM)
  //max_unload_array(array);
#else
  //max_unload(mavDFE);
#endif
  max_file_free(mavMaxFile);
  freeFunc();  
}

//DFE float uses IEEE style, not C long double style - bias is 32767 not 16383 (if (16, 64) used so we use (15, 64) for identical bias), mantissa stores 63 bits not 64, must adjust manually
long double dfeFloatToLD(__int128 res)
{
    __int128 temp = res >> 63;
    if ((temp & 0x7FFF) == 0) //+/- 0
        res = (res & ((1ULL<<63)-1)) | (temp << 64);
    else if ((temp & 0x7FFF) == 0x7FFF) //+/- inf or +/- NaN
        res = ((res & ((1ULL<<62)-1)) | (1ULL << 63)) | (temp << 64);
    else
        res = ((res & ((1ULL<<63)-1)) | (1ULL << 63)) | (temp << 64);
    long double* pld = (long double*)&res;
    return *pld;
}

#define MAT_SIZE 40
#define INITS 4
#define COLDIV (MAT_SIZE / INITS)
/**
@brief Interface function to calculate the Permanent using Glynns formula on DFE
*/
#ifdef USE_FLOAT
void calcPermanentGlynnDFEF(const ComplexFix16** mtx_data, const long double* renormalize_data, const uint64_t rows, const uint64_t cols, const uint64_t totalPerms, Complex16* perm)
#else
void calcPermanentGlynnDFE(const ComplexFix16** mtx_data, const long double* renormalize_data, const uint64_t rows, const uint64_t cols, const uint64_t totalPerms, Complex16* perm)
#endif
{
    if (!initialized) return;
    
    uint64_t numOfPartialPerms = rows;

	// variable to store the result
	__int128* res = (__int128*)malloc(sizeof(__int128) * 2 * totalPerms);
#ifdef DUAL
	__int128* res2 = (__int128*)malloc(sizeof(__int128) * 2 * totalPerms);
#endif

union {
#ifdef MAXELER_SIM
#ifdef USE_FLOAT
#ifndef DUAL
      PermanentGlynn_singleSIMF_actions_t glynnRowsGray;
#else
      PermanentGlynn_dualSIMF_actions_t dualGlynnRowsGray;
#endif
#else    
#ifndef DUAL
      PermanentGlynn_singleSIM_actions_t glynnRowsGray;
#else
      PermanentGlynn_dualSIM_actions_t dualGlynnRowsGray;
#endif
#endif
#else
#ifdef USE_FLOAT
#ifndef DUAL
      PermanentGlynn_singleDFEF_actions_t glynnRowsGray;
#else
      PermanentGlynn_dualDFEF_actions_t dualGlynnRowsGray;
#endif
#else
#ifndef DUAL
      PermanentGlynn_singleDFE_actions_t glynnRowsGray;
#else
      PermanentGlynn_dualDFE_actions_t dualGlynnRowsGray;
#endif
#endif
#endif
} actions
#if defined(DUAL) && !defined(MAXELER_SIM)
    , dualactions;
    void *arractions[2] = {&actions, &dualactions};
#else
    ;
#endif

    // simulation
#ifndef DUAL
      actions.glynnRowsGray.param_ticksMax = numOfPartialPerms, actions.glynnRowsGray.param_cols = cols, actions.glynnRowsGray.param_totalPerms = totalPerms;
      actions.glynnRowsGray.outstream_res = res, actions.glynnRowsGray.outstream_size_res = sizeof(__int128)*2*totalPerms;
      actions.glynnRowsGray.instream_InputMtx0 = (__int64_t*)mtx_data[0]; actions.glynnRowsGray.instream_size_InputMtx0 = sizeof(ComplexFix16)*COLDIV*rows*totalPerms;
      actions.glynnRowsGray.instream_InputMtx1 = (__int64_t*)mtx_data[1]; actions.glynnRowsGray.instream_size_InputMtx1 = cols > COLDIV ? sizeof(ComplexFix16)*COLDIV*rows*totalPerms : 0;
      actions.glynnRowsGray.instream_InputMtx2 = (__int64_t*)mtx_data[2]; actions.glynnRowsGray.instream_size_InputMtx2 = cols > 2*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows*totalPerms : 0;
      actions.glynnRowsGray.instream_InputMtx3 = (__int64_t*)mtx_data[3]; actions.glynnRowsGray.instream_size_InputMtx3 = cols > 3*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows*totalPerms : 0;
#else
      //Simulation of manager I/Os of purpose OTHER_FPGA not yet supported.
#ifdef MAXELER_SIM
      actions.dualGlynnRowsGray.param_ticksMax = numOfPartialPerms, actions.dualGlynnRowsGray.param_cols = cols, actions.dualGlynnRowsGray.param_totalPerms = totalPerms;
      actions.dualGlynnRowsGray.outstream_res = res, actions.dualGlynnRowsGray.outstream_size_res = sizeof(__int128)*2*totalPerms, actions.dualGlynnRowsGray.outstream_res2 = res2, actions.dualGlynnRowsGray.outstream_size_res2 = sizeof(__int128)*2*totalPerms;
      actions.dualGlynnRowsGray.instream_InputMtx0 = (__int64_t*)mtx_data[0]; actions.dualGlynnRowsGray.instream_size_InputMtx0 = sizeof(ComplexFix16)*COLDIV*rows*totalPerms;
      actions.dualGlynnRowsGray.instream_InputMtx1 = (__int64_t*)mtx_data[1]; actions.dualGlynnRowsGray.instream_size_InputMtx1 = cols > COLDIV ? sizeof(ComplexFix16)*COLDIV*rows*totalPerms : 0;
      actions.dualGlynnRowsGray.instream_InputMtx2 = (__int64_t*)mtx_data[2]; actions.dualGlynnRowsGray.instream_size_InputMtx2 = cols > 2*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows*totalPerms : 0;
      actions.dualGlynnRowsGray.instream_InputMtx3 = (__int64_t*)mtx_data[3]; actions.dualGlynnRowsGray.instream_size_InputMtx3 = cols > 3*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows*totalPerms : 0;
      actions.dualGlynnRowsGray.instream_InputMtx4 = (__int64_t*)mtx_data[0]; actions.dualGlynnRowsGray.instream_size_InputMtx4 = sizeof(ComplexFix16)*COLDIV*rows*totalPerms;
      actions.dualGlynnRowsGray.instream_InputMtx5 = (__int64_t*)mtx_data[1]; actions.dualGlynnRowsGray.instream_size_InputMtx5 = cols > COLDIV ? sizeof(ComplexFix16)*COLDIV*rows*totalPerms : 0;
      actions.dualGlynnRowsGray.instream_InputMtx6 = (__int64_t*)mtx_data[2]; actions.dualGlynnRowsGray.instream_size_InputMtx6 = cols > 2*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows*totalPerms : 0;
      actions.dualGlynnRowsGray.instream_InputMtx7 = (__int64_t*)mtx_data[3]; actions.dualGlynnRowsGray.instream_size_InputMtx7 = cols > 3*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows*totalPerms : 0;
#else
      actions.dualGlynnRowsGray.param_isLocal = 1, actions.dualGlynnRowsGray.param_ticksMax = numOfPartialPerms, actions.dualGlynnRowsGray.param_cols = cols, actions.dualGlynnRowsGray.param_totalPerms = 1;
      actions.dualGlynnRowsGray.outstream_res = res, actions.dualGlynnRowsGray.outstream_size_res = sizeof(__int128)*2*totalPerms;
      actions.dualGlynnRowsGray.instream_InputMtx0 = (__int64_t*)mtx_data[0]; actions.dualGlynnRowsGray.instream_size_InputMtx0 = sizeof(ComplexFix16)*COLDIV*rows*totalPerms;
      actions.dualGlynnRowsGray.instream_InputMtx1 = (__int64_t*)mtx_data[1]; actions.dualGlynnRowsGray.instream_size_InputMtx1 = cols > COLDIV ? sizeof(ComplexFix16)*COLDIV*rows*totalPerms : 0;
      actions.dualGlynnRowsGray.instream_InputMtx2 = (__int64_t*)mtx_data[2]; actions.dualGlynnRowsGray.instream_size_InputMtx2 = cols > 2*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows*totalPerms : 0;
      actions.dualGlynnRowsGray.instream_InputMtx3 = (__int64_t*)mtx_data[3]; actions.dualGlynnRowsGray.instream_size_InputMtx3 = cols > 3*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows*totalPerms : 0;
      dualactions.dualGlynnRowsGray.param_isLocal = 0, dualactions.dualGlynnRowsGray.param_ticksMax = numOfPartialPerms, dualactions.dualGlynnRowsGray.param_cols = cols, dualactions.dualGlynnRowsGray.param_totalPerms = 1;
      dualactions.dualGlynnRowsGray.outstream_res = res2, dualactions.dualGlynnRowsGray.outstream_size_res = sizeof(__int128)*2*totalPerms;
      dualactions.dualGlynnRowsGray.instream_InputMtx0 = (__int64_t*)mtx_data[0]; dualactions.dualGlynnRowsGray.instream_size_InputMtx0 = sizeof(ComplexFix16)*COLDIV*rows*totalPerms;
      dualactions.dualGlynnRowsGray.instream_InputMtx1 = (__int64_t*)mtx_data[1]; dualactions.dualGlynnRowsGray.instream_size_InputMtx1 = cols > COLDIV ? sizeof(ComplexFix16)*COLDIV*rows*totalPerms : 0;
      dualactions.dualGlynnRowsGray.instream_InputMtx2 = (__int64_t*)mtx_data[2]; dualactions.dualGlynnRowsGray.instream_size_InputMtx2 = cols > 2*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows*totalPerms : 0;
      dualactions.dualGlynnRowsGray.instream_InputMtx3 = (__int64_t*)mtx_data[3]; dualactions.dualGlynnRowsGray.instream_size_InputMtx3 = cols > 3*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows*totalPerms : 0;
#endif
#endif

#ifdef DEBUG
	printf("Start permanent calulation on DFE\n");
#endif

#if defined(DUAL) && !defined(MAXELER_SIM)
    //runArrayFunc(array, arractions);
    max_run_t* run0 = runFunc(group, arractions[0]), *run1 = runFunc(group, arractions[1]); max_wait(run0); max_wait(run1);
    //max_actions_t* dualactions[2] = { PermanentGlynn_dualDFE_convert(mavMaxFile, arractions[0]), PermanentGlynn_dualDFE_convert(mavMaxFile, arractions[1]) }; 
    //max_run_group_multi(group, dualactions);
    //max_actions_free(dualactions[0]), max_actions_free(dualactions[1]);
#else
    //runFunc(mavDFE, &actions);
    runFunc(group, &actions);
#endif

#ifdef DEBUG
	printf("Permanent calulation on DFE finished\n");
#endif

    numOfPartialPerms = 1ULL << (numOfPartialPerms-1);
    for (size_t i = 0; i < totalPerms; i++) {
#ifdef USE_FLOAT
    perm[i].real = dfeFloatToLD(res[i*2]);
    perm[i].imag = dfeFloatToLD(res[i*2+1]);
#ifdef DUAL
    perm[i].real += dfeFloatToLD(res2[i*2]);
    perm[i].imag += dfeFloatToLD(res2[i*2+1]);
#endif
    perm[i].real /= numOfPartialPerms;
    perm[i].imag /= numOfPartialPerms;
#else
#ifdef DUAL
    res[i*2] += res2[i*2], res[i*2+1] += res2[i*2+1];
#endif
    //128-bit fixed point with 124 fractional bits conversion by dividing by 2^124==(2^62)*(2^62) 
    long double factor = (long double)(1ULL<<62);

    long double real = ((long double)res[i*2])/factor/factor;
    long double imag = ((long double)res[i*2+1])/factor/factor;

    real /= numOfPartialPerms;
    imag /= numOfPartialPerms;
    size_t offset = i * cols;
    // renormalize the result according to the normalization of the input matrix
    for (int jdx=0; jdx<cols; jdx++ ) {
        real *= renormalize_data[offset+jdx];
        imag *= renormalize_data[offset+jdx];
    }
    perm[i].real = real; perm[i].imag = imag;
#endif
    }
    free(res);
#ifdef DUAL
    free(res2);
#endif
    return;
}



