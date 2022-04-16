#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <MaxSLiCInterface.h>

#ifdef MAXELER_SIM
#ifndef DUAL
#include "PermRepGlynn_singleSIM.h"
#define MTX_SIZE PermRepGlynn_singleSIM_MTXSIZE
#define BASEKERNPOW2 PermRepGlynn_singleSIM_BASEKERNPOW2
#else
#include "PermRepGlynn_dualSIM.h"
#define MTX_SIZE PermRepGlynn_dualSIM_MTXSIZE
#define BASEKERNPOW2 PermRepGlynn_dualSIM_BASEKERNPOW2
#endif
#else
#ifndef DUAL
#include "PermRepGlynn_singleDFE.h"
#define MTX_SIZE PermRepGlynn_singleDFE_MTXSIZE
#define BASEKERNPOW2 PermRepGlynn_singleDFE_BASEKERNPOW2
#else
#include "PermRepGlynn_dualDFE.h"
#define MTX_SIZE PermRepGlynn_dualDFE_MTXSIZE
#define BASEKERNPOW2 PermRepGlynn_dualDFE_BASEKERNPOW2
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
#if defined(DUAL) && !defined(MAXELER_SIM)
//typedef void (*RUNARRAYFUNC)(max_engarray_t*, void**);
//static max_engarray_t* array = NULL;
//static RUNARRAYFUNC runArrayFunc;
static max_group_t* group = NULL;
typedef max_run_t*(*RUNGROUPFUNC)(max_group_t*, void*);
static RUNGROUPFUNC runFunc;
#elif !defined(MAXELER_SIM)
static max_group_t* group = NULL;
typedef void(*RUNGROUPFUNC)(max_group_t*, void*);
static max_engine_t* mavDFE;
static RUNGROUPFUNC runGroupFunc;
static RUNFUNC runFunc;
static bool useGroup = 1;
#else
static max_engine_t* mavDFE;
static RUNFUNC runFunc;
#endif

void releiveRep_DFE();
/**
@brief Interface function to initialize DFE array
*/
int initializeRep_DFE(int groupMode, size_t* mtx_size, size_t* basekernpow2)
{

	if (initialized) return 1;
  max_file_t* (*initFunc)(void) = NULL;
#ifndef DUAL
#ifdef MAXELER_SIM
    initFunc = PermRepGlynn_singleSIM_init, runFunc = (RUNFUNC)PermRepGlynn_singleSIM_run, freeFunc = PermRepGlynn_singleSIM_free;
#else
    initFunc = PermRepGlynn_singleDFE_init, runFunc = (RUNFUNC)PermRepGlynn_singleDFE_run, runGroupFunc = (RUNGROUPFUNC)PermRepGlynn_singleDFE_run_group, freeFunc = PermRepGlynn_singleDFE_free;
#endif  
#else
#ifdef MAXELER_SIM
    initFunc = PermRepGlynn_dualSIM_init, runFunc = (RUNFUNC)PermRepGlynn_dualSIM_run, freeFunc = PermRepGlynn_dualSIM_free;
#else
    initFunc = PermRepGlynn_dualDFE_init, runFunc = (RUNGROUPFUNC)PermRepGlynn_dualDFE_run_group_nonblock, freeFunc = PermRepGlynn_dualDFE_free; //runArrayFunc = (RUNARRAYFUNC)PermRepGlynn_dualDFE_run_array
#endif  
#endif
	// initialize the max file

#ifdef DEBUG
	printf("Maxfile initialized\n");
#endif
	

  if (!initFunc) return 0;
  mavMaxFile = initFunc();
  if (!mavMaxFile) return 0;
//#if defined(DUAL) && !defined(MAXELER_SIM)
  //array = max_load_array(mavMaxFile, 2, "*");
  //if (!array) { max_file_free(mavMaxFile); return 0; }
#if defined(DUAL) && !defined(MAXELER_SIM)
  group = max_load_group(mavMaxFile, MAXOS_EXCLUSIVE, "local:*", 2);
  if (!group) { max_file_free(mavMaxFile); return 0; }
#elif !defined(MAXELER_SIM)
  if (groupMode) {
      group = max_load_group(mavMaxFile, MAXOS_EXCLUSIVE, "local:*", 2);
      if (!group) { max_file_free(mavMaxFile); return 0; }
  } else {
      mavDFE = max_load(mavMaxFile, "local:*");
      if (!mavDFE) { max_file_free(mavMaxFile); return 0; }
  }
  useGroup = groupMode;
#else
  mavDFE = max_load(mavMaxFile, "local:*");
  if (!mavDFE) { max_file_free(mavMaxFile); return 0; }
#endif
  initialized = true;
#ifdef DEBUG
	printf("Maxfile uploaded to DFE\n");
#endif

    *mtx_size = MTX_SIZE;
    *basekernpow2 = BASEKERNPOW2;
    return 1;
}




/**
@brief Interface function to releive DFE array
*/
void releiveRep_DFE()
{

	if (!initialized) return;

#ifdef DEBUG
        printf("Unloading Maxfile\n");
#endif

	// unload the max files from the devices
  initialized = false;
//#if defined(DUAL) && !defined(MAXELER_SIM)
  //max_unload_array(array);
#if defined(DUAL) && !defined(MAXELER_SIM)
  max_unload_group(group);
#elif !defined(MAXELER_SIM)
  if (useGroup) max_unload_group(group);
  else max_unload(mavDFE);
#else
  max_unload(mavDFE);
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

#define INITS 4
#define COLDIV (MTX_SIZE / INITS)

/**
@brief Interface function to calculate the Permanent using Glynns formula on DFE
*/
void calcPermanentGlynnRepDFE(const ComplexFix16** mtx_data, const long double* renormalize_data, const uint64_t rows, const uint64_t cols, const unsigned char* colIndices,
  const uint8_t* rowchange_indices, const uint8_t photons, const uint8_t onerows, const uint64_t* mplicity, const uint64_t changecount, const uint8_t mulsum, Complex16* perm)
{
    if (!initialized) return;
    
    uint64_t numOfPartialPerms = onerows;
    //numOfPartialPerms = max(numOfPartialPerms, BASEKERNPOW2+1+1+(useDual ? 1 : 0));//extra 1 since maxTicks cannot be 1, minimum of 2

    //printf("%lld, %d\n", numOfPartialPerms, rows);

	// variable to store the result
	//__int128 res[2];
    size_t resbytes = sizeof(__int128)*2*(changecount+1);
    __int128* res = (__int128*)malloc(resbytes);
#ifdef DUAL
    __int128* res2 = (__int128*)malloc(resbytes);
#endif

    union {
#ifdef MAXELER_SIM    
#ifndef DUAL
      PermRepGlynn_singleSIM_actions_t glynnRowsGray;
#else
      PermRepGlynn_dualSIM_actions_t dualGlynnRowsGray;
#endif
#else
#ifndef DUAL
      PermRepGlynn_singleDFE_actions_t glynnRowsGray;
#else
      PermRepGlynn_dualDFE_actions_t dualGlynnRowsGray;
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
      actions.glynnRowsGray.param_ticksMax = numOfPartialPerms, actions.glynnRowsGray.outstream_res = res, actions.glynnRowsGray.outstream_size_res = resbytes;
      actions.glynnRowsGray.param_rows = rows, actions.glynnRowsGray.param_msize = cols, actions.glynnRowsGray.param_photons = photons, actions.glynnRowsGray.param_changeCount = changecount+1;
      actions.glynnRowsGray.instream_InputMtx0 = (__int64_t*)mtx_data[0]; actions.glynnRowsGray.instream_size_InputMtx0 = sizeof(ComplexFix16)*COLDIV*rows;
      actions.glynnRowsGray.instream_InputMtx1 = (__int64_t*)mtx_data[1]; actions.glynnRowsGray.instream_size_InputMtx1 = cols > COLDIV ? sizeof(ComplexFix16)*COLDIV*rows : 0;
      actions.glynnRowsGray.instream_InputMtx2 = (__int64_t*)mtx_data[2]; actions.glynnRowsGray.instream_size_InputMtx2 = cols > 2*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows : 0;
      actions.glynnRowsGray.instream_InputMtx3 = (__int64_t*)mtx_data[3]; actions.glynnRowsGray.instream_size_InputMtx3 = cols > 3*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows : 0;
      actions.glynnRowsGray.instream_colIndex = colIndices;
      actions.glynnRowsGray.instream_size_colIndex = photons + (photons % 16 == 0 ? 0 : (16 - photons % 16));
      actions.glynnRowsGray.instream_rowChangeIndices = rowchange_indices;
      actions.glynnRowsGray.instream_size_rowChangeIndices = photons + (photons % 16 == 0 ? 0 : (16 - photons % 16));
      actions.glynnRowsGray.routing_string = "colIndex0 -> colIndexFanout, colIndex1 -> colIndexFanout, colIndex2 -> colIndexFanout, colIndex3 -> colIndexFanout, "
        "rowChangeIndices0 -> rowChangeIndicesFanout, rowChangeIndices1 -> rowChangeIndicesFanout, rowChangeIndices2 -> rowChangeIndicesFanout, rowChangeIndices3 -> rowChangeIndicesFanout, "
        "curMplicity0 -> binCoeffFanout, curMplicity1 -> binCoeffFanout, curMplicity2 -> binCoeffFanout, curMplicity3 -> binCoeffFanout";
      //max_actions_t* mat = PermRepGlynn_singleSIM_convert(mavMaxFile, &actions.glynnRowsGray);
      //int loopLength = max_get_offset_auto_loop_size(mat, "InitializeColSumDFEKernel_0", "loopLength");
      //printf("Loop Length: %d\n", loopLength);
      //max_actions_free(mat);
#else
      //Simulation of manager I/Os of purpose OTHER_FPGA not yet supported.
#ifdef MAXELER_SIM
      actions.dualGlynnRowsGray.param_ticksMax = numOfPartialPerms, actions.dualGlynnRowsGray.outstream_res = res, actions.dualGlynnRowsGray.outstream_size_res = resbytes;
      actions.dualGlynnRowsGray.outstream_res2 = res2, actions.dualGlynnRowsGray.outstream_size_res2 = resbytes;
      actions.dualGlynnRowsGray.param_rows = rows, actions.dualGlynnRowsGray.param_msize = cols, actions.dualGlynnRowsGray.param_photons = photons, actions.dualGlynnRowsGray.param_changeCount = changecount+1;
      actions.dualGlynnRowsGray.instream_InputMtx0 = (__int64_t*)mtx_data[0]; actions.dualGlynnRowsGray.instream_size_InputMtx0 = sizeof(ComplexFix16)*COLDIV*rows;
      actions.dualGlynnRowsGray.instream_InputMtx1 = (__int64_t*)mtx_data[1]; actions.dualGlynnRowsGray.instream_size_InputMtx1 = cols > COLDIV ? sizeof(ComplexFix16)*COLDIV*rows : 0;
      actions.dualGlynnRowsGray.instream_InputMtx2 = (__int64_t*)mtx_data[2]; actions.dualGlynnRowsGray.instream_size_InputMtx2 = cols > 2*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows : 0;
      actions.dualGlynnRowsGray.instream_InputMtx3 = (__int64_t*)mtx_data[3]; actions.dualGlynnRowsGray.instream_size_InputMtx3 = cols > 3*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows : 0;
      actions.dualGlynnRowsGray.instream_InputMtx4 = (__int64_t*)mtx_data[0]; actions.dualGlynnRowsGray.instream_size_InputMtx4 = sizeof(ComplexFix16)*COLDIV*rows;
      actions.dualGlynnRowsGray.instream_InputMtx5 = (__int64_t*)mtx_data[1]; actions.dualGlynnRowsGray.instream_size_InputMtx5 = cols > COLDIV ? sizeof(ComplexFix16)*COLDIV*rows : 0;
      actions.dualGlynnRowsGray.instream_InputMtx6 = (__int64_t*)mtx_data[2]; actions.dualGlynnRowsGray.instream_size_InputMtx6 = cols > 2*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows : 0;
      actions.dualGlynnRowsGray.instream_InputMtx7 = (__int64_t*)mtx_data[3]; actions.dualGlynnRowsGray.instream_size_InputMtx7 = cols > 3*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows : 0;
      actions.dualGlynnRowsGray.instream_colIndex = colIndices;
      actions.dualGlynnRowsGray.instream_size_colIndex = photons + (photons % 16 == 0 ? 0 : (16 - photons % 16));
      actions.dualGlynnRowsGray.instream_rowChangeIndices = rowchange_indices;
      actions.dualGlynnRowsGray.instream_size_rowChangeIndices = photons + (photons % 16 == 0 ? 0 : (16 - photons % 16));
      actions.dualGlynnRowsGray.routing_string = "colIndex0 -> colIndexFanout, colIndex1 -> colIndexFanout, colIndex2 -> colIndexFanout, colIndex3 -> colIndexFanout, "
                                                 "colIndex4 -> colIndexFanout, colIndex5 -> colIndexFanout, colIndex6 -> colIndexFanout, colIndex7 -> colIndexFanout, "
                                                 "rowChangeIndices0 -> rowChangeIndicesFanout, rowChangeIndices1 -> rowChangeIndicesFanout, rowChangeIndices2 -> rowChangeIndicesFanout, rowChangeIndices3 -> rowChangeIndicesFanout, "
                                                 "rowChangeIndices4 -> rowChangeIndicesFanout, rowChangeIndices5 -> rowChangeIndicesFanout, rowChangeIndices6 -> rowChangeIndicesFanout, rowChangeIndices7 -> rowChangeIndicesFanout, "
                                                 "curMplicity0 -> binCoeffFanout, curMplicity1 -> binCoeffFanout, curMplicity2 -> binCoeffFanout, curMplicity3 -> binCoeffFanout, "
                                                 "curMplicity4 -> binCoeffFanout2, curMplicity5 -> binCoeffFanout2, curMplicity6 -> binCoeffFanout2, curMplicity7 -> binCoeffFanout2";
#else
      actions.dualGlynnRowsGray.param_isLocal = 1, actions.dualGlynnRowsGray.param_ticksMax = numOfPartialPerms, actions.dualGlynnRowsGray.outstream_res = res, actions.dualGlynnRowsGray.outstream_size_res = resbytes;
      actions.dualGlynnRowsGray.instream_InputMtx0 = (__int64_t*)mtx_data[0]; actions.dualGlynnRowsGray.instream_size_InputMtx0 = sizeof(ComplexFix16)*COLDIV*rows;
      actions.dualGlynnRowsGray.instream_InputMtx1 = (__int64_t*)mtx_data[1]; actions.dualGlynnRowsGray.instream_size_InputMtx1 = cols > COLDIV ? sizeof(ComplexFix16)*COLDIV*rows : 0;
      actions.dualGlynnRowsGray.instream_InputMtx2 = (__int64_t*)mtx_data[2]; actions.dualGlynnRowsGray.instream_size_InputMtx2 = cols > 2*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows : 0;
      actions.dualGlynnRowsGray.instream_InputMtx3 = (__int64_t*)mtx_data[3]; actions.dualGlynnRowsGray.instream_size_InputMtx3 = cols > 3*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows : 0;
      dualactions.dualGlynnRowsGray.param_isLocal = 0, dualactions.dualGlynnRowsGray.param_ticksMax = numOfPartialPerms, dualactions.dualGlynnRowsGray.outstream_res = NULL, dualactions.dualGlynnRowsGray.outstream_size_res = 0;
      dualactions.dualGlynnRowsGray.instream_InputMtx0 = (__int64_t*)mtx_data[0]; dualactions.dualGlynnRowsGray.instream_size_InputMtx0 = sizeof(ComplexFix16)*COLDIV*rows;
      dualactions.dualGlynnRowsGray.instream_InputMtx1 = (__int64_t*)mtx_data[1]; dualactions.dualGlynnRowsGray.instream_size_InputMtx1 = cols > COLDIV ? sizeof(ComplexFix16)*COLDIV*rows : 0;
      dualactions.dualGlynnRowsGray.instream_InputMtx2 = (__int64_t*)mtx_data[2]; dualactions.dualGlynnRowsGray.instream_size_InputMtx2 = cols > 2*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows : 0;
      dualactions.dualGlynnRowsGray.instream_InputMtx3 = (__int64_t*)mtx_data[3]; dualactions.dualGlynnRowsGray.instream_size_InputMtx3 = cols > 3*COLDIV ? sizeof(ComplexFix16)*COLDIV*rows : 0;
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
#elif !defined(MAXELER_SIM)
    if (useGroup) runGroupFunc(group, &actions);
    else runFunc(mavDFE, &actions);
#else
    runFunc(mavDFE, &actions);
#endif

#ifdef DEBUG
	printf("Permanent calulation on DFE finished\n");
#endif

    //128-bit fixed point with 124 fractional bits conversion by dividing by 2^124==(2^62)*(2^62) 
    numOfPartialPerms = 1ULL << (numOfPartialPerms-1);
    long double factor = (long double)(1ULL<<62);
    perm->real = 0, perm->imag = 0;
    int parity = 0;
    for (uint64_t i = 0; i < changecount+1; i++) {
#ifdef DUAL
        res[i*2] += res2[i*2];
        res[i*2+1] += res2[i*2+1];
#endif    
        long double real = ((long double)res[i*2])/factor/factor;
        long double imag = ((long double)res[i*2+1])/factor/factor;
    
        real /= numOfPartialPerms;
        imag /= numOfPartialPerms;
        real *= mplicity[i];
        imag *= mplicity[i];
    
        //printf("%llu, %Lf %Lf\n", mplicity[i], real, imag);
        if (parity) {
            perm->real -= real; perm->imag -= imag;
        } else {
            perm->real += real; perm->imag += imag;
        }
        parity = ~parity;
    }
    // renormalize the result according to the normalization of the input matrix
    for (int jdx=0; jdx<photons; jdx++ ) {
        perm->real *= renormalize_data[colIndices[jdx]];
        perm->imag *= renormalize_data[colIndices[jdx]];
    }
    uint64_t mulSumPerms = 1ULL << mulsum;
    perm->real /= mulSumPerms, perm->imag /= mulSumPerms;
    free(res);
#ifdef DUAL
    free(res2);
#endif
    return;
}



