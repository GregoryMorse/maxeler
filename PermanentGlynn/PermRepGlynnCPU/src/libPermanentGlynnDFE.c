#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <MaxSLiCInterface.h>

#ifdef MAXELER_SIM
#ifdef USE_FLOAT
#ifndef DUAL
#include "PermRepGlynn_singleSIMF.h"
#define MTX_SIZE PermRepGlynn_singleSIMF_MTXSIZE
#define BASEKERNPOW2 PermRepGlynn_singleSIMF_BASEKERNPOW2
#define INITS PermRepGlynn_singleSIMF_INITKERNS
#define LOOPLENGTH PermRepGlynn_singleSIMF_LOOPLENGTH
#define FREQ PermRepGlynn_singleSIMF_FREQ
#else
#include "PermRepGlynn_dualSIMF.h"
#define MTX_SIZE PermRepGlynn_dualSIMF_MTXSIZE
#define BASEKERNPOW2 PermRepGlynn_dualSIMF_BASEKERNPOW2
#define INITS PermRepGlynn_dualSIMF_INITKERNS
#define LOOPLENGTH PermRepGlynn_dualSIMF_LOOPLENGTH
#define FREQ PermRepGlynn_dualSIMF_FREQ
#endif
#else
#ifndef DUAL
#include "PermRepGlynn_singleSIM.h"
#define MTX_SIZE PermRepGlynn_singleSIM_MTXSIZE
#define BASEKERNPOW2 PermRepGlynn_singleSIM_BASEKERNPOW2
#define INITS PermRepGlynn_singleSIM_INITKERNS
#define LOOPLENGTH PermRepGlynn_singleSIM_LOOPLENGTH
#define FREQ PermRepGlynn_singleSIM_FREQ
#else
#include "PermRepGlynn_dualSIM.h"
#define MTX_SIZE PermRepGlynn_dualSIM_MTXSIZE
#define BASEKERNPOW2 PermRepGlynn_dualSIM_BASEKERNPOW2
#define INITS PermRepGlynn_dualSIM_INITKERNS
#define LOOPLENGTH PermRepGlynn_dualSIM_LOOPLENGTH
#define FREQ PermRepGlynn_dualSIM_FREQ
#endif
#endif
#else
#ifdef USE_FLOAT
#ifndef DUAL
#include "PermRepGlynn_singleDFEF.h"
#define MTX_SIZE PermRepGlynn_singleDFEF_MTXSIZE
#define BASEKERNPOW2 PermRepGlynn_singleDFEF_BASEKERNPOW2
#define INITS PermRepGlynn_singleDFEF_INITKERNS
#define LOOPLENGTH PermRepGlynn_singleDFEF_LOOPLENGTH
#define FREQ PermRepGlynn_singleDFEF_FREQ
#else
#include "PermanentGlynn_dualDFEF.h"
#define MTX_SIZE PermRepGlynn_dualDFEF_MTXSIZE
#define BASEKERNPOW2 PermRepGlynn_dualDFEF_BASEKERNPOW2
#define INITS PermRepGlynn_dualDFEF_INITKERNS
#define LOOPLENGTH PermRepGlynn_dualDFEF_LOOPLENGTH
#define FREQ PermRepGlynn_dualDFEF_FREQ
#endif
#else
#ifndef DUAL
#include "PermRepGlynn_singleDFE.h"
#define MTX_SIZE PermRepGlynn_singleDFE_MTXSIZE
#define BASEKERNPOW2 PermRepGlynn_singleDFE_BASEKERNPOW2
#define INITS PermRepGlynn_singleDFE_INITKERNS
#define LOOPLENGTH PermRepGlynn_singleDFE_LOOPLENGTH
#define FREQ PermRepGlynn_singleDFE_FREQ
#else
#include "PermRepGlynn_dualDFE.h"
#define MTX_SIZE PermRepGlynn_dualDFE_MTXSIZE
#define BASEKERNPOW2 PermRepGlynn_dualDFE_BASEKERNPOW2
#define INITS PermRepGlynn_dualDFE_INITKERNS
#define LOOPLENGTH PermRepGlynn_dualDFE_LOOPLENGTH
#define FREQ PermRepGlynn_dualDFE_FREQ
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

#ifdef USE_FLOAT
void releive_DFEF();
#else
void releive_DFE();
#endif
/**
@brief Interface function to initialize DFE array
*/
#ifdef USE_FLOAT
int initializeRep_DFEF(int groupMode, size_t* mtx_size, size_t* basekernpow2, size_t* loopLength)
#else
int initializeRep_DFE(int groupMode, size_t* mtx_size, size_t* basekernpow2, size_t* loopLength)
#endif
{

	if (initialized) return 1;
  max_file_t* (*initFunc)(void) = NULL;
#ifndef DUAL
#ifdef MAXELER_SIM
#ifdef USE_FLOAT
    initFunc = PermRepGlynn_singleSIMF_init, runFunc = (RUNFUNC)PermRepGlynn_singleSIMF_run, freeFunc = PermRepGlynn_singleSIMF_free;
#else
    initFunc = PermRepGlynn_singleSIM_init, runFunc = (RUNFUNC)PermRepGlynn_singleSIM_run, freeFunc = PermRepGlynn_singleSIM_free;
#endif
#else
#ifdef USE_FLOAT
    initFunc = PermRepGlynn_singleDFEF_init, runFunc = (RUNFUNC)PermRepGlynn_singleDFEF_run, runGroupFunc = (RUNGROUPFUNC)PermRepGlynn_singleDFEF_run_group, freeFunc = PermRepGlynn_singleDFEF_free;
#else
    initFunc = PermRepGlynn_singleDFE_init, runFunc = (RUNFUNC)PermRepGlynn_singleDFE_run, runGroupFunc = (RUNGROUPFUNC)PermRepGlynn_singleDFE_run_group, freeFunc = PermRepGlynn_singleDFE_free;
#endif
#endif  
#else
#ifdef MAXELER_SIM
#ifdef USE_FLOAT
    initFunc = PermRepGlynn_dualSIMF_init, runFunc = (RUNFUNC)PermRepGlynn_dualSIMF_run, freeFunc = PermRepGlynn_dualSIMF_free;
#else
    initFunc = PermRepGlynn_dualSIM_init, runFunc = (RUNFUNC)PermRepGlynn_dualSIM_run, freeFunc = PermRepGlynn_dualSIM_free;
#endif
#else
#ifdef USE_FLOAT
    initFunc = PermRepGlynn_dualDFEF_init, runFunc = (RUNGROUPFUNC)PermRepGlynn_dualDFEF_run_group_nonblock, freeFunc = PermRepGlynn_dualDFEF_free; //runArrayFunc = (RUNARRAYFUNC)PermRepGlynn_dualDFEF_run_array
#else
    initFunc = PermRepGlynn_dualDFE_init, runFunc = (RUNGROUPFUNC)PermRepGlynn_dualDFE_run_group_nonblock, freeFunc = PermRepGlynn_dualDFE_free; //runArrayFunc = (RUNARRAYFUNC)PermRepGlynn_dualDFE_run_array
#endif
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
    *loopLength = LOOPLENGTH;
    return 1;
}




/**
@brief Interface function to releive DFE array
*/
#ifdef USE_FLOAT
void releiveRep_DFEF()
#else
void releiveRep_DFE()
#endif
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

#define COLDIV (MTX_SIZE / INITS)
#define USECOLMUX 0

#ifdef USE_FLOAT
typedef __int128 Fix192;
#else
typedef struct { //little-endian, must be 64-bit aligned
    __uint128_t lowBits;
    __int128 highBits;
} __attribute__((packed)) Fix256;

int fix256to128(Fix256* fix)
{
    long long check = fix->highBits >> 64;
    int nextBit = (fix->highBits & 0x8000000000000000ULL) != 0;
    if ((check == 0 && !nextBit) || (check == -1 && nextBit)) { //run of 65 0s or 65 1s to handle positive/negative
        fix->highBits <<= 64;
        fix->highBits |= fix->lowBits >> 64;
        return 1;
    } else return 0;
}
#endif

uint64_t roundUp(uint64_t num, uint64_t nearest)
{
    return num + (num % nearest == 0 ? 0 : (nearest - num % nearest));
}

/**
@brief Interface function to calculate the Permanent using Glynns formula on DFE
*/
#ifdef USE_FLOAT
void calcPermanentGlynnRepDFEF(const ComplexFix16** mtx_data, const long double* renormalize_data, const uint64_t rows, const uint64_t cols, const unsigned char* colIndices,
  const uint8_t* rowchange_indices, const uint8_t* initDirections, const uint8_t photons, const uint8_t onerows, const uint64_t* mplicity, const uint64_t changecount, const uint8_t mulsum, const int initParities, uint64_t totalPerms, Complex16* perm)
#else
void calcPermanentGlynnRepDFE(const ComplexFix16** mtx_data, const long double* renormalize_data, const uint64_t rows, const uint64_t cols, const unsigned char* colIndices,
  const uint8_t* rowchange_indices, const uint8_t* initDirections, const uint8_t photons, const uint8_t onerows, const uint64_t* mplicity, const uint64_t changecount, const uint8_t mulsum, const int initParities, uint64_t totalPerms, Complex16* perm)
#endif
{
    if (!initialized) return;
    int adjLoopLength = changecount+1 < (unsigned)LOOPLENGTH && rowchange_indices[rows-1] == 1 ? changecount+1 : LOOPLENGTH;
    uint64_t numOfPartialPerms = onerows;
    //numOfPartialPerms = max(numOfPartialPerms, BASEKERNPOW2+1+1+(useDual ? 1 : 0));//extra 1 since maxTicks cannot be 1, minimum of 2

    //printf("%lld, %d\n", numOfPartialPerms, rows);

	// variable to store the result
    size_t resbytes = sizeof(Fix256) * 2 * totalPerms; //*(changecount+1);
    Fix256* res = (Fix256*)malloc(resbytes);
#ifdef DUAL
    Fix256* res2 = (Fix256*)malloc(resbytes);
#endif

    union {
#ifdef MAXELER_SIM
#ifdef USE_FLOAT
#ifndef DUAL
      PermRepGlynn_singleSIMF_actions_t glynnRowsGray;
#else
      PermRepGlynn_dualSIMF_actions_t dualGlynnRowsGray;
#endif
#else
#ifndef DUAL
      PermRepGlynn_singleSIM_actions_t glynnRowsGray;
#else
      PermRepGlynn_dualSIM_actions_t dualGlynnRowsGray;
#endif
#endif
#else
#ifdef USE_FLOAT
#ifndef DUAL
      PermRepGlynn_singleDFEF_actions_t glynnRowsGray;
#else
      PermRepGlynn_dualDFEF_actions_t dualGlynnRowsGray;
#endif
#else
#ifndef DUAL
      PermRepGlynn_singleDFE_actions_t glynnRowsGray;
#else
      PermRepGlynn_dualDFE_actions_t dualGlynnRowsGray;
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
      actions.glynnRowsGray.outstream_res = res, actions.glynnRowsGray.outstream_size_res = resbytes;
      actions.glynnRowsGray.param_totalPerms = totalPerms, actions.glynnRowsGray.param_initParities = initParities,
      actions.glynnRowsGray.param_msize = cols, actions.glynnRowsGray.param_photons = photons, actions.glynnRowsGray.param_changeCount = changecount+1;
      actions.glynnRowsGray.instream_InputMtx0 = (__int64_t*)mtx_data[0]; actions.glynnRowsGray.instream_size_InputMtx0 = sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms;
      actions.glynnRowsGray.instream_InputMtx1 = (__int64_t*)mtx_data[1]; actions.glynnRowsGray.instream_size_InputMtx1 = cols > COLDIV ? sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms : 0;
      actions.glynnRowsGray.instream_InputMtx2 = (__int64_t*)mtx_data[2]; actions.glynnRowsGray.instream_size_InputMtx2 = cols > 2*COLDIV ? sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms : 0;
      actions.glynnRowsGray.instream_InputMtx3 = (__int64_t*)mtx_data[3]; actions.glynnRowsGray.instream_size_InputMtx3 = cols > 3*COLDIV ? sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms : 0;
      //max_actions_t* mat = PermRepGlynn_singleSIM_convert(mavMaxFile, &actions.glynnRowsGray);
      //int loopLength = max_get_offset_auto_loop_size(mat, "InitializeColSumDFEKernel_0", "loopLength");
      //printf("Loop Length: %d\n", loopLength);
      //max_actions_free(mat);
#else
      //Simulation of manager I/Os of purpose OTHER_FPGA not yet supported.
#ifdef MAXELER_SIM
      actions.dualGlynnRowsGray.outstream_res = res, actions.dualGlynnRowsGray.outstream_size_res = resbytes;
      actions.dualGlynnRowsGray.outstream_res2 = res2, actions.dualGlynnRowsGray.outstream_size_res2 = resbytes;
      actions.dualGlynnRowsGray.param_totalPerms = totalPerms, actions.dualGlynnRowsGray.param_initParities = initParities,
      actions.dualGlynnRowsGray.param_msize = cols, actions.dualGlynnRowsGray.param_photons = photons, actions.dualGlynnRowsGray.param_changeCount = changecount+1;
      actions.dualGlynnRowsGray.instream_InputMtx0 = (__int64_t*)mtx_data[0]; actions.dualGlynnRowsGray.instream_size_InputMtx0 = sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms;
      actions.dualGlynnRowsGray.instream_InputMtx1 = (__int64_t*)mtx_data[1]; actions.dualGlynnRowsGray.instream_size_InputMtx1 = cols > COLDIV ? sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms : 0;
      actions.dualGlynnRowsGray.instream_InputMtx2 = (__int64_t*)mtx_data[2]; actions.dualGlynnRowsGray.instream_size_InputMtx2 = cols > 2*COLDIV ? sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms : 0;
      actions.dualGlynnRowsGray.instream_InputMtx3 = (__int64_t*)mtx_data[3]; actions.dualGlynnRowsGray.instream_size_InputMtx3 = cols > 3*COLDIV ? sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms : 0;
      actions.dualGlynnRowsGray.instream_InputMtx4 = (__int64_t*)mtx_data[0]; actions.dualGlynnRowsGray.instream_size_InputMtx4 = sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms;
      actions.dualGlynnRowsGray.instream_InputMtx5 = (__int64_t*)mtx_data[1]; actions.dualGlynnRowsGray.instream_size_InputMtx5 = cols > COLDIV ? sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms : 0;
      actions.dualGlynnRowsGray.instream_InputMtx6 = (__int64_t*)mtx_data[2]; actions.dualGlynnRowsGray.instream_size_InputMtx6 = cols > 2*COLDIV ? sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms : 0;
      actions.dualGlynnRowsGray.instream_InputMtx7 = (__int64_t*)mtx_data[3]; actions.dualGlynnRowsGray.instream_size_InputMtx7 = cols > 3*COLDIV ? sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms : 0;
#else
      actions.dualGlynnRowsGray.param_isLocal = 1, actions.dualGlynnRowsGray.outstream_res = res, actions.dualGlynnRowsGray.outstream_size_res = resbytes;
      actions.dualGlynnRowsGray.param_totalPerms = totalPerms, actions.dualGlynnRowsGray.param_initParities = initParities,
      actions.dualGlynnRowsGray.param_msize = cols, actions.dualGlynnRowsGray.param_photons = photons, actions.dualGlynnRowsGray.param_changeCount = changecount+1;
      actions.dualGlynnRowsGray.instream_InputMtx0 = (__int64_t*)mtx_data[0]; actions.dualGlynnRowsGray.instream_size_InputMtx0 = sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms;
      actions.dualGlynnRowsGray.instream_InputMtx1 = (__int64_t*)mtx_data[1]; actions.dualGlynnRowsGray.instream_size_InputMtx1 = cols > COLDIV ? sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms : 0;
      actions.dualGlynnRowsGray.instream_InputMtx2 = (__int64_t*)mtx_data[2]; actions.dualGlynnRowsGray.instream_size_InputMtx2 = cols > 2*COLDIV ? sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms : 0;
      actions.dualGlynnRowsGray.instream_InputMtx3 = (__int64_t*)mtx_data[3]; actions.dualGlynnRowsGray.instream_size_InputMtx3 = cols > 3*COLDIV ? sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms : 0;
      dualactions.dualGlynnRowsGray.param_isLocal = 0, dualactions.dualGlynnRowsGray.outstream_res = res2, dualactions.dualGlynnRowsGray.outstream_size_res = resbytes;
      dualactions.dualGlynnRowsGray.param_totalPerms = totalPerms, dualactions.dualGlynnRowsGray.param_initParities = initParities,
      dualactions.dualGlynnRowsGray.param_msize = cols, dualactions.dualGlynnRowsGray.param_photons = photons, dualactions.dualGlynnRowsGray.param_changeCount = changecount+1;
      dualactions.dualGlynnRowsGray.instream_InputMtx0 = (__int64_t*)mtx_data[0]; dualactions.dualGlynnRowsGray.instream_size_InputMtx0 = sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms;
      dualactions.dualGlynnRowsGray.instream_InputMtx1 = (__int64_t*)mtx_data[1]; dualactions.dualGlynnRowsGray.instream_size_InputMtx1 = cols > COLDIV ? sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms : 0;
      dualactions.dualGlynnRowsGray.instream_InputMtx2 = (__int64_t*)mtx_data[2]; dualactions.dualGlynnRowsGray.instream_size_InputMtx2 = cols > 2*COLDIV ? sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms : 0;
      dualactions.dualGlynnRowsGray.instream_InputMtx3 = (__int64_t*)mtx_data[3]; dualactions.dualGlynnRowsGray.instream_size_InputMtx3 = cols > 3*COLDIV ? sizeof(ComplexFix16)*(COLDIV+1)*((rows-1)*LOOPLENGTH+adjLoopLength)*totalPerms : 0;   
#endif
#endif

#ifdef DEBUG
	printf("Start permanent calulation on DFE\n");
#endif

#ifdef MAXELER_SIM
    max_config_set_int64(MAX_CONFIG_PCIE_TIMEOUT, totalPerms*(30+(changecount+1)/(FREQ*1000ULL)));
    max_config_set_int64(MAX_CONFIG_ACTION_TIMEOUT, totalPerms*(30+(changecount+1)/(FREQ*1000ULL)));
#else
    max_config_set_int64(MAX_CONFIG_PCIE_TIMEOUT, totalPerms*(30+(changecount+1)/(FREQ*1000000ULL)));
    max_config_set_int64(MAX_CONFIG_ACTION_TIMEOUT, totalPerms*(30+(changecount+1)/(FREQ*1000000ULL)));
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

    //perm->real = 0, perm->imag = 0;
    //int parity = 0;
    for (uint64_t i = 0; i < totalPerms; i++) {
#ifdef USE_FLOAT
#ifdef DUAL
    perm[i].real = dfeFloatToLD(res[i*2]) + dfeFloatToLD(res2[i*2]);
    perm[i].imag = dfeFloatToLD(res[i*2+1]) + dfeFloatToLD(res2[i*2+1]);
#else
    perm[i].real = dfeFloatToLD(res[i*2]);
    perm[i].imag = dfeFloatToLD(res[i*2+1]);
#endif
    perm[i].real = ldexp(perm[i].real, -(mulsum + numOfPartialPerms-1));
    perm[i].imag = ldexp(perm[i].imag, -(mulsum + numOfPartialPerms-1));
#else
#ifdef DUAL
    int ha = res[i*2].lowBits >> 63, ha2 = res2[i*2].lowBits >> 63;
    int hb = res[i*2+1].lowBits >> 63, hb2 = res2[i*2+1].lowBits >> 63;
    res[i*2].lowBits += res2[i*2].lowBits, res[i*2+1].lowBits += res2[i*2+1].lowBits;
    res[i*2].highBits += res2[i*2].highBits + ((ha & ha2) | ((ha ^ ha2) & !(res[i*2].lowBits >> 63)));
    res[i*2+1].highBits += res2[i*2+1].highBits + ((hb & hb2) | ((hb ^ hb2) & !(res[i*2+1].lowBits >> 63)));
#endif
    int adjust1 = fix256to128(&res[i*2]), adjust2 = fix256to128(&res[i*2+1]); //start with (256, -250) adjust=0 then (128, -122) else (128, -186)
        long double real = ldexpl((long double)res[i*2].highBits, (adjust1 ? -186 : -122) - (mulsum + numOfPartialPerms-1)),
                    imag = ldexpl((long double)res[i*2+1].highBits, (adjust2 ? -186 : -122) - (mulsum + numOfPartialPerms-1));
        // renormalize the result according to the normalization of the input matrix
        for (int jdx=0; jdx<photons; jdx++ ) {
            real *= renormalize_data[colIndices[i*photons+jdx]];
            imag *= renormalize_data[colIndices[i*photons+jdx]];
        }
        perm[i].real = real; perm[i].imag = imag;
        //real *= mplicity[i];
        //imag *= mplicity[i];
    
        //printf("%llu, %Lf %Lf\n", mplicity[i], real, imag);
        //if (parity) {
        //    perm->real -= real; perm->imag -= imag;
        //} else {
        //    perm->real += real; perm->imag += imag;
        //}
        //parity = ~parity;
#endif
    }
    free(res);
#ifdef DUAL
    free(res2);
#endif
    return;
}



