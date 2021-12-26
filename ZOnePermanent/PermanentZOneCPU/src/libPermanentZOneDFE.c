#include <stdint.h>
#include <stdlib.h>
#include <MaxSLiCInterface.h>

#ifdef MAXELER_SIM
#include "PermanentZOneSIM.h"
#include "PermanentZOneGraySIM.h"
#include "PermanentZOneRowsSIM.h"
#include "PermanentZOneRowsGraySIM.h"
#include "PermanentZOneGlynnSIM.h"
#include "PermanentZOneGlynnGraySIM.h"
#include "PermanentZOneGlynnRowsSIM.h"
#include "PermanentZOneGlynnRowsGraySIM.h"
#include "PermanentZOneDualSIM.h"
#include "PermanentZOneGrayDualSIM.h"
#include "PermanentZOneRowsDualSIM.h"
#include "PermanentZOneRowsGrayDualSIM.h"
#include "PermanentZOneGlynnDualSIM.h"
#include "PermanentZOneGlynnGrayDualSIM.h"
#include "PermanentZOneGlynnRowsDualSIM.h"
#include "PermanentZOneGlynnRowsGrayDualSIM.h"
#else
#include "PermanentZOneRowsGrayDFE.h"
//#include "PermanentZOneGlynnRowsGrayDFE.h"
#include "PermanentZOneRowsGrayDualDFE.h"
//#include "PermanentZOneGlynnRowsGrayDualDFE.h"
#endif

/// static variable to indicate whether DFE is initialized
typedef void (*RUNFUNC)(max_engine_t*, void*);
static bool initialized = false, isGray, isRows, useGlynn, useDual;
static max_file_t* mavMaxFile;
static max_engine_t* mavDFE;
static void (*freeFunc)(void);
static RUNFUNC runFunc;
#ifndef MAXELER_SIM
typedef void (*RUNARRAYFUNC)(max_engarray_t*, void**);
static max_engarray_t* array = NULL;
static RUNARRAYFUNC runArrayFunc;
#endif

/**
@brief Interface function to initialize DFE array
*/
void releive_ZOne_DFE();
void initialize_ZOne_DFE(int gray, int rows, int glynn, int dual)
{
	if (initialized) {
    if (gray==isGray && rows==isRows && glynn==useGlynn && dual==useDual) return;
    else releive_ZOne_DFE();
  } 
  max_file_t* (*initFunc)(void) = NULL;
  isGray = gray, isRows = rows, useGlynn = glynn, useDual = dual;
  if (!useDual) {
#ifdef MAXELER_SIM  
    if (!isGray && !isRows && !useGlynn) {
      initFunc = PermanentZOneSIM_init, runFunc = (RUNFUNC)PermanentZOneSIM_run, freeFunc = PermanentZOneSIM_free;
    } else if (isGray && !isRows && !useGlynn) {
      initFunc = PermanentZOneGraySIM_init, runFunc = (RUNFUNC)PermanentZOneGraySIM_run, freeFunc = PermanentZOneGraySIM_free;
    } else if (!isGray && isRows && !useGlynn) {
      initFunc = PermanentZOneRowsSIM_init, runFunc = (RUNFUNC)PermanentZOneRowsSIM_run, freeFunc = PermanentZOneRowsSIM_free;
    } else if (isGray && isRows && !useGlynn) {
      initFunc = PermanentZOneRowsGraySIM_init, runFunc = (RUNFUNC)PermanentZOneRowsGraySIM_run, freeFunc = PermanentZOneRowsGraySIM_free;
    } else if (!isGray && !isRows && useGlynn) {
      initFunc = PermanentZOneGlynnSIM_init, runFunc = (RUNFUNC)PermanentZOneGlynnSIM_run, freeFunc = PermanentZOneGlynnSIM_free;
    } else if (isGray && !isRows && useGlynn) {
      initFunc = PermanentZOneGlynnGraySIM_init, runFunc = (RUNFUNC)PermanentZOneGlynnGraySIM_run, freeFunc = PermanentZOneGlynnGraySIM_free;
    } else if (!isGray && isRows && useGlynn) {
      initFunc = PermanentZOneGlynnRowsSIM_init, runFunc = (RUNFUNC)PermanentZOneGlynnRowsSIM_run, freeFunc = PermanentZOneGlynnRowsSIM_free;
    } else if (isGray && isRows && useGlynn) {
      initFunc = PermanentZOneGlynnRowsGraySIM_init, runFunc = (RUNFUNC)PermanentZOneGlynnRowsGraySIM_run, freeFunc = PermanentZOneGlynnRowsGraySIM_free;
    }
#else
    if (isGray && isRows && !useGlynn) {
      initFunc = PermanentZOneRowsGrayDFE_init, runFunc = (RUNFUNC)PermanentZOneRowsGrayDFE_run, freeFunc = PermanentZOneRowsGrayDFE_free;
    } else if (isGray && isRows && useGlynn) {
      //initFunc = PermanentZOneGlynnRowsGrayDFE_init, runFunc = (RUNFUNC)PermanentZOneGlynnRowsGrayDFE_run, freeFunc = PermanentZOneGlynnRowsGrayDFE_free;
    }
#endif
  } else {
    //Simulation of manager I/Os of purpose OTHER_FPGA not yet supported.
#ifdef MAXELER_SIM
    if (!isGray && !isRows && !useGlynn) {
      initFunc = PermanentZOneDualSIM_init, runFunc = (RUNFUNC)PermanentZOneDualSIM_run, freeFunc = PermanentZOneDualSIM_free;
    } else if (isGray && !isRows && !useGlynn) {
      initFunc = PermanentZOneGrayDualSIM_init, runFunc = (RUNFUNC)PermanentZOneGrayDualSIM_run, freeFunc = PermanentZOneGrayDualSIM_free;
    } else if (!isGray && isRows && !useGlynn) {
      initFunc = PermanentZOneRowsDualSIM_init, runFunc = (RUNFUNC)PermanentZOneRowsDualSIM_run, freeFunc = PermanentZOneRowsDualSIM_free;
    } else if (isGray && isRows && !useGlynn) {
      initFunc = PermanentZOneRowsGrayDualSIM_init, runFunc = (RUNFUNC)PermanentZOneRowsGrayDualSIM_run, freeFunc = PermanentZOneRowsGrayDualSIM_free;
    } else if (!isGray && !isRows && useGlynn) {
      initFunc = PermanentZOneGlynnDualSIM_init, runFunc = (RUNFUNC)PermanentZOneGlynnDualSIM_run, freeFunc = PermanentZOneGlynnDualSIM_free;
    } else if (isGray && !isRows && useGlynn) {
      initFunc = PermanentZOneGlynnGrayDualSIM_init, runFunc = (RUNFUNC)PermanentZOneGlynnGrayDualSIM_run, freeFunc = PermanentZOneGlynnGrayDualSIM_free;
    } else if (!isGray && isRows && useGlynn) {
      initFunc = PermanentZOneGlynnRowsDualSIM_init, runFunc = (RUNFUNC)PermanentZOneGlynnRowsDualSIM_run, freeFunc = PermanentZOneGlynnRowsDualSIM_free;
    } else if (isGray && isRows && useGlynn) {
      initFunc = PermanentZOneGlynnRowsGrayDualSIM_init, runFunc = (RUNFUNC)PermanentZOneGlynnRowsGrayDualSIM_run, freeFunc = PermanentZOneGlynnRowsGrayDualSIM_free;
    }
#else
    if (isGray && isRows && !useGlynn) {
      initFunc = PermanentZOneRowsGrayDualDFE_init, runFunc = (RUNARRAYFUNC)PermanentZOneRowsGrayDualDFE_run_array, freeFunc = PermanentZOneRowsGrayDualDFE_free;
    } else if (isGray && isRows && useGlynn) {
      //initFunc = PermanentZOneGlynnRowsGrayDualDFE_init, runFunc = (RUNFUNC)PermanentZOneGlynnRowsGrayDualDFE_run, freeFunc = PermanentZOneGlynnRowsGrayDualDFE_free;
    }
#endif
  }
  if (!initFunc) return;
  mavMaxFile = initFunc();
#ifdef MAXELER_SIM 
  mavDFE = max_load(mavMaxFile, "local:*");
#else
  if (dual) array = max_load_array(mavMaxFile, 2, "local:*");
  else mavDFE = max_load(mavMaxFile, "local:*");
#endif
  initialized = true;
}

/**
@brief Interface function to releive DFE array
*/
void releive_ZOne_DFE()
{
	if (!initialized) return;
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
@brief Interface function to calculate the Permanent using 0-1 formula on DFE
*/
void calcPermanentZOneDFE(const uint64_t* mtx_data, const uint64_t rows, const uint64_t cols, uint64_t* perm)
{
    if (!initialized) return;
    uint64_t numOfPartialPerms = rows;

#ifdef DEBUG
	printf("Start permanent calulation on DFE\n");
#endif

    //max_file_t* (*initFunc)(void) = NULL;
    //void (*freeFunc)(void) = NULL;
    //RUNFUNC runFunc = NULL;
    union {
#ifdef MAXELER_SIM    
      PermanentZOneSIM_actions_t none;
      PermanentZOneGraySIM_actions_t gray;
      PermanentZOneRowsSIM_actions_t rows;
      PermanentZOneRowsGraySIM_actions_t rowsGray;
      PermanentZOneGlynnSIM_actions_t glynn;
      PermanentZOneGlynnGraySIM_actions_t glynnGray;
      PermanentZOneGlynnRowsSIM_actions_t glynnRows;
      PermanentZOneGlynnRowsGraySIM_actions_t glynnRowsGray;
      PermanentZOneDualSIM_actions_t dual;
      PermanentZOneGrayDualSIM_actions_t dualGray;
      PermanentZOneRowsDualSIM_actions_t dualRows;
      PermanentZOneRowsGrayDualSIM_actions_t dualRowsGray;
      PermanentZOneGlynnDualSIM_actions_t dualGlynn;
      PermanentZOneGlynnGrayDualSIM_actions_t dualGlynnGray;
      PermanentZOneGlynnRowsDualSIM_actions_t dualGlynnRows;
      PermanentZOneGlynnRowsGrayDualSIM_actions_t dualGlynnRowsGray;
#else
      PermanentZOneRowsGrayDFE_actions_t rowsGray;
      //PermanentZOneGlynnRowsGrayDFE_actions_t glynnRowsGray;
      PermanentZOneRowsGrayDualDFE_actions_t dualRowsGray;
      //PermanentZOneGlynnRowsGrayDualDFE_actions_t dualGlynnRowsGray;
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
#ifdef MAXELER_SIM
      if (!isGray && !isRows && !useGlynn) {
        actions.none.param_ticksMax = numOfPartialPerms, actions.none.param_InputMtx = mtx_data, actions.none.outstream_res = perm;
      } else if (isGray && !isRows && !useGlynn) {
        actions.gray.param_ticksMax = numOfPartialPerms, actions.gray.param_InputMtx = mtx_data, actions.gray.outstream_res = perm;
      } else if (!isGray && isRows && !useGlynn) {
        actions.rows.param_ticksMax = numOfPartialPerms, actions.rows.param_InputMtx = mtx_data, actions.rows.outstream_res = perm;
      } else
#endif
      if (isGray && isRows && !useGlynn) {
        actions.rowsGray.param_ticksMax = numOfPartialPerms, actions.rowsGray.param_InputMtx = mtx_data, actions.rowsGray.outstream_res = perm;
      }
#ifdef MAXELER_SIM
      else if (!isGray && !isRows && useGlynn) {
        actions.glynn.param_ticksMax = numOfPartialPerms, actions.glynn.param_InputMtx = mtx_data, actions.glynn.outstream_res = perm;
      } else if (isGray && !isRows && useGlynn) {
        actions.glynnGray.param_ticksMax = numOfPartialPerms, actions.glynnGray.param_InputMtx = mtx_data, actions.glynnGray.outstream_res = perm;
      } else if (!isGray && isRows && useGlynn) {
        actions.glynnRows.param_ticksMax = numOfPartialPerms, actions.glynnRows.param_InputMtx = mtx_data, actions.glynnRows.outstream_res = perm;
      } else
#endif
#ifdef MAXELER_SIM
      if (isGray && isRows && useGlynn) {
        actions.glynnRowsGray.param_ticksMax = numOfPartialPerms, actions.glynnRowsGray.param_InputMtx = mtx_data, actions.glynnRowsGray.outstream_res = perm;
      }
#endif
    } else {
      //Simulation of manager I/Os of purpose OTHER_FPGA not yet supported.
#ifdef MAXELER_SIM
      if (!isGray && !isRows && !useGlynn) {
        actions.dual.param_ticksMax = numOfPartialPerms, actions.dual.param_InputMtx = mtx_data, actions.dual.outstream_res = perm;
      } else if (isGray && !isRows && !useGlynn) {
        actions.dualGray.param_ticksMax = numOfPartialPerms, actions.dualGray.param_InputMtx = mtx_data, actions.dualGray.outstream_res = perm;
      } else if (!isGray && isRows && !useGlynn) {
        actions.dualRows.param_isLocal = 1, actions.dualRows.param_ticksMax = numOfPartialPerms, actions.dualRows.param_InputMtx = mtx_data, actions.dualRows.outstream_res = perm;
      } else if (isGray && isRows && !useGlynn) {
        actions.dualRowsGray.param_isLocal = 1, actions.dualRowsGray.param_ticksMax = numOfPartialPerms, actions.dualRowsGray.param_InputMtx = mtx_data, actions.dualRowsGray.outstream_res = perm;
      } else if (!isGray && !isRows && useGlynn) {
        actions.dualGlynn.param_ticksMax = numOfPartialPerms, actions.dualGlynn.param_InputMtx = mtx_data, actions.dualGlynn.outstream_res = perm;
      } else if (isGray && !isRows && useGlynn) {
        actions.dualGlynnGray.param_ticksMax = numOfPartialPerms, actions.dualGlynnGray.param_InputMtx = mtx_data, actions.dualGlynnGray.outstream_res = perm;
      } else if (!isGray && isRows && useGlynn) {
        actions.dualGlynnRows.param_isLocal = 1, actions.dualGlynnRows.param_ticksMax = numOfPartialPerms, actions.dualGlynnRows.param_InputMtx = mtx_data, actions.dualGlynnRows.outstream_res = perm;
      } else if (isGray && isRows && useGlynn) {
        actions.dualGlynnRowsGray.param_isLocal = 1, actions.dualGlynnRowsGray.param_ticksMax = numOfPartialPerms, actions.dualGlynnRowsGray.param_InputMtx = mtx_data, actions.dualGlynnRowsGray.outstream_res = perm;
      }
#else
      if (isGray && isRows && !useGlynn) {
        actions.dualRowsGray.param_isLocal = 1, actions.dualRowsGray.param_ticksMax = numOfPartialPerms, actions.dualRowsGray.param_InputMtx = mtx_data, actions.dualRowsGray.outstream_res = perm;
        dualactions.dualRowsGray.param_isLocal = 0, dualactions.dualRowsGray.param_ticksMax = numOfPartialPerms, dualactions.dualRowsGray.param_InputMtx = mtx_data, dualactions.dualRowsGray.outstream_res = perm;
      } else if (isGray && isRows && useGlynn) {
        //actions.dualGlynnRowsGray.param_isLocal = 1, actions.dualGlynnRowsGray.param_ticksMax = numOfPartialPerms, actions.dualGlynnRowsGray.param_InputMtx = mtx_data, actions.dualGlynnRowsGray.outstream_res = perm;
        //dualactions.dualGlynnRowsGray.param_isLocal = 1, dualactions.dualGlynnRowsGray.param_ticksMax = numOfPartialPerms, dualactions.dualGlynnRowsGray.param_InputMtx = mtx_data, dualactions.dualGlynnRowsGray.outstream_res = perm;
      }
#endif
    }

    //max_file_t* mavMaxFile = initFunc();
    //max_engine_t* mavDFE = max_load(mavMaxFile, "local:*");
    //PermanentZOneDFE_actions_t actions = { numOfPartialPerms, mtx_data, perm };
    //PermanentZOneDFE( numOfPartialPerms, mtx_data, perm);
#ifdef MAXELER_SIM
    runFunc(mavDFE, &actions);
#else
    if (useDual) runArrayFunc(array, arractions);
    else runFunc(mavDFE, &actions);
#endif
    //max_unload(mavDFE);
    //max_file_free(mavMaxFile);
    //freeFunc();

#ifdef DEBUG
	printf("Permanent calulation on DFE finished\n");
#endif
    /*max_file_t* mavMaxFile = PermanentZOneDualDFE_init();
    array = max_load_array(mavMaxFile, 2, "local:*");
    PermanentZOneDualDFE_actions_t interface_actions[2] = {{ numOfPartialPerms, 1, mtx_data, perm }, { numOfPartialPerms, 0, mtx_data, perm }};
    PermanentZOneDualDFE_actions_t *actions[2] = {&interface_actions[0], &interface_actions[1]};
    PermanentZOneDualDFE_run_array(array, actions);
    max_unload_array(array);
    max_file_free(mavMaxFile);
    PermanentZOneDualDFE_free();*/
     
    return;
}

