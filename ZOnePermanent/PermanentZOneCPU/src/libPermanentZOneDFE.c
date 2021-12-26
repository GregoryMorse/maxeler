#include <stdint.h>
#include <stdlib.h>
#include <MaxSLiCInterface.h>

#include "PermanentZOneDFE.h"
#include "PermanentZOneGrayDFE.h"
#include "PermanentZOneRowsDFE.h"
#include "PermanentZOneRowsGrayDFE.h"
#include "PermanentZOneGlynnDFE.h"
#include "PermanentZOneGlynnGrayDFE.h"
#include "PermanentZOneGlynnRowsDFE.h"
#include "PermanentZOneGlynnRowsGrayDFE.h"
#include "PermanentZOneDualDFE.h"
#include "PermanentZOneGrayDualDFE.h"
#include "PermanentZOneRowsDualDFE.h"
#include "PermanentZOneRowsGrayDualDFE.h"
#include "PermanentZOneGlynnDualDFE.h"
#include "PermanentZOneGlynnGrayDualDFE.h"
#include "PermanentZOneGlynnRowsDualDFE.h"
#include "PermanentZOneGlynnRowsGrayDualDFE.h"

/// static variable to indicate whether DFE is initialized
static bool initialized = false;

/**
@brief Interface function to initialize DFE array
*/
void initialize_ZOne_DFE()
{
	if (initialized) return;
  initialized = true;
}

/**
@brief Interface function to releive DFE array
*/
void releive_ZOne_DFE()
{
	if (~initialized) return;
  initialized = false;
}

typedef void (*RUNFUNC)(max_engine_t*, void*);

/**
@brief Interface function to calculate the Permanent using 0-1 formula on DFE
*/
void calcPermanentZOneDFE(const uint64_t* mtx_data, const uint64_t rows, const uint64_t cols, uint64_t* perm, int isSim, int isGray, int isRows, int useGlynn, int useDual)
{
    uint64_t numOfPartialPerms = rows;

    if (isSim) {
#ifdef DEBUG
	printf("Start permanent calulation on DFE\n");
#endif

    max_file_t* (*initFunc)(void) = NULL;
    void (*freeFunc)(void) = NULL;
    RUNFUNC runFunc = NULL;
    union {
      PermanentZOneDFE_actions_t none;
      PermanentZOneGrayDFE_actions_t gray;
      PermanentZOneRowsDFE_actions_t rows;
      PermanentZOneRowsGrayDFE_actions_t rowsGray;
      PermanentZOneGlynnDFE_actions_t glynn;
      PermanentZOneGlynnGrayDFE_actions_t glynnGray;
      PermanentZOneGlynnRowsDFE_actions_t glynnRows;
      PermanentZOneGlynnRowsGrayDFE_actions_t glynnRowsGray;
      PermanentZOneDualDFE_actions_t dual;
      PermanentZOneGrayDualDFE_actions_t dualGray;
      PermanentZOneRowsDualDFE_actions_t dualRows;
      PermanentZOneRowsGrayDualDFE_actions_t dualRowsGray;
      PermanentZOneGlynnDualDFE_actions_t dualGlynn;
      PermanentZOneGlynnGrayDualDFE_actions_t dualGlynnGray;
      PermanentZOneGlynnRowsDualDFE_actions_t dualGlynnRows;
      PermanentZOneGlynnRowsGrayDualDFE_actions_t dualGlynnRowsGray;
    } actions;
    // simulation
    if (!useDual) {
      if (!isGray && !isRows && !useGlynn) {
        initFunc = PermanentZOneDFE_init, runFunc = (RUNFUNC)PermanentZOneDFE_run, freeFunc = PermanentZOneDFE_free;
        actions.none.param_ticksMax = numOfPartialPerms, actions.none.param_InputMtx = mtx_data, actions.none.outstream_res = perm;
      } else if (isGray && !isRows && !useGlynn) {
        initFunc = PermanentZOneGrayDFE_init, runFunc = (RUNFUNC)PermanentZOneGrayDFE_run, freeFunc = PermanentZOneGrayDFE_free;
        actions.gray.param_ticksMax = numOfPartialPerms, actions.gray.param_InputMtx = mtx_data, actions.gray.outstream_res = perm;
      } else if (!isGray && isRows && !useGlynn) {
        initFunc = PermanentZOneRowsDFE_init, runFunc = (RUNFUNC)PermanentZOneRowsDFE_run, freeFunc = PermanentZOneRowsDFE_free;
        actions.rows.param_ticksMax = numOfPartialPerms, actions.rows.param_InputMtx = mtx_data, actions.rows.outstream_res = perm;
      } else if (isGray && isRows && !useGlynn) {
        initFunc = PermanentZOneRowsGrayDFE_init, runFunc = (RUNFUNC)PermanentZOneRowsGrayDFE_run, freeFunc = PermanentZOneRowsGrayDFE_free;
        actions.rowsGray.param_ticksMax = numOfPartialPerms, actions.rowsGray.param_InputMtx = mtx_data, actions.rowsGray.outstream_res = perm;
      } else if (!isGray && !isRows && useGlynn) {
        initFunc = PermanentZOneGlynnDFE_init, runFunc = (RUNFUNC)PermanentZOneGlynnDFE_run, freeFunc = PermanentZOneGlynnDFE_free;
        actions.glynn.param_ticksMax = numOfPartialPerms, actions.glynn.param_InputMtx = mtx_data, actions.glynn.outstream_res = perm;
      } else if (isGray && !isRows && useGlynn) {
        initFunc = PermanentZOneGlynnGrayDFE_init, runFunc = (RUNFUNC)PermanentZOneGlynnGrayDFE_run, freeFunc = PermanentZOneGlynnGrayDFE_free;
        actions.glynnGray.param_ticksMax = numOfPartialPerms, actions.glynnGray.param_InputMtx = mtx_data, actions.glynnGray.outstream_res = perm;
      } else if (!isGray && isRows && useGlynn) {
        initFunc = PermanentZOneGlynnRowsDFE_init, runFunc = (RUNFUNC)PermanentZOneGlynnRowsDFE_run, freeFunc = PermanentZOneGlynnRowsDFE_free;
        actions.glynnRows.param_ticksMax = numOfPartialPerms, actions.glynnRows.param_InputMtx = mtx_data, actions.glynnRows.outstream_res = perm;
      } else if (isGray && isRows && useGlynn) {
        initFunc = PermanentZOneGlynnRowsGrayDFE_init, runFunc = (RUNFUNC)PermanentZOneGlynnRowsGrayDFE_run, freeFunc = PermanentZOneGlynnRowsGrayDFE_free;
        actions.glynnRowsGray.param_ticksMax = numOfPartialPerms, actions.glynnRowsGray.param_InputMtx = mtx_data, actions.glynnRowsGray.outstream_res = perm;
      }
    } else {
      //Simulation of manager I/Os of purpose OTHER_FPGA not yet supported.
      if (!isGray && !isRows && !useGlynn) {
        initFunc = PermanentZOneDualDFE_init, runFunc = (RUNFUNC)PermanentZOneDualDFE_run, freeFunc = PermanentZOneDualDFE_free;
        actions.dual.param_ticksMax = numOfPartialPerms, actions.dual.param_InputMtx = mtx_data, actions.dual.outstream_res = perm;
      } else if (isGray && !isRows && !useGlynn) {
        initFunc = PermanentZOneGrayDualDFE_init, runFunc = (RUNFUNC)PermanentZOneGrayDualDFE_run, freeFunc = PermanentZOneGrayDualDFE_free;
        actions.dualGray.param_ticksMax = numOfPartialPerms, actions.dualGray.param_InputMtx = mtx_data, actions.dualGray.outstream_res = perm;
      } else if (!isGray && isRows && !useGlynn) {
        initFunc = PermanentZOneRowsDualDFE_init, runFunc = (RUNFUNC)PermanentZOneRowsDualDFE_run, freeFunc = PermanentZOneRowsDualDFE_free;
        actions.dualRows.param_ticksMax = numOfPartialPerms, actions.dualRows.param_InputMtx = mtx_data, actions.dualRows.outstream_res = perm;
      } else if (isGray && isRows && !useGlynn) {
        initFunc = PermanentZOneRowsGrayDualDFE_init, runFunc = (RUNFUNC)PermanentZOneRowsGrayDualDFE_run, freeFunc = PermanentZOneRowsGrayDualDFE_free;
        actions.dualRowsGray.param_ticksMax = numOfPartialPerms, actions.dualRowsGray.param_InputMtx = mtx_data, actions.dualRowsGray.outstream_res = perm;
      } else if (!isGray && !isRows && useGlynn) {
        initFunc = PermanentZOneGlynnDualDFE_init, runFunc = (RUNFUNC)PermanentZOneGlynnDualDFE_run, freeFunc = PermanentZOneGlynnDualDFE_free;
        actions.dualGlynn.param_ticksMax = numOfPartialPerms, actions.dualGlynn.param_InputMtx = mtx_data, actions.dualGlynn.outstream_res = perm;
      } else if (isGray && !isRows && useGlynn) {
        initFunc = PermanentZOneGlynnGrayDualDFE_init, runFunc = (RUNFUNC)PermanentZOneGlynnGrayDualDFE_run, freeFunc = PermanentZOneGlynnGrayDualDFE_free;
        actions.dualGlynnGray.param_ticksMax = numOfPartialPerms, actions.dualGlynnGray.param_InputMtx = mtx_data, actions.dualGlynnGray.outstream_res = perm;
      } else if (!isGray && isRows && useGlynn) {
        initFunc = PermanentZOneGlynnRowsDualDFE_init, runFunc = (RUNFUNC)PermanentZOneGlynnRowsDualDFE_run, freeFunc = PermanentZOneGlynnRowsDualDFE_free;
        actions.dualGlynnRows.param_ticksMax = numOfPartialPerms, actions.dualGlynnRows.param_InputMtx = mtx_data, actions.dualGlynnRows.outstream_res = perm;
      } else if (isGray && isRows && useGlynn) {
        initFunc = PermanentZOneGlynnRowsGrayDualDFE_init, runFunc = (RUNFUNC)PermanentZOneGlynnRowsGrayDualDFE_run, freeFunc = PermanentZOneGlynnRowsGrayDualDFE_free;
        actions.dualGlynnRowsGray.param_ticksMax = numOfPartialPerms, actions.dualGlynnRowsGray.param_InputMtx = mtx_data, actions.dualGlynnRowsGray.outstream_res = perm;
      }
    }

    max_file_t* mavMaxFile = initFunc();
    max_engine_t* mavDFE = max_load(mavMaxFile, "local:*");
    //PermanentZOneDFE_actions_t actions = { numOfPartialPerms, mtx_data, perm };
    //PermanentZOneDFE( numOfPartialPerms, mtx_data, perm);
    runFunc(mavDFE, &actions);
    max_unload(mavDFE);
    max_file_free(mavMaxFile);
    freeFunc();

#ifdef DEBUG
	printf("Permanent calulation on DFE finished\n");
#endif
    } else {
      if (!useDual) {
      } else {
        //max_engarray_t* array = NULL;
        if (!isGray && !isRows && !useGlynn) {
          /*max_file_t* mavMaxFile = PermanentZOneDualDFE_init();
          array = max_load_array(mavMaxFile, 2, "local:*");
          PermanentZOneDualDFE_actions_t interface_actions[2] = {{ numOfPartialPerms, 1, mtx_data, perm }, { numOfPartialPerms, 0, mtx_data, perm }};
          PermanentZOneDualDFE_actions_t *actions[2] = {&interface_actions[0], &interface_actions[1]};
          PermanentZOneDualDFE_run_array(array, actions);
          max_unload_array(array);
          max_file_free(mavMaxFile);
          PermanentZOneDualDFE_free();*/
        }
      }
    }
     
    return;
}

