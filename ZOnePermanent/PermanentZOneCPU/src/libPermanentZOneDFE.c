#include <stdint.h>
#include <stdlib.h>
#include <MaxSLiCInterface.h>

#include "PermanentZOneDFE.h"
#include "PermanentZOneGrayDFE.h"
#include "PermanentZOneRowsDFE.h"
#include "PermanentZOneRowsGrayDFE.h"
#include "PermanentZOneDualDFE.h"

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

/**
@brief Interface function to calculate the Permanent using 0-1 formula on DFE
*/
void calcPermanentZOneDFE(const uint64_t* mtx_data, const uint64_t rows, const uint64_t cols, uint64_t* perm, int isSim, int isGray, int isRows, int useGlynn, int useDual)
{
    uint64_t numOfPartialPerms = 1 << (isRows ? rows : (useDual ? (rows-3) : (rows-2))); //numkernels==4 or 8

    if (isSim) {
#ifdef DEBUG
	printf("Start permanent calulation on DFE\n");
#endif

    // simulation
    max_engine_t* mavDFE = NULL;
    if (!useDual) {
      if (!isGray && !isRows && !useGlynn) {
        max_file_t* mavMaxFile = PermanentZOneDFE_init();
        mavDFE = max_load(mavMaxFile, "local:*");
        PermanentZOneDFE_actions_t actions = { numOfPartialPerms, mtx_data, perm };
        //PermanentZOneDFE( numOfPartialPerms, mtx_data, perm);
        PermanentZOneDFE_run(mavDFE, &actions);
        max_unload(mavDFE);
        max_file_free(mavMaxFile);
        PermanentZOneDFE_free();
      } else if (isGray && !isRows && !useGlynn) {
        max_file_t* mavMaxFile = PermanentZOneGrayDFE_init();
        mavDFE = max_load(mavMaxFile, "local:*");
        PermanentZOneGrayDFE_actions_t actions = { numOfPartialPerms, mtx_data, perm };
        //PermanentZOneGrayDFE( numOfPartialPerms, mtx_data, perm);
        PermanentZOneGrayDFE_run(mavDFE, &actions);
        max_unload(mavDFE);
        max_file_free(mavMaxFile);
        PermanentZOneGrayDFE_free();
      } else if (!isGray && isRows && !useGlynn) {
        max_file_t* mavMaxFile = PermanentZOneRowsDFE_init();
        mavDFE = max_load(mavMaxFile, "local:*");
        PermanentZOneRowsDFE_actions_t actions = { numOfPartialPerms, mtx_data, perm };
        //PermanentZOneRowsDFE( numOfPartialPerms, mtx_data, perm);
        PermanentZOneRowsDFE_run(mavDFE, &actions);
        max_unload(mavDFE);
        max_file_free(mavMaxFile);
        PermanentZOneRowsDFE_free();
      } else if (isGray && isRows && !useGlynn) {
        max_file_t* mavMaxFile = PermanentZOneRowsGrayDFE_init();
        mavDFE = max_load(mavMaxFile, "local:*");
        PermanentZOneRowsGrayDFE_actions_t actions = { numOfPartialPerms, mtx_data, perm };
        //PermanentZOneRowsGrayDFE( numOfPartialPerms, mtx_data, perm);
        PermanentZOneRowsGrayDFE_run(mavDFE, &actions);
        max_unload(mavDFE);
        max_file_free(mavMaxFile);
        PermanentZOneRowsGrayDFE_free();
      } else if (!isGray && isRows && useGlynn) {}
      else if (isGray && isRows && useGlynn) {}
    } else {
      //Simulation of manager I/Os of purpose OTHER_FPGA not yet supported.
      if (!isGray && !isRows && !useGlynn) {
        max_file_t* mavMaxFile = PermanentZOneDualDFE_init();
        mavDFE = max_load(mavMaxFile, "local:*");
        PermanentZOneDualDFE_actions_t actions = { numOfPartialPerms, 1, mtx_data, perm };
        //PermanentZOneDualDFE( numOfPartialPerms, mtx_data, perm);
        PermanentZOneDualDFE_run(mavDFE, &actions);
        max_unload(mavDFE);
        max_file_free(mavMaxFile);
        PermanentZOneDualDFE_free();
      }
    }
    

#ifdef DEBUG
	printf("Permanent calulation on DFE finished\n");
#endif
    } else {
      if (!useDual) {
      } else {
        max_engarray_t* array = NULL;
        if (!isGray && !isRows && !useGlynn) {
          max_file_t* mavMaxFile = PermanentZOneDualDFE_init();
          array = max_load_array(mavMaxFile, 2, "local:*");
          PermanentZOneDualDFE_actions_t interface_actions[2] = {{ numOfPartialPerms, 1, mtx_data, perm }, { numOfPartialPerms, 0, mtx_data, perm }};
          PermanentZOneDualDFE_actions_t *actions[2] = {&interface_actions[0], &interface_actions[1]};
          PermanentZOneDualDFE_run_array(array, actions);
          max_unload_array(array);
          max_file_free(mavMaxFile);
          PermanentZOneDualDFE_free();
        }
      }
    }
     
    return;
}

