#include <stdint.h>
#include <stdlib.h>
#include <MaxSLiCInterface.h>

#include "PermanentZOneDFE.h"
#include "PermanentZOneGrayDFE.h"
#include "PermanentZOneRowsDFE.h"
#include "PermanentZOneRowsGrayDFE.h"

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
void calcPermanentZOneDFE(const uint64_t* mtx_data, const uint64_t rows, const uint64_t cols, uint64_t* perm)
{
    uint64_t numOfPartialPerms = 1 << (rows-2);

#ifdef DEBUG
	printf("Start permanent calulation on DFE\n");
#endif

    // simulation
    max_file_t* mavMaxFile = PermanentZOneDFE_init();
    max_engine_t* mavDFE = max_load(mavMaxFile, "local:*");
    PermanentZOneDFE_actions_t actions = { numOfPartialPerms, mtx_data, perm };    
    //PermanentZOneDFE( numOfPartialPerms, mtx_data, perm);
    PermanentZOneDFE_run(mavDFE, &actions);
    
    max_unload(mavDFE);
    PermanentZOneDFE_free();

#ifdef DEBUG
	printf("Permanent calulation on DFE finished\n");
#endif
     
    return;
}

/**
@brief Interface function to calculate the Permanent using 0-1 formula on DFE
*/
void calcPermanentZOneGrayDFE(const uint64_t* mtx_data, const uint64_t rows, const uint64_t cols, uint64_t* perm)
{
    uint64_t numOfPartialPerms = 1 << (rows-2);

    printf("%llu, %llu\n", numOfPartialPerms, rows);
    
    for (size_t i = 0; i < rows; i++) printf("%llX ", mtx_data[i]); printf("\n");
 
#ifdef DEBUG
	printf("Start permanent calulation on DFE\n");
#endif

    max_file_t* mavMaxFile = PermanentZOneGrayDFE_init();
    max_engine_t* mavDFE = max_load(mavMaxFile, "local:*");
    PermanentZOneGrayDFE_actions_t actions = { numOfPartialPerms, mtx_data, perm };    
    //PermanentZOneGrayDFE( numOfPartialPerms, mtx_data, perm);
    PermanentZOneGrayDFE_run(mavDFE, &actions);
    
    max_unload(mavDFE);
    PermanentZOneGrayDFE_free();

#ifdef DEBUG
	printf("Permanent calulation on DFE finished\n");
#endif
    printf("%llX\n", perm[0]);
     
    return;
}

/**
@brief Interface function to calculate the Permanent using 0-1 formula on DFE
*/
void calcPermanentZOneRowsDFE(const uint64_t* mtx_data, const uint64_t rows, const uint64_t cols, uint64_t* perm)
{
    uint64_t numOfPartialPerms = 1 << rows;

#ifdef DEBUG
	printf("Start permanent calulation on DFE\n");
#endif

    max_file_t* mavMaxFile = PermanentZOneRowsDFE_init();
    max_engine_t* mavDFE = max_load(mavMaxFile, "local:*");
    PermanentZOneRowsDFE_actions_t actions = { numOfPartialPerms, mtx_data, perm };    
    //PermanentZOneRowsDFE( numOfPartialPerms, mtx_data, perm);
    PermanentZOneRowsDFE_run(mavDFE, &actions);
    
    max_unload(mavDFE);
    PermanentZOneRowsDFE_free();

#ifdef DEBUG
	printf("Permanent calulation on DFE finished\n");
#endif
     
    return;
}

/**
@brief Interface function to calculate the Permanent using 0-1 formula on DFE
*/
void calcPermanentZOneRowsGrayDFE(const uint64_t* mtx_data, const uint64_t rows, const uint64_t cols, uint64_t* perm)
{
    uint64_t numOfPartialPerms = 1 << rows;
  
#ifdef DEBUG
	printf("Start permanent calulation on DFE\n");
#endif

    max_file_t* mavMaxFile = PermanentZOneRowsGrayDFE_init();
    max_engine_t* mavDFE = max_load(mavMaxFile, "local:*");
    PermanentZOneRowsGrayDFE_actions_t actions = { numOfPartialPerms, mtx_data, perm };    
    //PermanentZOneRowsGrayDFE( numOfPartialPerms, mtx_data, perm);
    PermanentZOneRowsGrayDFE_run(mavDFE, &actions);
    
    max_unload(mavDFE);
    PermanentZOneRowsGrayDFE_free();


#ifdef DEBUG
	printf("Permanent calulation on DFE finished\n");
#endif
     
    return;
}
