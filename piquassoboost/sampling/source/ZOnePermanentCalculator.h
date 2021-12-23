#ifndef ZOnePermanentCalculator_H
#define ZOnePermanentCalculator_H

#include "matrix.h"
#include "matrix32.h"
#include "PicState.h"
#include <vector>
#include "PicVector.hpp"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif


namespace pic {

/**
@brief Call to print the elements of a container
@param vec a container
@return Returns with the sum of the elements of the container
*/
template <typename Container>
void print_state( Container state );


/**
@brief Interface class representing a 0-1 permanent calculator
*/
class ZOnePermanentCalculator {

protected:
    std::vector<uint64_t> mtx;

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
ZOnePermanentCalculator();



/**
@brief Call to calculate the permanent via 0-1 formula scaling with n*2^n. (Does not use gray coding, but does the calculation is similar but scalable fashion)
@param mtx The effective scattering matrix of a boson sampling instance
@return Returns with the calculated permanent
*/
std::vector<uint64_t> calculate(std::vector<uint64_t> &mtx, int isGray, int isRows, int useGlynn, int useDual);


}; //ZOnePermanentCalculator





// relieve Python extension from TBB functionalities
#ifndef CPYTHON


/**
@brief Class to calculate a partial permanent via 0-1 formula scaling with n*2^n. (Does not use gray coding, but does the calculation is similar but scalable fashion) 
*/
class ZOnePermanentCalculatorTask {

public:

    /// Unitary describing a quantum circuit
    std::vector<uint64_t> mtx;
    /// thread local storage for partial permanents
    tbb::combinable<std::vector<uint64_t>> priv_addend;

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
ZOnePermanentCalculatorTask();


/**
@brief Call to calculate the permanent via 0-1 formula. scales with n*2^n
@param mtx The effective scattering matrix of a boson sampling instance
@return Returns with the calculated permanent
*/
std::vector<uint64_t> calculate(std::vector<uint64_t> &mtx, int isGray, int isRows, int useGlynn, int useDual);



}; // partial permanent_Task


#endif // CPYTHON




} // PIC

#endif
