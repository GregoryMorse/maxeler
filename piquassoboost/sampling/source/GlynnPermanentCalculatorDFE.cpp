#include "GlynnPermanentCalculatorDFE.h"


#ifndef CPYTHON
#include <tbb/tbb.h>
#endif

CALCPERMGLYNNDFE calcPermanentGlynnDFE = NULL;
INITPERMGLYNNDFE initialize_DFE = NULL;
FREEPERMGLYNNDFE releive_DFE = NULL;

namespace pic {

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void
GlynnPermanentCalculator_DFE(matrix& matrix_mtx, Complex16& perm, int useDual)
{
    if (matrix_mtx.rows < 4 || matrix_mtx.rows < 5 && use_dual) {} //compute with other method
    // calulate the maximal sum of the columns to normalize the matrix
    matrix colSumMax( matrix_mtx.cols, 1);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex16) );
    for (int idx=0; idx<matrix_mtx.rows; idx++) {
        for( int jdx=0; jdx<matrix_mtx.cols; jdx++) {
            Complex16 value1 = colSumMax[jdx] + matrix_mtx[ idx*matrix_mtx.stride + jdx];
            Complex16 value2 = colSumMax[jdx] - matrix_mtx[ idx*matrix_mtx.stride + jdx];
            if ( std::abs( value1 ) < std::abs( value2 ) ) {
                colSumMax[jdx] = value2;
            }
            else {
                colSumMax[jdx] = value1;
            }

        }

    }




    // calculate the renormalization coefficients
    matrix_base<double> renormalize_data(matrix_mtx.cols, 1);
    for (int jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
        renormalize_data[jdx] = std::abs(colSumMax[jdx]);
    }

    // renormalize the input matrix
    for (int idx=0; idx<matrix_mtx.rows; idx++) {
        for( int jdx=0; jdx<matrix_mtx.cols; jdx++) {
            matrix_mtx[ idx*matrix_mtx.stride + jdx] = matrix_mtx[ idx*matrix_mtx.stride + jdx]/renormalize_data[jdx];
        }

    }    


    // SLR and DFE input matrix with 1.0 filling on top row, 0 elsewhere 
    size_t max_dim = useDual ? MAX_FPGA_DIM : MAX_SINGLE_FPGA_DIM;
    size_t max_fpga_rows =  max_dim;
    size_t max_fpga_cols =  max_dim;

    size_t rows = matrix_mtx.rows;

    matrix mtx = matrix(max_fpga_rows, max_fpga_cols);
    Complex16* mtx_data = mtx.get_data();

    Complex16 padding_element(1.0,0.0);
    for (size_t idx=0; idx<rows; idx++) {
        size_t offset = idx*matrix_mtx.stride;
        size_t offset_small = idx*mtx.stride;
        for (size_t jdx=0; jdx<rows; jdx++) {
            mtx_data[offset_small+jdx] = matrix_mtx[offset+jdx];
        }

        for (size_t jdx=rows; jdx<max_fpga_cols; jdx++) {
            mtx_data[offset_small+jdx] = padding_element;
        }
        padding_element.real(0.0);
    }

    memset( mtx_data + rows*mtx.stride, 0.0, (max_fpga_rows-rows)*max_fpga_cols*sizeof(Complex16));
   
    calcPermanentGlynnDFE( mtx_data, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, &perm);


    return;
}

}
