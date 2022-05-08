#include <iostream>
#include "GlynnPermanentCalculatorRepeatedInf.h"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include <math.h>

namespace pic {




GlynnPermanentCalculatorRepeatedInf::GlynnPermanentCalculatorRepeatedInf() {}


Complex16 GlynnPermanentCalculatorRepeatedInf::calculate(
    matrix &mtx,
    PicState_int64& input_state,
    PicState_int64& output_state
) {

    // column multiplicities are determined by the input state
    PicState_int col_multiplicities =
        convert_PicState_int64_to_PicState_int(input_state);
    PicState_int row_multiplicities =
    // row multiplicities are determined by the output state
        convert_PicState_int64_to_PicState_int(output_state);

    GlynnPermanentCalculatorRepeatedInfTask calculator(mtx, row_multiplicities, col_multiplicities);
    return calculator.calculate( );
    
}

#define REALPART(c) reinterpret_cast<FloatInf(&)[2]>(c)[0]
#define IMAGPART(c) reinterpret_cast<FloatInf(&)[2]>(c)[1]

GlynnPermanentCalculatorRepeatedInfTask::GlynnPermanentCalculatorRepeatedInfTask(
    matrix &mtx,
    PicState_int& row_multiplicities,
    PicState_int& col_multiplicities
)
    : mtx(mtx)
    , row_multiplicities(row_multiplicities)
    , col_multiplicities(col_multiplicities)
{
    Complex16* mtx_data = mtx.get_data();
    
    // calculate and store 2*mtx being used later in the recursive calls
    mtx2 = matrix32( mtx.rows, mtx.cols);
    Complex32* mtx2_data = mtx2.get_data();

    tbb::parallel_for( tbb::blocked_range<size_t>(0, mtx.rows), [&](tbb::blocked_range<size_t> r) {
        for (size_t row_idx=r.begin(); row_idx<r.end(); ++row_idx){

            size_t row_offset   = row_idx*mtx.stride;
            size_t row_offset_2 = row_idx*mtx2.stride;
            for (size_t col_idx=0; col_idx<mtx.cols; ++col_idx) {
                mtx2_data[row_offset_2+col_idx] = 2*mtx_data[ row_offset + col_idx ];
            }

        }
    });

    // minimal index set to higher than maximal in case of empty row_multiplicities
    minimalIndex = row_multiplicities.size();

    // deltaLimits stores the index range of the deltas:
    // first not zero has to be one smaller than the multiplicity
    // others have to be the multiplicity
    deltaLimits = PicState_int(row_multiplicities.size());
    for (size_t i = 0; i < row_multiplicities.size(); i++){
        if (row_multiplicities[i]>0){
            if (minimalIndex > i){
                deltaLimits[i] = row_multiplicities[i]-1;
                minimalIndex = i;
            }else{
                deltaLimits[i] = row_multiplicities[i];
            }
        }else{
            deltaLimits[i] = 0;
        }
    }
}


Complex16
GlynnPermanentCalculatorRepeatedInfTask::calculate() {

    // if all the elements in row multiplicities are zero, returning default value
    if (minimalIndex == row_multiplicities.size()){
        return Complex16(1.0, 0.0);
    }

    Complex16* mtx_data = mtx.get_data();

    // calculate the initial sum of the columns
    ComplexInf* colSum_data = new ComplexInf[mtx.cols];
    memset( colSum_data, 0.0, colSum.size()*sizeof(Complex32));

    tbb::parallel_for( tbb::blocked_range<size_t>(0, mtx.cols), [&](tbb::blocked_range<size_t> r) {
        for (size_t col_idx=r.begin(); col_idx<r.end(); ++col_idx){

            size_t row_offset = 0;
            for (size_t row_idx=0; row_idx<mtx.rows; ++row_idx) {
                REALPART(colSum_data[col_idx]) += row_multiplicities[row_idx] * mtx_data[ row_offset + col_idx ].real();
                IMAGPART(colSum_data[col_idx]) += row_multiplicities[row_idx] * mtx_data[ row_offset + col_idx ].imag();
                row_offset += mtx.stride;
            }

        }
    });

    // thread local storage for partial permanent
    priv_addend = tbb::combinable<ComplexInf> {[](){return ComplexInf();}};


    // start the iterations over vectors of deltas
    // values :
    //   colSum              : vector of sums of the columns
    //   sign                : multiplication of all deltas: 1 by default (all deltas are +1)
    //   index_min           : first index greater than 0
    //   currentMultiplicity : multiplicity of the current delta vector
    IterateOverDeltas( colSum_data, 1, minimalIndex, 1 );


    // sum up partial permanents
    Complex32 permanent( 0.0, 0.0 );

    priv_addend.combine_each([&](ComplexM<long double> &a) {
        permanent = permanent + a.get();
    });

    size_t sumMultiplicities = 0;
    for (size_t idx = 0; idx < row_multiplicities.size(); idx++){
        sumMultiplicities += row_multiplicities[idx];
    }

    permanent = permanent / (long double)power_of_2( (unsigned long long) (sumMultiplicities-1) );

    return Complex16(permanent.real(), permanent.imag());
}


void 
GlynnPermanentCalculatorRepeatedInfTask::IterateOverDeltas(
    pic::ComplexInf* colSum_data,
    int sign,
    size_t index_min,
    int currentMultiplicity
) {

    // Calculate the partial permanent
    pic::ComplexInf colSumProd(colSum_data[0]);
    for (unsigned int idx=1; idx<mtx2.cols; idx++) {
        for (int jdx = 0; jdx < col_multiplicities[idx]; jdx++){
            //colSumProd *= colSum_data[idx];
            FloatInf ac(std::move(REALPART(colSumProd) * REALPART(colSum_data[idx])));
            FloatInf bd(std::move(IMAGPART(colSumProd) * IMAGPART(colSum_data[idx])));
            FloatInf p(std::move(REALPART(colSumProd) + IMAGPART(colSumProd)));
            p *= REALPART(colSum_data[idx]) + IMAGPART(colSum_data[idx]); p -= ac + bd;
            ac -= bd;
            REALPART(colSumProd) = ac;
            IMAGPART(colSumProd) = p;            
        }
    }

    // add partial permanent to the local value
    // multiplicity is given by the binomial formula of multiplicity over sign
    pic::ComplexInf &permanent_priv = priv_addend.local();
    REALPART(colSumProd) *= currentMultiplicity;
    IMAGPART(colSumProd) *= currentMultiplicity;
    if (sign > 0) {
        REALPART(permanent_priv) += REALPART(colSumProd);
        IMAGPART(permanent_priv) += IMAGPART(colSumProd);
    } else {
        REALPART(permanent_priv) -= REALPART(colSumProd);
        IMAGPART(permanent_priv) -= IMAGPART(colSumProd);
    }    

    tbb::parallel_for( tbb::blocked_range<int>(index_min, mtx.rows), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx){
            int localSign = sign;
            // create an altered vector from the current delta
            pic::ComplexInf* colSum_new_data = new pic::ComplexInf[mtx2.cols];

            Complex32* mtx2_data = mtx2.get_data();

            size_t row_offset = idx*mtx2.stride;
            
            // deltaLimits is the same as row_multiplicity except the first non-zero element
            for (int indexOfMultiplicity = 1; indexOfMultiplicity <= deltaLimits[idx]; indexOfMultiplicity++){
                for (unsigned int jdx=0; jdx<mtx2.cols; jdx++) {
                    REALPART(colSum_new_data[jdx]) = REALPART(colSum_data[jdx]);
                    REALPART(colSum_new_data[jdx]) -= mtx2_data[row_offset+jdx].real();
                    IMAGPART(colSum_new_data[jdx]) = IMAGPART(colSum_data[jdx]);
                    IMAGPART(colSum_new_data[jdx]) -= mtx2_data[row_offset+jdx].imag();
                }

                localSign *= -1;
                IterateOverDeltas(
                    colSum_new, 
                    localSign,
                    idx+1,
                    currentMultiplicity*binomialCoeff(deltaLimits[idx], indexOfMultiplicity)
                );
            }
            delete [] colSum_new_data;
        }
    });

    return;
}









} // PIC
