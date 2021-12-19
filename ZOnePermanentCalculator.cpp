#include <iostream>
#include "ZOnePermanentCalculator.h"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include <math.h>


namespace pic {




/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
ZOnePermanentCalculator::ZOnePermanentCalculator() {}


/**
@brief Wrapper method to calculate the permanent via 0-1 formula. scales with n*2^n
@param mtx Unitary describing a quantum circuit
@return Returns with the calculated permanent
*/
std::vector<uint64_t>
ZOnePermanentCalculator::calculate(std::vector<uint64_t> &mtx) {
    if (mtx.size() == 0)
        return std::vector<uint64_t> { 1, 0, 0, 0, 0 };

    ZOnePermanentCalculatorTask calculator;
    return calculator.calculate( mtx );
}



/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
ZOnePermanentCalculatorTask::ZOnePermanentCalculatorTask() {}

void add_uint64_vectors(std::vector<uint64_t>& left, const std::vector<uint64_t>& right)
{
  //high bits of last uint64_t in vectors are sign bits
  //for (size_t x = 0; x < left.size(); x++) printf("%llX ", left[x]);
  //printf("+");
  //for (size_t x = 0; x < right.size(); x++) printf("%llX ", right[x]);
  int carry = 0;
  int rightsigned = right.size() && (right[right.size()-1] & (1UL << 63));
  int leftsigned = left.size() && (left[left.size()-1] & (1UL << 63));
  size_t i, iterat = std::max(right.size(), left.size());
  if (iterat == 0) return;
  for (i = 0; i < iterat-1; i++) {
    uint64_t li = i < left.size() ? left[i] : -leftsigned, ri = i < right.size() ? right[i] : -rightsigned; 
    int nextcarry = carry && ri == -1UL;
    uint64_t tmp = ri + carry;
    uint64_t res = li + tmp;
    nextcarry = nextcarry || (((li & tmp) & (1UL << 63)) || ((li ^ tmp) & (1UL << 63)) && !(res & (1UL << 63)));
    if (i >= left.size()) left.push_back(res);
    else left[i] = res;
    carry = nextcarry;
  }
  uint64_t li = i < left.size() ? left[i] : -leftsigned, ri = i < right.size() ? right[i] : -rightsigned;
  uint64_t tmp = ri + carry;
  uint64_t res = li + tmp;
  if (i >= left.size()) left.push_back(res);
  else left[i] = res;
  if (!leftsigned && !rightsigned && (res & (1UL << 63)) && left[left.size()-1] != 0) left.push_back(0);
  else if (rightsigned && leftsigned && !(res & (1UL << 63)) && left[left.size()-1] != -1UL) left.push_back(-1UL);
  while (left.size()>=2 && (left[left.size()-1] == 0 && !(left[left.size()-2] & (1UL << 63)) ||
                            left[left.size()-1] == -1UL && (left[left.size()-2] & (1UL << 63)))) left.pop_back();
  //printf("=");
  //for (size_t x = 0; x < left.size(); x++) printf("%llX ", left[x]);
  //printf("\n");
}

void test_signed_add()
{
  {
    std::vector<uint64_t> res { 1 };
    add_uint64_vectors(res, std::vector<uint64_t> { 1 });
    assert((res == std::vector<uint64_t> { 1 << 1 }));
  }
  {
    std::vector<uint64_t> res { -1UL };
    add_uint64_vectors(res, std::vector<uint64_t> { -1UL });
    assert((res == std::vector<uint64_t> { -1UL * 2 }));
  }
  {
    std::vector<uint64_t> res { (1UL << 63) - 1 };
    add_uint64_vectors(res, std::vector<uint64_t> { (1UL << 63) - 1 });
    assert((res == std::vector<uint64_t> { ((1UL << 63) - 1) << 1, 0 }));
  }
  {
    std::vector<uint64_t> res { (1UL << 63) };
    add_uint64_vectors(res, std::vector<uint64_t> { (1UL << 63) });
    assert((res == std::vector<uint64_t> { 0, -1UL }));
  }
  {
    std::vector<uint64_t> res { -1UL, 0 };
    add_uint64_vectors(res, std::vector<uint64_t> { -1UL, 0 });
    assert((res == std::vector<uint64_t> { -2UL, 1 }));
  }
  {
    std::vector<uint64_t> res { -1UL, -1UL, 0 };
    add_uint64_vectors(res, std::vector<uint64_t> { -1UL });
    assert((res == std::vector<uint64_t> { -2UL, -1UL, 0 }));
  }
  {
    std::vector<uint64_t> res { -1UL };
    add_uint64_vectors(res, std::vector<uint64_t> { -1UL, -1UL, 0 });
    assert((res == std::vector<uint64_t> { -2UL, -1UL, 0 }));
  }
  {
    std::vector<uint64_t> res { -1UL, -1UL, 0 };
    add_uint64_vectors(res, std::vector<uint64_t> { 1 });
    assert((res == std::vector<uint64_t> { 0, 0, 1 }));
  }
  {
    std::vector<uint64_t> res { 1 };
    add_uint64_vectors(res, std::vector<uint64_t> { -1UL, -1UL, 0 });
    assert((res == std::vector<uint64_t> { 0, 0, 1 }));
  }
  {
    std::vector<uint64_t> res { -1UL, (1UL << 63) - 1 };
    add_uint64_vectors(res, std::vector<uint64_t> { 1 });
    assert((res == std::vector<uint64_t> { 0, (1UL << 63), 0 }));
  }
  {
    std::vector<uint64_t> res { 1 };
    add_uint64_vectors(res, std::vector<uint64_t> { -1UL, (1UL << 63) - 1 });
    assert((res == std::vector<uint64_t> { 0, (1UL << 63), 0 }));
  }
  {
    std::vector<uint64_t> res { 1UL << 63, 0 };
    add_uint64_vectors(res, std::vector<uint64_t> { -1UL, -1UL });
    assert((res == std::vector<uint64_t> { (1UL << 63) - 1 }));
  }
  {
    std::vector<uint64_t> res { -1UL, -1UL };
    add_uint64_vectors(res, std::vector<uint64_t> { 1UL << 63, 0 });
    assert((res == std::vector<uint64_t> { (1UL << 63) - 1 }));
  }
   
}

void mul_uint64_vector(std::vector<uint64_t>& left, const uint8_t right)
{
  uint8_t carry = 0;
  for (size_t i = 0; i < left.size(); i++) {
    uint64_t first40 = (left[i] & ((1UL << 32)-1)) * right + carry; //maximum 41 bits
    uint64_t next40 = (left[i] >> 32) * right; //0xFF*0xFFFFFFFF==0xFE FFFFFF01 so 8-bit carry is safe
    left[i] = first40 + (next40 << 32);
    carry = (next40 >> 32) + ((next40 & (1UL << 31)) && !(left[i] & (1UL << 63)));
  }
  if (carry) left.push_back(carry);
}

void neg_uint64_vector(std::vector<uint64_t>& left)
{
  if (left.size() == 0) return;
  for (size_t i = 0; i < left.size(); i++) left[i] = ~left[i]; 
  add_uint64_vectors(left, std::vector<uint64_t> { 1 });
}

void unsigned_to_signed_vector(std::vector<uint64_t>& left)
{
  if (left.size() != 0 && (left[left.size()-1] & (1UL << 63))) left.push_back(0);
}


//popcount64c https://en.wikipedia.org/wiki/Hamming_weight
static long numberOfSetBits(uint64_t i)
{
    i = i - ((i >> 1) & 0x5555555555555555);
    i = (i & 0x3333333333333333) + ((i >> 2) & 0x3333333333333333);
    return (((i + (i >> 4)) & 0xF0F0F0F0F0F0F0F) * 0x101010101010101) >> 56;
}

/**
@brief Call to calculate the permanent via 0-1 formula. scales with n*2^n
@param mtx Unitary describing a quantum circuit
@return Returns with the calculated permanent
*/
std::vector<uint64_t>
ZOnePermanentCalculatorTask::calculate(std::vector<uint64_t> &mtx) {
    //test_signed_add();
    uint64_t* mtx_data = mtx.data();
    size_t rows = mtx.size();
    if (rows == 0) return std::vector<uint64_t> { 1 };
    
    priv_addend = tbb::combinable<std::vector<uint64_t>> {[](){return std::vector<uint64_t>();}};
    
    tbb::parallel_for(tbb::blocked_range<size_t>(0, 1 << rows), [&](tbb::blocked_range<size_t> r) {
      std::vector<uint64_t> &permanent_priv = priv_addend.local();
      for (size_t set_idx=r.begin(); set_idx<r.end(); ++set_idx){
        std::vector<uint64_t> prod { 1 };
        std::vector<uint64_t> masks; masks.reserve(rows);
        for (size_t col_idx=0; col_idx<rows; ++col_idx) {
          uint64_t mask = set_idx & (1 << col_idx);
          if (mask) masks.push_back(mask);
        }
        for (std::vector<uint64_t>::iterator row_idx=mtx.begin(); row_idx!=mtx.end(); row_idx++) {
          uint8_t val = 0; uint64_t row = *row_idx;
          for (std::vector<uint64_t>::iterator col_idx=masks.begin(); col_idx!=masks.end(); col_idx++) {
            if (*col_idx & row) val++;
          }
          mul_uint64_vector(prod, val);
        }
        unsigned_to_signed_vector(prod);
        if (masks.size() & 1) neg_uint64_vector(prod); //numberOfSetBits(set_idx)
        //printf("%llu %llu\n", set_idx, prod.size() ? prod[0] : 0);
        //printf("%llu %llu\n", permanent_priv.size() ? permanent_priv[0] : 0, prod.size() ? prod[0] : 0);
        add_uint64_vectors(permanent_priv, prod);
        //printf("%llu\n", permanent_priv.size() ? permanent_priv[0] : 0);
        
      }
    });
    
    // sum up partial permanents
    std::vector<uint64_t> permanent;

    priv_addend.combine_each([&](std::vector<uint64_t> &a) {
        add_uint64_vectors(permanent, a);
    });
    if (rows & 1) neg_uint64_vector(permanent);

    return permanent;


}


} // PIC
