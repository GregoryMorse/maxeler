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
ZOnePermanentCalculator::calculate(std::vector<uint64_t> &mtx, int isGray, int isRows, int useGlynn, int useDual) {
    if (mtx.size() == 0)
        return std::vector<uint64_t> { 1, 0, 0, 0, 0 };

    ZOnePermanentCalculatorTask calculator;
    return calculator.calculate( mtx, isGray, isRows, useGlynn, useDual );
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
  if (iterat == 0) return; else iterat--;
  for (i = 0; i < iterat; i++) {
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

void shr_vector(std::vector<uint64_t>& left, unsigned int bits)
{
  if (bits == 0 or left.size() == 0) return;
  uint64_t carry = 0;
  for (size_t i = left.size()-1; i != 0; i--) {
    uint64_t nextcarry = left[i] << (64 - bits);
    left[i] = (left[i] >> bits) | carry;
    carry = nextcarry;
  }
  left[0] = (left[0] >> bits) | carry;
  while (left.size()!=0 && left[left.size()-1] == 0) left.pop_back();
}


//popcount64c https://en.wikipedia.org/wiki/Hamming_weight
static long numberOfSetBits(uint64_t i)
{
    i = i - ((i >> 1) & 0x5555555555555555);
    i = (i & 0x3333333333333333) + ((i >> 2) & 0x3333333333333333);
    return (((i + (i >> 4)) & 0xF0F0F0F0F0F0F0F) * 0x101010101010101) >> 56;
}

//https://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightParallel
static unsigned int trailingZeros(uint64_t i)
{
  unsigned int v;      // 32-bit word input to count zero bits on right
  unsigned int c = 64; // c will be the number of zero bits on the right
  v &= ~v+1;
  if (v) c--;
  if (v & 0x0000FFFF) c -= 16;
  if (v & 0x00FF00FF) c -= 8;
  if (v & 0x0F0F0F0F) c -= 4;
  if (v & 0x33333333) c -= 2;
  if (v & 0x55555555) c -= 1;
  return c;
}

/**
@brief Call to calculate the permanent via 0-1 formula. scales with n*2^n
@param mtx Unitary describing a quantum circuit
@return Returns with the calculated permanent
*/
std::vector<uint64_t>
ZOnePermanentCalculatorTask::calculate(std::vector<uint64_t> &mtx, int isGray, int isRows, int useGlynn, int useDual) {
    //test_signed_add();
    uint64_t* mtx_data = mtx.data();
    size_t rows = mtx.size();
    if (rows == 0) return std::vector<uint64_t> { 1 };
    priv_addend = tbb::combinable<std::vector<uint64_t>> {[](){return std::vector<uint64_t>();}};
    if (useGlynn) {
      if (isGray) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, 1 << (rows-1)), [&](tbb::blocked_range<size_t> r) {
          std::vector<uint64_t> &permanent_priv = priv_addend.local();
          size_t set_idx=r.begin();
          int sign = 0;
          uint64_t gcode = set_idx ^ (set_idx >> 1); //nth Gray code
          std::vector<uint64_t> prod { 1 };
          std::vector<int8_t> rowsums; rowsums.reserve(rows);
          for (std::vector<uint64_t>::iterator row_idx=mtx.begin(); row_idx!=mtx.end(); row_idx++) {
            uint64_t row = *row_idx; int8_t val = ((1<<(rows-1)) & row) != 0;
            for (uint64_t col_idx=(1<<(rows-1))>>1; col_idx!=0; col_idx>>=1) {
              if (col_idx & row) {
                if (gcode & col_idx) val--;
                else val++; 
              }
            }
            rowsums.push_back(val);
            if (val < 0) { val = -val; sign = !sign; }
            mul_uint64_vector(prod, val);
          }
          unsigned_to_signed_vector(prod);
          if ((set_idx & 1) ^ sign) neg_uint64_vector(prod); //numberOfSetBits()
          add_uint64_vectors(permanent_priv, prod);
          for (set_idx++; set_idx<r.end(); ++set_idx) {
            int sign = 0;
            uint64_t gcode = set_idx ^ (set_idx >> 1); //nth Gray code
            std::vector<uint64_t> prod { 1 };
            uint64_t mask = set_idx & (~set_idx + 1);
            std::vector<int8_t>::iterator rowsumidx = rowsums.begin();
            if (mask & gcode) {
              for (std::vector<uint64_t>::iterator row_idx=mtx.begin(); row_idx!=mtx.end(); row_idx++,rowsumidx++) {
                int8_t val = *rowsumidx-=(((mask & *row_idx) != 0)<<1);
                if (val < 0) { val = -val; sign = !sign; }
                mul_uint64_vector(prod, val);
              }
            } else {
              for (std::vector<uint64_t>::iterator row_idx=mtx.begin(); row_idx!=mtx.end(); row_idx++,rowsumidx++) {
                int8_t val = *rowsumidx+=(((mask & *row_idx) != 0)<<1);
                if (val < 0) { val = -val; sign = !sign; }
                mul_uint64_vector(prod, val);
              }
            }
            unsigned_to_signed_vector(prod);
            if ((set_idx & 1) ^ sign) neg_uint64_vector(prod); //numberOfSetBits(gcode)&1
            add_uint64_vectors(permanent_priv, prod);
          }          
        });
      } else {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, 1 << (rows-1)), [&](tbb::blocked_range<size_t> r) {
          std::vector<uint64_t> &permanent_priv = priv_addend.local();
          for (size_t set_idx=r.begin(); set_idx<r.end(); ++set_idx) {
            int sign = 0;
            uint64_t gcode = set_idx ^ (set_idx >> 1);
            std::vector<uint64_t> prod { 1 };
            for (std::vector<uint64_t>::iterator row_idx=mtx.begin(); row_idx!=mtx.end(); row_idx++) {
              uint64_t row = *row_idx; int8_t val = ((1<<(rows-1)) & row) != 0; 
              for (uint64_t col_idx=(1<<(rows-1))>>1; col_idx!=0; col_idx>>=1) {
                if (col_idx & row) {
                  if (gcode & col_idx) val--;
                  else val++;
                }
              }
              if (val < 0) { val = -val; sign = !sign; }
              mul_uint64_vector(prod, val);
            }            
            unsigned_to_signed_vector(prod);
            if ((set_idx & 1) ^ sign) neg_uint64_vector(prod); //numberOfSetBits(gcode)&1
            add_uint64_vectors(permanent_priv, prod);            
          }
        });
      }
    } else if (isGray) {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, 1 << rows), [&](tbb::blocked_range<size_t> r) {
        std::vector<uint64_t> &permanent_priv = priv_addend.local();
        size_t set_idx=r.begin();
        uint64_t gcode = set_idx ^ (set_idx >> 1); //nth Gray code
        std::vector<uint64_t> prod { 1 };
        std::vector<uint8_t> rowsums; rowsums.reserve(rows);
        std::vector<uint64_t> masks; masks.reserve(rows);
        for (size_t col_idx=0; col_idx<rows; ++col_idx) {
          uint64_t mask = gcode & (1 << col_idx);
          if (mask) masks.push_back(mask);
        }
        for (std::vector<uint64_t>::iterator row_idx=mtx.begin(); row_idx!=mtx.end(); row_idx++) {
          uint8_t val = 0; uint64_t row = *row_idx;
          for (std::vector<uint64_t>::iterator col_idx=masks.begin(); col_idx!=masks.end(); col_idx++) {
            if (*col_idx & row) val++;
          }
          rowsums.push_back(val);
          mul_uint64_vector(prod, val);
        }
        unsigned_to_signed_vector(prod);
        if (set_idx & 1) neg_uint64_vector(prod); //numberOfSetBits(gcode)&1
        add_uint64_vectors(permanent_priv, prod);
        for (set_idx++; set_idx<r.end(); ++set_idx) {
          uint64_t gcode = set_idx ^ (set_idx >> 1); //nth Gray code
          std::vector<uint64_t> prod { 1 };
          uint64_t mask = set_idx & (~set_idx + 1);
          std::vector<uint8_t>::iterator rowsumidx = rowsums.begin();
          if (mask & gcode) {
            for (std::vector<uint64_t>::iterator row_idx=mtx.begin(); row_idx!=mtx.end(); row_idx++,rowsumidx++) {
              mul_uint64_vector(prod, *rowsumidx+=((mask & *row_idx) != 0));
            }
          } else {
            for (std::vector<uint64_t>::iterator row_idx=mtx.begin(); row_idx!=mtx.end(); row_idx++,rowsumidx++) {
              mul_uint64_vector(prod, *rowsumidx-=((mask & *row_idx) != 0));
            }
          }
          unsigned_to_signed_vector(prod);
          if (set_idx & 1) neg_uint64_vector(prod); //numberOfSetBits(gcode)&1
          add_uint64_vectors(permanent_priv, prod);
        }
      });
    } else {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, 1 << rows), [&](tbb::blocked_range<size_t> r) {
        std::vector<uint64_t> &permanent_priv = priv_addend.local();
        for (size_t set_idx=r.begin(); set_idx<r.end(); ++set_idx) {
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
    }
    
    // sum up partial permanents
    std::vector<uint64_t> permanent;

    priv_addend.combine_each([&](std::vector<uint64_t> &a) {
        add_uint64_vectors(permanent, a);
    });    
    if (useGlynn) shr_vector(permanent, rows-1);
    else if (rows & 1) neg_uint64_vector(permanent);

    return permanent;


}


} // PIC
