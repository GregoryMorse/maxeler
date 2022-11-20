#include <stdint.h>
#include <stdlib.h>
#include <MaxSLiCInterface.h>
#include <string.h>
#include <math.h>
#include <gmp.h>

#include "PermanentTest_singleSIM.h"

#define SIZE 8

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

void printnum(__uint128_t* chunks, size_t bytes)
{
    for (int chunk = bytes/16-1; chunk >= 0; chunk--)
        printf("%016lX %016lX ", (__uint64_t)(chunks[chunk] >> 64), (__uint64_t)chunks[chunk]);
    printf("\n");
}

void gen_randdata(void* p, int bits)
{
    const int RAND_BITS = 32; //15
    for (int i = 0; i < bits; i += RAND_BITS)
    {
        //RAND_MAX guaranteed at least 32767, so 15 bits of random data, but here we get a full 32
        //if (bits - i < RAND_BITS) {
        //    *((short*)(((char*)p)+i/16)) |= (short)(rand() << (i%16));            
        //} else {
        //    *((int*)(((char*)p)+i/16)) |= rand() << (i%16);
        //}
        *((int*)(((char*)p)+i/32)) |= rand() & ((1<<(bits - i))-1);
    }
}

int main(void)
{
    max_file_t* mavMaxFile = PermanentTest_singleSIM_init();
    max_engine_t* mavDFE = max_load(mavMaxFile, "local:*");
    int xbits = PermanentTest_singleSIM_INPXBITS, ybits = PermanentTest_singleSIM_INPYBITS;
#if PermanentTest_singleSIM_USEFLOAT != 1
    int signedX = xbits < 0, signedY = ybits < 0;
#endif
    if (xbits < 0) xbits = -xbits;
    if (ybits < 0) ybits = -ybits;
    int outpbits = xbits + ybits;
    size_t xBytes = (xbits+128-1)/128*16,
        yBytes = (ybits+128-1)/128*16,
        outBytes = (outpbits+128-1)/128*16;
    __uint128_t* inp1 = malloc(xBytes);
    __uint128_t* inp2 = malloc(yBytes);
    __uint128_t* outp = malloc(outBytes);
    PermanentTest_singleSIM_actions_t actions = { 1, inp1, xBytes, inp2, yBytes, outp, outBytes };
    
#if PermanentTest_singleSIM_USEFLOAT == 1
#if PermanentTest_singleSIM_INPXBITS == -32
#define XTYPE float
#elif PermanentTest_singleSIM_INPXBITS == -64
#define XTYPE double
#elif PermanentTest_singleSIM_INPXBITS == -79
#define XTYPE long double
#endif
#if PermanentTest_singleSIM_INPYBITS == -32
#define YTYPE float
#elif PermanentTest_singleSIM_INPYBITS == -64
#define YTYPE double
#elif PermanentTest_singleSIM_INPYBITS == -79
#define YTYPE long double
#endif
    for (int i = 0; i < 512; i++) {
        XTYPE a = 0.0; YTYPE b = 0.0;
        gen_randdata(&a, sizeof(XTYPE) * 8);
        gen_randdata(&b, sizeof(YTYPE) * 8);
#if PermanentTest_singleSIM_ISCOMPLEX == 1
#else
#if PermanentTest_singleSIM_ADDSUBMUL == 2
        long double c = a * b;
#elif PermanentTest_singleSIM_ADDSUBMUL == 1
        long double c = a - b;
#else
        long double c = a + b;
#endif
#endif
        memcpy(inp1, &a, sizeof(a));
        memcpy(inp2, &b, sizeof(b));
        PermanentTest_singleSIM_run(mavDFE, &actions);
#if PermanentTest_singleSIM_INPXBITS == -79 || PermanentTest_singleSIM_INPYBITS == -79    
        long double res;
#elif PermanentTest_singleSIM_INPXBITS == -64 || PermanentTest_singleSIM_INPYBITS == -64
        double res;
#elif PermanentTest_singleSIM_INPXBITS == -32 || PermanentTest_singleSIM_INPYBITS == -32
        float res;
#endif
        memcpy(&res, (char*)outp + outBytes - sizeof(res), sizeof(res));
        if (res != c && !(isnan(res) && isnan(c))) {
            printf("%Lf %Lf %Lf %Lf\n", (long double)a, (long double)b, (long double)c, (long double)res);
        }
    }
#else
    mpz_t a, b, c, res;
    mpz_t maskX, maskY, aexp, bexp;
    gmp_randstate_t state;
    gmp_randinit_default(state);
    mpz_init_set_ui(a, 1); mpz_init_set_ui(b, 1); mpz_init(c); mpz_init(res);
    mpz_mul_2exp(a, a, xbits); mpz_mul_2exp(b, b, ybits);
    mpz_sub_ui(a, a, 1); mpz_sub_ui(b, b, 1);
    if (signedX) { mpz_init(aexp); mpz_init_set(maskX, a); }
    if (signedY) { mpz_init(bexp); mpz_init_set(maskY, b); }
    for (int i = 0; i < 512; i++) {
        if (i == 1) {
            mpz_set_ui(a, 1); mpz_set_ui(b, 1);
            mpz_mul_2exp(a, a, xbits); mpz_mul_2exp(b, b, ybits);
            mpz_sub_ui(a, a, 2); mpz_sub_ui(b, b, 2);
        } else if (i == 2) {
            mpz_set_ui(a, 1); mpz_set_ui(b, 1);
            mpz_mul_2exp(a, a, xbits-1); mpz_mul_2exp(b, b, ybits-1);
        } else if (i == 3) {
            mpz_set_ui(a, 1); mpz_set_ui(b, 1);
            mpz_mul_2exp(a, a, xbits-1); mpz_mul_2exp(b, b, ybits-1);
            mpz_sub_ui(a, a, 1); mpz_sub_ui(b, b, 1);            
        } else if (i == 4) {
            mpz_set_ui(a, 1); mpz_set_ui(b, 1);
            mpz_mul_2exp(a, a, xbits-1); mpz_mul_2exp(b, b, ybits-1);
            mpz_add_ui(a, a, 1); mpz_add_ui(b, b, 1);            
        } else if (i != 0) {
            mpz_urandomb(a, state, xbits);
            mpz_urandomb(b, state, ybits);
        }
        if (signedX && mpz_sizeinbase(a, 2) == xbits) { mpz_clrbit(a, xbits-1); mpz_neg(a, a); }
        if (signedY && mpz_sizeinbase(b, 2) == ybits) { mpz_clrbit(b, ybits-1); mpz_neg(b, b); }
#if PermanentTest_singleSIM_ISCOMPLEX == 1
#if PermanentTest_singleSIM_ADDSUBMUL == 2
#else
#endif
#else
#if PermanentTest_singleSIM_ADDSUBMUL == 2
        mpz_mul(c, a, b);
#elif PermanentTest_singleSIM_ADDSUBMUL == 1
        mpz_sub(c, a, b);
#else
        mpz_add(c, a, b);
#endif
#endif
        memset(inp1, 0, xBytes); memset(inp2, 0, yBytes); memset(outp, 0, outBytes);
        if (signedX && mpz_sgn(a) < 0) {
            mpz_xor(aexp, a, maskX);
            mpz_add_ui(aexp, aexp, 1);
            mpz_export(inp1, NULL, -1, 16, 0, 0, aexp);
        } else
        mpz_export(inp1, NULL, -1, 16, 0, 0, a);
        if (signedY && mpz_sgn(b) < 0) {
            mpz_xor(bexp, b, maskY);
            mpz_add_ui(bexp, bexp, 1);
            mpz_export(inp2, NULL, -1, 16, 0, 0, bexp);
        } else
        mpz_export(inp2, NULL, -1, 16, 0, 0, b);

        PermanentTest_singleSIM_run(mavDFE, &actions);
        mpz_import(res, outBytes/16, -1, 16, 0, 0, outp);
        if ((signedX || signedY) && mpz_tstbit(res, outpbits-1)) {
            //mpz_com(res, res);
            for (int j = 0; j < outpbits; j++) {
                mpz_combit(res, j);
            }
            for (int j = outpbits; j < outBytes*8; j++) {
                mpz_clrbit(res, j);
            }             
            mpz_add_ui(res, res, 1);
            mpz_neg(res, res);
        }
        if (mpz_cmp(res, c) != 0) {
            gmp_printf("%Zx %Zx %Zx %Zx\n", a, b, c, res);
            return -1;
        }
    }
    gmp_randclear(state);
    mpz_clear(a); mpz_clear(b);
    mpz_clear(res); mpz_clear(c);
    if (signedX) { mpz_clear(aexp); mpz_clear(maskX); }
    if (signedY) { mpz_clear(bexp); mpz_clear(maskY); }
#endif
    free(inp1); free(inp2); free(outp);
    max_unload(mavDFE);
    max_file_free(mavMaxFile);
    PermanentTest_singleSIM_free();

    return 0;
}
