#include <stdint.h>
#include <stdlib.h>
#include <MaxSLiCInterface.h>
#include <string.h>
#include <math.h>
#include <gmp.h>
#include <unistd.h>

#include "PermanentTest_singleSIM.h"

#define SIZE 8

//DFE float uses IEEE style, not C long double style - bias is 32767 not 16383 (if (16, 64) used so we use (15, 64) for identical bias), mantissa stores 63 bits not 64, must adjust manually
long double dfeFloatToLD(void* p)
{
    __int128 res = *((__int128*)p);
    __int128 temp = res >> 63;
    if ((temp & 0x7FFF) == 0) //+/- 0
        res = (res & ((1ULL<<63)-1)) | (temp << 64);
    else //normal, +/- inf or +/- NaN
        res = ((res & ((1ULL<<63)-1)) | (1ULL << 63)) | (temp << 64);
    long double* pld = (long double*)&res;
    return *pld;
}

__int128 dfeLDToFloat(void* p)
{
    __uint64_t* pLD = p;
    __int128 res = *pLD & ((1ULL<<63)-1); //pseudo denormal, pseudo-infinity, pseudo NaN, Floating point Indefinite, Quiet Not a Number disregarded
    res |= (__uint64_t)(*(__uint16_t*)(pLD+1)) << 63;
    res |= (__int128)(*(__uint16_t*)(pLD+1) >> 1) << 64;
    return res;
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

#define SUBTRACTIONTESTS(s, t, mantExp, i) (s(t)((mantExp-1) - ((i == 64 ? 0x1.0p64l : (1ULL<<(i)))-1)))/mantExp
#define UNDERFLOWTESTS(s, t, mantExp, biasm2, i) (s(t)(mantExp-1))/mantExp*(i == 64 ? 0x1.0p64l : (1ULL<<(i)))*2/biasm2
#define UNDERFLOWSUBTESTS(s, t, mantExp, biasm2, i) (s(t)((mantExp-1) - ((i == 64 ? 0x1.0p64l : (1ULL<<(i)))-1)))/mantExp/biasm2
#define OVERFLOWTESTS(s, t, mantExp, biasm2, i) (s(t)(mantExp-1))/mantExp/(i == 64 ? 0x1.0p64l : (1ULL<<(i)))*biasm2*4
#define OVERFLOWSUBTESTS(s, t, mantExp, biasm2, i) (s(t)((mantExp-1) - ((i == 64 ? 0x1.0p64l : (1ULL<<(i)))-1)))/mantExp*biasm2*8

#define REP0(f, i, ...) f(__VA_ARGS__, i)
#define REP1(f, i, ...) REP0(f, i-1, __VA_ARGS__), f(__VA_ARGS__, i)
#define REP2(f, i, ...) REP1(f, i-1, __VA_ARGS__), f(__VA_ARGS__, i)
#define REP3(f, i, ...) REP2(f, i-1, __VA_ARGS__), f(__VA_ARGS__, i)
#define REP4(f, i, ...) REP3(f, i-1, __VA_ARGS__), f(__VA_ARGS__, i)
#define REP5(f, i, ...) REP4(f, i-1, __VA_ARGS__), f(__VA_ARGS__, i)
#define REP6(f, i, ...) REP5(f, i-1, __VA_ARGS__), f(__VA_ARGS__, i)
#define REP7(f, i, ...) REP6(f, i-1, __VA_ARGS__), f(__VA_ARGS__, i)
#define REP8(f, i, ...) REP7(f, i-1, __VA_ARGS__), f(__VA_ARGS__, i)
#define REP9(f, i, ...) REP8(f, i-1, __VA_ARGS__), f(__VA_ARGS__, i)
#define REP19(f, i, ...) REP9(f, i-10, __VA_ARGS__), REP9(f, i, __VA_ARGS__)
#define REP29(f, i, ...) REP19(f, i-10, __VA_ARGS__), REP9(f, i, __VA_ARGS__)
#define REP39(f, i, ...) REP29(f, i-10, __VA_ARGS__), REP9(f, i, __VA_ARGS__)
#define REP49(f, i, ...) REP39(f, i-10, __VA_ARGS__), REP9(f, i, __VA_ARGS__)
#define REP59(f, i, ...) REP49(f, i-10, __VA_ARGS__), REP9(f, i, __VA_ARGS__)

#define REP23(f, i, ...) REP19(f, i-4, __VA_ARGS__), REP3(f, i, __VA_ARGS__)
#define REP24(f, i, ...) REP19(f, i-5, __VA_ARGS__), REP4(f, i, __VA_ARGS__)
#define REP52(f, i, ...) REP49(f, i-3, __VA_ARGS__), REP2(f, i, __VA_ARGS__)
#define REP53(f, i, ...) REP49(f, i-4, __VA_ARGS__), REP3(f, i, __VA_ARGS__)
#define REP64(f, i, ...) REP59(f, i-5, __VA_ARGS__), REP4(f, i, __VA_ARGS__)
#define REP65(f, i, ...) REP59(f, i-6, __VA_ARGS__), REP5(f, i, __VA_ARGS__)

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define REPN(f, i, ...) CAT(REP, i)(f, i, __VA_ARGS__)

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
#if PermanentTest_singleSIM_USEFLOAT != 1
#if PermanentTest_singleSIM_ADDSUBMUL == 3
    int outpbits = 63-__builtin_clzl(xbits-1)+1 + 1 + 63-__builtin_clzl(ybits-1)+1 + 1;
#elif PermanentTest_singleSIM_ADDSUBMUL == 2
    int outpbits = xbits + ybits;
#else
    int outpbits = xbits + ybits + 1;
#endif
#else
    int outpbits = xbits > ybits ? xbits : ybits;
#endif
    size_t xBytes = (xbits+128-1)/128*16,
        yBytes = (ybits+128-1)/128*16,
        outBytes = (outpbits+128-1)/128*16;
    __uint128_t* inp1 = malloc(xBytes);
    __uint128_t* inp2 = malloc(yBytes);
    __uint128_t* outp = malloc(outBytes);
    PermanentTest_singleSIM_actions_t actions = { 1, inp1, xBytes, inp2, yBytes, outp, outBytes };
    
//#define BIAS (1<<(BITS-MANT))/2-1
#if PermanentTest_singleSIM_USEFLOAT == 1
#if PermanentTest_singleSIM_INPXBITS == -32
#define XTYPE float
#define XMANT 24
#define XMANTPOW2 0x1.0p24f 
//#define XBIAS 127
#define XBIASM2 0x1.0p125f
#elif PermanentTest_singleSIM_INPXBITS == -64
#define XTYPE double
#define XMANT 53
#define XMANTPOW2 0x1.0p53
//#define XBIAS 1023
#define XBIASM2 0x1.0p1021
#elif PermanentTest_singleSIM_INPXBITS == -79
#define XTYPE long double
#define XMANT 64
#define XMANTPOW2  0x1.0p64l
//#define XBIAS 16383
#define XBIASM2 0x1.0p16381l
#endif
#if PermanentTest_singleSIM_INPYBITS == -32
#define YTYPE float
#define YMANT 24
#define YMANTPOW2 0x1.0p24f 
//#define YBIAS 127
#define YBIASM2 0x1.0p125f
#elif PermanentTest_singleSIM_INPYBITS == -64
#define YTYPE double
#define YMANT 53
#define YMANTPOW2 0x1.0p53
//#define YBIAS 1023
#define YBIASM2 0x1.0p1021
#elif PermanentTest_singleSIM_INPYBITS == -79
#define YTYPE long double
#define YMANT 64
#define YMANTPOW2  0x1.0p64l
//#define YBIAS 16383
#define YBIASM2 0x1.0p16381l
#endif
    XTYPE bvaX[] = { 0.0, -0.0, 1.0, -1.0, 2.0, -2.0,
        REPN(SUBTRACTIONTESTS, XMANT, +, XTYPE, XMANTPOW2),
        REPN(UNDERFLOWTESTS, XMANT, +, XTYPE, XMANTPOW2, XBIASM2),
        REPN(UNDERFLOWSUBTESTS, XMANT, +, XTYPE, XMANTPOW2, XBIASM2),
        REPN(OVERFLOWTESTS, XMANT, +, XTYPE, XMANTPOW2, XBIASM2),
        REPN(OVERFLOWSUBTESTS, XMANT, +, XTYPE, XMANTPOW2, XBIASM2),        
        INFINITY, -INFINITY, NAN, -NAN }; 
    YTYPE bvaY[] = { 0.0, -0.0, 1.0, -1.0, 2.0, -2.0,
        REPN(SUBTRACTIONTESTS, YMANT, -, YTYPE, YMANTPOW2),
        REPN(UNDERFLOWTESTS, YMANT, -, YTYPE, YMANTPOW2, YBIASM2),
        REPN(UNDERFLOWSUBTESTS, YMANT, -, YTYPE, YMANTPOW2, YBIASM2),
        REPN(OVERFLOWTESTS, YMANT, -, YTYPE, YMANTPOW2, YBIASM2),
        REPN(OVERFLOWSUBTESTS, YMANT, -, YTYPE, YMANTPOW2, YBIASM2),    
        INFINITY, -INFINITY, NAN, -NAN };
    for (int i = 0; i < sizeof(bvaX)/sizeof(XTYPE); i++) {
        printf("TEST (%lu %lu): %16La\n", sizeof(bvaY)/sizeof(YTYPE)*i, actions.ticks_PermanentTestKernel, (long double)bvaX[i]);
        for (int j = 0; j < sizeof(bvaY)/sizeof(YTYPE); j++) {
    //for (int i = 0; i < 512; i++) {
        XTYPE a = 0.0; YTYPE b = 0.0;
        a = bvaX[i]; //gen_randdata(&a, sizeof(XTYPE) * 8);
        b = bvaY[j]; //gen_randdata(&b, sizeof(YTYPE) * 8);
#if PermanentTest_singleSIM_INPXBITS == -79 || PermanentTest_singleSIM_INPYBITS == -79    
        long double res, c;
#elif PermanentTest_singleSIM_INPXBITS == -64 || PermanentTest_singleSIM_INPYBITS == -64
        double res, c;
#elif PermanentTest_singleSIM_INPXBITS == -32 || PermanentTest_singleSIM_INPYBITS == -32
        float res, c;
#endif
#if PermanentTest_singleSIM_ISCOMPLEX == 1
#else
#if PermanentTest_singleSIM_ADDSUBMUL == 2
        c = a * b;
#elif PermanentTest_singleSIM_ADDSUBMUL == 1
        c = a - b;
#else
        c = a + b;
#endif
#endif
#if PermanentTest_singleSIM_INPXBITS == -79
        __int128 t1 = dfeLDToFloat(&a);
        memcpy(inp1, &t1, sizeof(t1));
#else    
        memcpy(inp1, &a, sizeof(a));
#endif
#if PermanentTest_singleSIM_INPYBITS == -79
        __int128 t2 = dfeLDToFloat(&b);
        memcpy(inp2, &t2, sizeof(t2));
#else
        memcpy(inp2, &b, sizeof(b));
#endif
        //usleep(100);
        PermanentTest_singleSIM_run(mavDFE, &actions);
#if PermanentTest_singleSIM_INPXBITS == -79 || PermanentTest_singleSIM_INPYBITS == -79    
        res = dfeFloatToLD(outp);
#else
        memcpy(&res, outp, sizeof(res));
#endif
        int resc = fpclassify(res), cc = fpclassify(c);
        if (res != c && !(resc == FP_NAN && cc == FP_NAN) && !((resc == FP_ZERO || resc == FP_SUBNORMAL) && (cc == FP_ZERO || cc == FP_SUBNORMAL))) {
            printf("%16La %16La %16La %16La\n", (long double)a, (long double)b, (long double)c, (long double)res);
        }
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
#if PermanentTest_singleSIM_ADDSUBMUL == 3
        int offs1 = 63-__builtin_clzl(xbits-1)+1, offs2 = 63-__builtin_clzl(ybits-1)+1;
        int isz1 = mpz_sgn(a) == 0, isz2 = mpz_sgn(b) == 0;        
        size_t lzc1 = isz1 ? (1<<offs1)-1 : xbits - mpz_sizeinbase(a, 2),
                lzc2 = isz2 ? (1<<offs2)-1 : ybits - mpz_sizeinbase(b, 2);
        mpz_set_ui(c, lzc1 | isz1 << offs1 | lzc2 << (1+offs1) | isz2 << (offs2+1+offs1));
#elif PermanentTest_singleSIM_ADDSUBMUL == 2
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
