#include <stdint.h>
#include <stdlib.h>
#include <MaxSLiCInterface.h>
#include <string.h>
#include <gmp.h>

#include "PermanentTest_singleSIM.h"

#define SIZE 8

void printnum(__uint128_t* chunks, size_t bytes)
{
    for (int chunk = bytes/16-1; chunk >= 0; chunk--)
        printf("%016lX %016lX ", (__uint64_t)(chunks[chunk] >> 64), (__uint64_t)chunks[chunk]);
    printf("\n");
}

int main(void)
{
    max_file_t* mavMaxFile = PermanentTest_singleSIM_init();
    max_engine_t* mavDFE = max_load(mavMaxFile, "local:*");
    int xbits = PermanentTest_singleSIM_INPXBITS, ybits = PermanentTest_singleSIM_INPYBITS;
    int signedX = xbits < 0, signedY = ybits < 0;
    if (xbits < 0) xbits = -xbits;
    if (ybits < 0) ybits = -ybits;
    int outpbits = xbits + ybits;
    size_t xBytes = (xbits+128-1)/128*16,
        yBytes = (ybits+128-1)/128*16,
        outBytes = (outpbits+128-1)/128*16;
    __uint128_t* inp1 = malloc(xBytes);
    __uint128_t* inp2 = malloc(yBytes);
    __uint128_t* outp = malloc(outBytes);
    mpz_t a, b, c, res;
    mpz_t maskX, maskY, aexp, bexp;
    gmp_randstate_t state;
    gmp_randinit_default(state);
    mpz_init_set_ui(a, 1); mpz_init_set_ui(b, 1); mpz_init(c); mpz_init(res);
    mpz_mul_2exp(a, a, xbits); mpz_mul_2exp(b, b, ybits);
    mpz_sub_ui(a, a, 1); mpz_sub_ui(b, b, 1);
    if (signedX) { mpz_init(aexp); mpz_init_set(maskX, a); }
    if (signedY) { mpz_init(bexp); mpz_init_set(maskY, b); }
    PermanentTest_singleSIM_actions_t actions = { 1, inp1, xBytes, inp2, yBytes, outp, outBytes };
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
        mpz_mul(c, a, b);
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
    free(inp1); free(inp2); free(outp);
    max_unload(mavDFE);
    max_file_free(mavMaxFile);
    PermanentTest_singleSIM_free();

    return 0;
}
