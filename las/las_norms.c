/*
 * las_norms.c
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include "las.h"
#define HAVE_SSE41

#ifdef HAVE_SSE41
#include <smmintrin.h>
#else
#ifdef HAVE_SSSE3
#include <tmmintrin.h>
#else
#ifdef HAVE_SSE3
#include <pmmintrin.h>
#else
#include <emmintrin.h>
#endif
#endif
#endif

/* Guard for the logarithms of norms, so that the value does not wrap around
   zero due to roundoff errors. */
#define GUARD 1

/* Input: i, double. i >= 0 needed!
   Output: o , trunc(o) == trunc(log2(i)) && o <= log2(i) < o + 0.0861.
   Careful: o ~= log2(i) iif add = 0x3FF00000 & scale = 1/0x100000.
   Add & scale are need to compute o'=f(log2(i)) where f is an affine function.
*/
static inline u_int8_t inttruncfastlog2(double i, double add, double scale)
{
	__asm__ __volatile__(
			"psrlq $0x20,  %0    \n"
			"cvtdq2pd      %0, %0\n" /* Mandatory in packed double even it's non packed! */
			: "+&x" (i));             /* Really need + here! i is not modified in C code! */
	return (u_int8_t) ((i-add)*scale);
}

static inline void w128itruncfastlog2fabs(__m128d i, __m128d add, __m128d scale, u_int8_t *addr, ssize_t decal, __m128d un)
{
  __asm__ __volatile__ (
	   "psllq     $0x01,    %0       \n" /* Dont use pabsd! */
	   "psrlq     $0x01,    %0       \n"
	   "addpd     %1,       %0       \n"
	   "shufps    $0xED,    %0,    %0\n"
	   "cvtdq2pd  %0,       %0       \n"
	   : "+&x"(i):"x"(un));
  i = _mm_mul_pd(_mm_sub_pd(i, add), scale);
  __asm__ __volatile__ (
	   "cvttpd2dq %0,       %0       \n" /* 0000 0000 000X 000Y */
	   "packssdw  %0,       %0       \n" /* 0000 0000 0000 0X0Y */
	   "punpcklbw %0,       %0       \n" /* 0000 0000 00XX 00YY */
	   "pshuflw   $0xA0,    %0,    %0\n" /* 0000 0000 XXXX YYYY */
	   "shufps    $0x50,    %0,    %0\n" /* XXXX XXXX YYYY YYYY */
	   : "+&x"(i));
  *(__m128d *)&addr[decal] = i; /* addr and decal are 16 bytes aligned: MOVAPD */
}

static inline void double_poly_scale(double *u, const double *t, unsigned int d, double h)
{
	double hpow;
	u[d] = t[d];
	for(hpow = h; --d != UINT_MAX; hpow *= h) u[d] = t[d] * hpow;
	return;
}

/* Initialize lognorms on the rational side for the bucket_region
 * number N.
 * For the moment, nothing clever, wrt discarding (a,b) pairs that are
 * not coprime, except for the line j=0.
 */
void init_rat_norms_bucket_region(unsigned char *S, gls_config_t cfg, unsigned int j, u_int32_t I_dim)
{
	/* #define COMPUTE_Y(G) ((LIKELY((G) > 1.)) ? inttruncfastlog2 ((G), add, scale) : GUARD) */
#define COMPUTE_Y(G) (inttruncfastlog2 ((G) + 1., add, scale))
	/* #define COMPUTE_Y(G) (inttruncfastlog2 ((G), add, scale)) */

	// #define DEBUG_INIT_RAT 1 /* For internal debug: trace all */
	// #define CHECK_INIT_RAT 1 /* For internal debug: control all */

	const int halfI = 1 << (I_dim - 1);
	const double halfI_double = (double) halfI;
	const double halfI_double_minus_one = halfI_double - 1.0;
	const double u0 = cfg->fijd[RPOLY_IDX][0]; // gj
    const double u1 = cfg->fijd[RPOLY_IDX][1]; // gi
    const double scale = cfg->scale[RPOLY_IDX] * (1. / 0x100000);
    const double add = 0x3FF00000 - GUARD / scale;

    assert(u1 != 0.);
    const double invu1 = 1. / u1;
    double u0j, d0_init, g, rac, d0, d1, i;
    size_t ts;
    unsigned int j1, inc;
    int int_i;
    u_int8_t oy, y;

    assert(isfinite(invu1));
    j1 = BUCKET_SHIFT - I_dim;
    j = j << j1;
    j1 = (1U << j1) + j;
    u0j = u0 * j;
    d0_init = cfg->cexp2[APOLY_IDX][((unsigned int)GUARD) - 1U];
    if(!j)
    {
    	// compute only the norm for i = 1. Everybody else is 255.
    	memset(S, 255, halfI<<1);
    	S[halfI + 1] = inttruncfastlog2(fabs(u1), add, scale);
    	S += halfI << 1;
    	j++;
    	u0j += u0;
    }
    for(; j < j1 ; j++, u0j += u0)
    {
#ifdef CHECK_INIT_RAT
    	unsigned char *cS = S + halfI;
    	memset (S, 0, halfI << 1);
#endif
    	int_i = -halfI;
    	g = u0j + u1 * int_i;
    	rac = u0j * (-invu1);
    	d0 = d0_init;
    	d1 = rac - d0 * rac;
    	/* g sign is mandatory at the beginning of the line intialization.
    	 * If g ~= 0, the sign of g is not significant; so the sign of g
    	 * is the sign of u1 (g+u1 is the next g value). CAREFUL:
    	 * g ~== 0 is coded g + u0j == u0j. It's possible it's not sufficient;
    	 * in this case, the right test will be g >= fabs(u0j)*(1./(1ULL<<51))
    	 */
    	/* Bug #16388 :
    	 * u0 = 5.19229685853482763e+33, u1 = 4.27451782396958897e+33.
    	 * So, u0j = 8.75421250348971938e+36, rac = -2.04800000000000000e+03.
    	 * At the beginning, int_i = -2048, so the true value of g is 0.
    	 * In fact, the computed value of g is 1.18059162071741130e+21.
    	 * And u0j + g = 8.75421250348972056e+36; so u0j + g != u0j.
    	 * => This test is false!
    	 * => We have to use the slower but correct test,
    	 * fabs(g) * (double) (((u_int64_t) 1)<<51) >= fabs(u0j).
    	 */
    	if(fabs(g) * (double) (((u_int64_t) 1) << 51) >= fabs(u0j))
    	{
    		if(signbit(g))
    		{
    			g = -g;
    			y = COMPUTE_Y(g);
    			if(rac >= -halfI) goto cas3; else goto cas2;
    		}
    		else
    		{
    			y = COMPUTE_Y(g);
    			if(rac >= -halfI) goto cas1; else goto cas4;
    		}
    	}
    	else
    	{
    		y = GUARD;
    		if(signbit(u1)) goto cas2; else goto cas4;
    	}
cas1:
		/* In this case, we exit from the loop when ts == 0, at the exception
		 * of the first iteration. In this special case, old_i = -halfI and
		 * int_i = trunc (i), where i=[inverse of the function g](trunc(y)) and
		 * y=g(old_i).
		 * So, it's possible if y is very near trunc(y), old_i == int_i, so ts == 0.
		 * We have to iterate at least one time to avoid this case => this is the
		 * use of inc here.
		 */
		for(i = rac + cfg->cexp2[RPOLY_IDX][y] * invu1, inc = 1;; y--)
		{
			ts = -int_i;
			if(i >= halfI_double)
			{
				ts += halfI;
#ifdef DEBUG_INIT_RAT
				fprintf(stderr, "A1.END : i1=%ld i2=%d, ts=%ld, y=%u, rac=%e\n", halfI - ts, halfI, ts, y, rac);
#endif
				memset(S, y, ts);
				S += ts;
				goto nextj;
			}
			int_i = (int) i; /* Overflow is not possible here */
			ts += int_i;
#ifdef DEBUG_INIT_RAT
			fprintf(stderr, "A1 : i1=%ld, i2=%d, ts=%ld, y=%u, rac=%e\n", int_i - ts, int_i, ts, y, rac);
#endif
			if(ts <= MEMSET_MIN)
			{
				if(!(ts + inc)) goto np1;
				memset(S, y, MEMSET_MIN);
			}
			else
				memset(S, y, ts);
			S += ts;
			i = i * d0 + d1;
			inc = 0;
		}
np1:
		g = u0j + u1 * int_i;
		if(trunc(rac) >= halfI_double_minus_one)
		{
			for(; int_i < halfI; int_i++)
			{
				y = COMPUTE_Y(g);
#ifdef DEBUG_INIT_RAT
				fprintf(stderr, "A2.1 : i=%d, y=%u, rac=%e\n", int_i, y, rac);
#endif
				*S++ = y;
				g += u1;
			}
			goto nextj;
		}
		for(inc = 0; g > 0; g += u1)
		{
			y = COMPUTE_Y(g);
#ifdef DEBUG_INIT_RAT
			fprintf(stderr, "A2.2 : i=%d, y=%u, rac=%e\n", int_i + inc, y, rac);
#endif
			S[inc++] = y;
		}
		int_i += inc;
		S += inc;
		g = -g;
		y = COMPUTE_Y(g);
cas2:
		do
		{
#ifdef DEBUG_INIT_RAT
			fprintf(stderr, "A3 : i=%d, y=%u, rac=%e\n", int_i, y, rac);
#endif
			*S++ = y;
			if(++int_i >= halfI) goto nextj;
			oy = y;
			g -= u1;
			y = COMPUTE_Y(g);
		} while (oy != y);
		d0 = 1. / d0;
		d1 = rac - d0 * rac;
		y++;
		i = rac - cfg->cexp2[RPOLY_IDX][(unsigned int)y + 1] * invu1;
		for(;; y++)
		{
			ts = -int_i;
			if(i >= halfI_double)
			{
				ts += halfI;
#ifdef DEBUG_INIT_RAT
				fprintf(stderr, "A4.END : i1=%ld, i2=%d, ts=%ld, y=%u, rac=%e\n", halfI - ts, halfI, ts, y, rac);
#endif
				memset(S, y, ts);
				S += ts;
				goto nextj;
			}
			int_i = (int) i; /* Overflow is not possible here */
			ts += int_i;
#ifdef DEBUG_INIT_RAT
			fprintf(stderr, "A4 : i1=%ld, i2=%d, ts=%ld, y=%u, rac=%e\n", int_i - ts, int_i, ts, y, rac);
#endif
			if(ts <= MEMSET_MIN)
				memset(S, y, MEMSET_MIN);
			else
				memset(S, y, ts);
			S += ts;
			i = i * d0 + d1;
		}

		/* Now, the same from cas1 but log2(-g): CAREFUL, not the same formula */
cas3:
		for(i = rac - cfg->cexp2[RPOLY_IDX][y] * invu1, inc = 1;; y--)
		{
			ts = -int_i;
			if(i >= halfI_double)
			{
				ts += halfI;
#ifdef DEBUG_INIT_RAT
				fprintf(stderr, "B1.END : i1=%ld, i2=%d, ts=%ld, y=%u, rac=%e\n", halfI - ts, halfI, ts, y, rac);
#endif
				memset(S, y, ts);
				S += ts;
				goto nextj;
			}
			int_i = (int) i; /* Overflow is not possible here */
			ts += int_i;
#ifdef DEBUG_INIT_RAT
			fprintf(stderr, "B1 : i1=%ld, i2=%d, ts=%ld, y=%u, rac=%e\n", int_i - ts, int_i, ts, y, rac);
#endif
			if(ts <= MEMSET_MIN)
			{
				if(!(ts + inc)) goto np2;
				memset(S, y, MEMSET_MIN);
			}
			else
				memset(S, y, ts);
			S += ts;
			i = i * d0 + d1;
			inc = 0;
		}
np2:
		g = -(u0j + u1 * int_i);
		if(trunc(rac) >= halfI_double_minus_one)
		{
			for(; int_i < halfI; int_i++)
			{
				y = COMPUTE_Y(g);
#ifdef DEBUG_INIT_RAT
				fprintf(stderr, "B2.1 : i=%d, y=%u, rac=%e\n", int_i, y, rac);
#endif
				*S++ = y;
				g -= u1;
			}
			goto nextj;
		}
		for(inc = 0; g > 0; g -= u1)
		{
			y = COMPUTE_Y(g);
#ifdef DEBUG_INIT_RAT
			fprintf(stderr, "B2.2 : i=%d, y=%u, rac=%e\n", int_i + inc, y, rac);
#endif
			S[inc++] = y;
		}
		int_i += inc;
		S += inc;
		g = -g;
		y = COMPUTE_Y(g);
cas4:
		do
		{
#ifdef DEBUG_INIT_RAT
			fprintf(stderr, "B3 : i=%d, y=%u, rac=%e\n", int_i, y, rac);
#endif
			*S++ = y;
			if(++int_i == halfI)
				goto nextj;
			oy = y;
			g += u1;
			y = COMPUTE_Y(g);
		} while(oy != y);
		d0 = 1. / d0;
		d1 = rac - d0 * rac;
		y++;
		i = rac + cfg->cexp2[RPOLY_IDX][(unsigned int)y + 1] * invu1;
		for(;;y++)
		{
			ts = -int_i;
			if(i >= halfI_double)
			{
				ts += halfI;
#ifdef DEBUG_INIT_RAT
				fprintf(stderr, "B4.END : i1=%ld i2=%d, ts=%ld, y=%u, rac=%e\n", halfI - ts, halfI, ts, y, rac);
#endif
				memset(S, y, ts);
				S += ts;
				goto nextj;
			}
			int_i = (int) i; /* Overflow is not possible here */
			ts += int_i;
#ifdef DEBUG_INIT_RAT
			fprintf(stderr, "B4 : i1=%ld i2=%d, ts=%ld, y=%u, rac=%e\n", int_i - ts, int_i, ts, y, rac);
#endif
			if(ts <= MEMSET_MIN)
				memset(S, y, MEMSET_MIN);
			else
				memset(S, y, ts);
			S += ts;
			i = i * d0 + d1;
		}
nextj:
		; /* gcc needs something after a label */
#ifdef CHECK_INIT_RAT
		/* First MANDATORY condition. The exact line must be initialised. */
		if(cS + halfI != S)
		{
			fprintf(stderr, "init_rat_norms_bucket_region: S control Error: OldS(%p) + I(%d) != S(%p)", cS - halfI, si->I, S);
			exit (1);
		}
		/* Not really mandatory: when g ~= 0, the two formula (log2 & fastlog2) could really differ */
		int_i = -halfI;
		g = u0j + u1 * int_i;
		while(int_i < halfI)
		{
			y = (fabs(g) > 1.) ? log2(fabs(g))*cfg->scale[RPOLY_IDX] + GUARD : GUARD;
			if(fabs(cS[int_i] - y) > 1.)
			{
				fprintf(stderr, "init_rat_norms_bucket_region: possible problem in S, offset %d:\n"
						"real value=%d, S init value=%d. If g + uj0 ~= uj0,\n"
						"it could be OK: g=%e, u0j=%e\n", int_i, y, cS[int_i], g, u0j);
			}
			int_i++;
			g += u1;
		}
#endif
	}
    return;
}


#define VERT_NORM_STRIDE 4

#ifdef HAVE_SSE41
#define TSTZXMM(A) _mm_testz_si128(A,A)
#else
#define TSTZXMM(A) ((uint16_t)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_add_epi8(A,tt),tt))==0xFFFF)
#endif

/* This macro avoids a stupid & boring C types control */
#define _MM_SHUFFLE_EPI32(A,B,C) __asm__ __volatile__ ("pshufd $" #C ", %1, %0\n":"=x"(A):"x"(B))

/* CAREFUL! __m128 a=x|y => x is the strongest quadword, higher in memory.
   INITALGN: uN= (j^N)*(alg->fijd[N]) | (j^N)*(alg->fijd[N])
   INITALG0 to 2 are simple. 3,5,7,9 & 4,6,8 have similar algorithms. */
#define INITALG0 do {							\
    uu0=_mm_set1_epi64((__m64)(0x0101010101010101*			\
			       inttruncfastlog2(fabs(alg->fijd[0]),\
						*(double *)&add,*(double *)&scale))); \
  } while (0)

#define INITALG1(A) do {			\
    u1=_mm_load1_pd(cfg->fijd[APOLY_IDX]+1);		\
    u0=_mm_set1_pd((double)(A)*cfg->fijd[APOLY_IDX][0]);	\
  } while (0)

#define INITALG2(A) do {			\
    h=_mm_set1_pd((double)(A));			\
    u2=_mm_load1_pd(cfg->fijd[APOLY_IDX]+2);		\
    u1=_mm_load1_pd(cfg->fijd[APOLY_IDX]+1);		\
    u0=_mm_load1_pd(cfg->fijd[APOLY_IDX]);			\
    u1=_mm_mul_pd(u1,h);			\
    u0=_mm_mul_pd(u0,h);			\
    u0=_mm_mul_pd(u0,h);			\
  } while (0)

#define INITALG3(A) do {					\
    h=_mm_set_sd((double)(A));     /* h=0|j */			\
    _MM_SHUFFLE_EPI32(g, h, 0x44);				\
    g=_mm_mul_pd(g,g);             /* g=j2|j2 */		\
    u2=_mm_load_pd(cfg->fijd[APOLY_IDX]+2);   /* u2=d3|d2 */		\
    u0=_mm_load_pd(cfg->fijd[APOLY_IDX]);     /* u0=d1|d0 */		\
    u2=_mm_mul_sd(u2,h);           /* u2=d3|d2*j */		\
    u0=_mm_mul_sd(u0,h);					\
    u0=_mm_mul_pd(u0,g);					\
    _MM_SHUFFLE_EPI32(u1, u0, 0xEE);				\
    u0=_mm_unpacklo_pd(u0,u0);					\
    _MM_SHUFFLE_EPI32(u3, u2, 0xEE);				\
    u0=_mm_unpacklo_pd(u2,u2);					\
  } while (0)

#define INITALG4(A) do {					\
    h=_mm_set1_pd((double)(A));					\
    h=_mm_mul_sd(h,h);             /* h=j|j2 */			\
    _MM_SHUFFLE_EPI32(g, h, 0x44); /* g=j2|j2 */		\
    u4=_mm_load1_pd(cfg->fijd[APOLY_IDX]+4);				\
    u2=_mm_load_pd(cfg->fijd[APOLY_IDX]+2);				\
    u0=_mm_load_pd(cfg->fijd[APOLY_IDX]);					\
    u2=_mm_mul_pd(u2,h);					\
    u0=_mm_mul_pd(u0,h);					\
    u0=_mm_mul_pd(u0,g);					\
    _MM_SHUFFLE_EPI32(u1, u0, 0xEE);				\
    u0=_mm_unpacklo_pd(u0,u0);					\
    _MM_SHUFFLE_EPI32(u3, u2, 0xEE);				\
    u2=_mm_unpacklo_pd(u2,u2);					\
  } while (0)

#define INITALG5(A) do {					\
    h=_mm_set_sd((double)(A));					\
    _MM_SHUFFLE_EPI32(g, h, 0x44);				\
    g=_mm_mul_pd(g,g);						\
    u4=_mm_load_pd(cfg->fijd[APOLY_IDX]+4);				\
    u2=_mm_load_pd(cfg->fijd[APOLY_IDX]+2);				\
    u0=_mm_load_pd(cfg->fijd[APOLY_IDX]);					\
    u4=_mm_mul_sd(u4,h);					\
    u2=_mm_mul_sd(u2,h);					\
    u0=_mm_mul_sd(u0,h);					\
    u2=_mm_mul_pd(u2,g);					\
    u0=_mm_mul_pd(u0,g);					\
    u0=_mm_mul_pd(u0,g);					\
    _MM_SHUFFLE_EPI32(u1, u0, 0xEE);				\
    u0=_mm_unpacklo_pd(u0,u0);					\
    _MM_SHUFFLE_EPI32(u3, u2, 0xEE);				\
    u2=_mm_unpacklo_pd(u2,u2);					\
    _MM_SHUFFLE_EPI32(u5, u4, 0xEE);				\
    u4=_mm_unpacklo_pd(u4,u4);					\
  } while (0)

#define INITALG6(A) do {					\
    h=_mm_set1_pd((double)(A));					\
    h=_mm_mul_sd(h,h);						\
    _MM_SHUFFLE_EPI32(g, h, 0x44);				\
    u6=_mm_load1_pd(cfg->fijd[APOLY_IDX]+6);				\
    u4=_mm_load_pd(cfg->fijd[APOLY_IDX]+4);				\
    u2=_mm_load_pd(cfg->fijd[APOLY_IDX]+2);				\
    u0=_mm_load_pd(cfg->fijd[APOLY_IDX]);					\
    u4=_mm_mul_pd(u4,h);					\
    u2=_mm_mul_pd(u2,h);					\
    u0=_mm_mul_pd(u0,h);					\
    u2=_mm_mul_pd(u2,g);					\
    u0=_mm_mul_pd(u0,g);					\
    u0=_mm_mul_pd(u0,g);					\
    _MM_SHUFFLE_EPI32(u1, u0, 0xEE);				\
    u0=_mm_unpacklo_pd(u0,u0);					\
    _MM_SHUFFLE_EPI32(u3, u2, 0xEE);				\
    u2=_mm_unpacklo_pd(u2,u2);					\
    _MM_SHUFFLE_EPI32(u5, u4, 0xEE);				\
    u4=_mm_unpacklo_pd(u4,u4);					\
  } while (0)

#define INITALG7(A) do {					\
    h=_mm_set_sd((double)(A));					\
    _MM_SHUFFLE_EPI32(g, h, 0x44);				\
    g=_mm_mul_pd(g,g);						\
    u6=_mm_load_pd(cfg->fijd[APOLY_IDX]+6);				\
    u4=_mm_load_pd(cfg->fijd[APOLY_IDX]+4);				\
    u2=_mm_load_pd(cfg->fijd[APOLY_IDX]+2);				\
    u0=_mm_load_pd(cfg->fijd[APOLY_IDX]);					\
    u6=_mm_mul_sd(u6,h);					\
    u4=_mm_mul_sd(u4,h);					\
    u2=_mm_mul_sd(u2,h);					\
    u0=_mm_mul_sd(u0,h);					\
    u4=_mm_mul_pd(u4,g);					\
    u0=_mm_mul_pd(u0,g);					\
    g=_mm_mul_pd(g,g);						\
    u2=_mm_mul_pd(u2,g);					\
    u0=_mm_mul_pd(u0,g);					\
    _MM_SHUFFLE_EPI32(u1, u0, 0xEE);				\
    u0=_mm_unpacklo_pd(u0,u0);					\
    _MM_SHUFFLE_EPI32(u3, u2, 0xEE);				\
    u2=_mm_unpacklo_pd(u2,u2);					\
    _MM_SHUFFLE_EPI32(u5, u4, 0xEE);				\
    u4=_mm_unpacklo_pd(u4,u4);					\
    _MM_SHUFFLE_EPI32(u7, u6, 0xEE);				\
    u6=_mm_unpacklo_pd(u6,u6);					\
  } while (0)

#define INITALG8(A) do {					\
    h=_mm_set1_pd((double)(A));					\
    h=_mm_mul_sd(h,h);						\
    _MM_SHUFFLE_EPI32(g, h, 0x44);				\
    u8=_mm_load1_pd(cfg->fijd[APOLY_IDX]+8);				\
    u6=_mm_load_pd(cfg->fijd[APOLY_IDX]+6);				\
    u4=_mm_load_pd(cfg->fijd[APOLY_IDX]+4);				\
    u2=_mm_load_pd(cfg->fijd[APOLY_IDX]+2);				\
    u0=_mm_load_pd(cfg->fijd[APOLY_IDX]);					\
    u6=_mm_mul_pd(u6,h);					\
    u4=_mm_mul_pd(u4,h);					\
    u2=_mm_mul_pd(u2,h);					\
    u0=_mm_mul_pd(u0,h);					\
    u4=_mm_mul_pd(u4,g);					\
    u0=_mm_mul_pd(u0,g);					\
    g=_mm_mul_pd(g,g);						\
    u2=_mm_mul_pd(u2,g);					\
    u0=_mm_mul_pd(u0,g);					\
    _MM_SHUFFLE_EPI32(u1, u0, 0xEE);				\
    u0=_mm_unpacklo_pd(u0,u0);					\
    _MM_SHUFFLE_EPI32(u3, u2, 0xEE);				\
    u2=_mm_unpacklo_pd(u2,u2);					\
    _MM_SHUFFLE_EPI32(u5, u4, 0xEE);				\
    u4=_mm_unpacklo_pd(u4,u4);					\
    _MM_SHUFFLE_EPI32(u7, u6, 0xEE);				\
    u6=_mm_unpacklo_pd(u6,u6);					\
  } while (0)

#define INITALG9(A) do {					\
    h =_mm_set_sd((double)(A));					\
    _MM_SHUFFLE_EPI32(g, h, 0x44);				\
    g=_mm_mul_pd(g,g);						\
    u8=_mm_load_pd(cfg->fijd[APOLY_IDX]+8);				\
    u6=_mm_load_pd(cfg->fijd[APOLY_IDX]+6);				\
    u4=_mm_load_pd(cfg->fijd[APOLY_IDX]+4);				\
    u2=_mm_load_pd(cfg->fijd[APOLY_IDX]+2);				\
    u0=_mm_load_pd(cfg->fijd[APOLY_IDX]);					\
    u8=_mm_mul_sd(u8,h);					\
    u6=_mm_mul_sd(u6,h);					\
    u4=_mm_mul_sd(u4,h);					\
    u2=_mm_mul_sd(u2,h);					\
    u0=_mm_mul_sd(u0,h);					\
    u6=_mm_mul_pd(u6,g);					\
    u2=_mm_mul_pd(u2,g);					\
    g=_mm_mul_pd(g,g);						\
    u4=_mm_mul_pd(u4,g);					\
    u2=_mm_mul_pd(u2,g);					\
    g=_mm_mul_pd(g,g);						\
    u0=_mm_mul_pd(u0,g);					\
    _MM_SHUFFLE_EPI32(u1, u0, 0xEE);				\
    u0=_mm_unpacklo_pd(u0,u0);					\
    _MM_SHUFFLE_EPI32(u3, u2, 0xEE);				\
    u2=_mm_unpacklo_pd(u2,u2);					\
    _MM_SHUFFLE_EPI32(u5, u4, 0xEE);				\
    u4=_mm_unpacklo_pd(u4,u4);					\
    _MM_SHUFFLE_EPI32(u7, u6, 0xEE);				\
    u6=_mm_unpacklo_pd(u6,u6);					\
    _MM_SHUFFLE_EPI32(u9, u8, 0xEE);				\
    u8=_mm_unpacklo_pd(u8,u8);					\
  } while (0)

#define INITALGD(A) do {					\
	u_int32_t k; \
    double du[d+1];						\
    double_poly_scale(du, cfg->fijd[APOLY_IDX], d, (double) (A));		\
    for (k=0; k<=d; k++) u[k] = _mm_set1_pd(du[k]);	\
  } while (0)

#define G1 g=_mm_add_pd(_mm_mul_pd(g,u1),u0)

#define G2 g=_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(g,u2),u1),h),u0)

#define G3 g=_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(g,u3),u2),h),u1),h),u0)

#define G4 g=_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(g,u4),u3),h),u2),h),u1),h),u0)

#define G5 g=_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(g,u5),u4),h),u3),h),u2),h),u1),h),u0)

#define G6 g=_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(g,u6),u5),h),u4),h),u3),h),u2),h),u1),h),u0)

#define G7 g=_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(g,u7),u6),h),u5),h),u4),h),u3),h),u2),h),u1),h),u0)

#define G8 g=_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(g,u8),u7),h),u6),h),u5),h),u4),h),u3),h),u2),h),u1),h),u0)

#define G9 g=_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(_mm_add_pd(_mm_mul_pd(g,u9),u8),h),u7),h),u6),h),u5),h),u4),h),u3),h),u2),h),u1),h),u0)

#define GD do {								\
	unsigned int k; \
	g=u[d-1];							\
	for(k = d; --k != UINT_MAX; g = _mm_add_pd(_mm_mul_pd(g,h),u[k])); \
      } while (0)

#define ALG_INIT_SCALE_ADD_UN_A const __m128d				\
      un = _mm_set1_pd(1.),				\
	a = _mm_set1_pd(16.),				\
	add = _mm_set1_pd(0x3FF00000 - GUARD / (cfg->scale[APOLY_IDX] * (1./0x100000))), \
	scale = _mm_set1_pd(cfg->scale[APOLY_IDX] * (1./0x100000))

#define ALG_INIT_SCALE_ADD_UN_A_TT					\
  ALG_INIT_SCALE_ADD_UN_A;						\
  const __m128i tt =							\
    _mm_set1_epi64((__m64)(0x0101010101010101* (uint64_t)		\
			   ((cfg->bound[RPOLY_IDX] +		\
			     (cfg->bound[RPOLY_IDX] != 255 ? 1 : 0)))))

/**************** 2: HAVE_SSE2, ALG_LAZY, !ALG_RAT ***********************/
/* Smart initialization of the algebraics. Computes the central
   initialization value of a box of (horizontail=8,vertical=p) and
   propagates it in the box. 8 is locked by the code and p is the
   minimum of VERT_NORM_STRIDE (#define, =4) and ej, with ej =
   2^(LOG_BUCKET_REGION-I_dim).
   So, the fastest code is for logI <= 14.
   It's the fastest initialization of the algebraics.
   SSE version, 32bits compatible. */
void init_alg_norms_bucket_region(u_int8_t *S, gls_config_t cfg, unsigned int j, u_int32_t I_dim)
{
	u_int32_t I = (1U << I_dim);
	ssize_t ih, Idiv2 = (I >> 1);
	unsigned int ej = 1U << (BUCKET_SHIFT - I_dim);
	unsigned int d = cfg->poly[APOLY_IDX]->deg;
	unsigned int p = ej < VERT_NORM_STRIDE ? ej : VERT_NORM_STRIDE;

#define FILL_OTHER_LINES do \
	{ \
		if(p > 1) \
		{ \
			unsigned int k; \
			unsigned char *mS = S + Idiv2; \
			S -= Idiv2; \
			for(k = 1; k < p; k++, mS += (Idiv2 << 1)) \
				memcpy(mS, S, (Idiv2 << 1)); \
			S = mS + Idiv2;	\
			j += p; \
		} \
		else \
		{ \
			do \
			{ \
				S += (Idiv2 << 1); \
			j++; \
			} while(0); \
		} \
	} while (0)

	j *= ej;
	ej += j;
	S += Idiv2;
	switch(d)
	{
	case 0:
		{
			ALG_INIT_SCALE_ADD_UN_A;
			memset(S-Idiv2,inttruncfastlog2(fabs(cfg->fijd[APOLY_IDX][0]),*(double *)&add,*(double *)&scale),(Idiv2<<1)*(ej-j));
		}
		return;
	case 1:
		{
			__m128d h, g, u0, u1;
			ALG_INIT_SCALE_ADD_UN_A;
			while(j < ej)
			{
				INITALG1(j + (p >> 1));
				h = _mm_set_pd(11 - Idiv2, 3 - Idiv2);
				for(ih = -Idiv2; ih < Idiv2; ih += 16)
				{
					g = h;
					G1;
					w128itruncfastlog2fabs(g, add, scale, S, ih, un);
					h = _mm_add_pd(h, a);
				}
				FILL_OTHER_LINES;
			}
		}
		return;
	case 2:
		{
			__m128d h, g, u0, u1, u2;
			ALG_INIT_SCALE_ADD_UN_A;
			while(j < ej)
			{
				INITALG2(j + (p >> 1));
				h = _mm_set_pd(11 - Idiv2, 3 - Idiv2);
				for(ih = -Idiv2; ih < Idiv2; ih += 16)
				{
					g = h;
					G2;
					w128itruncfastlog2fabs(g, add, scale, S, ih, un);
					h = _mm_add_pd(h, a);
				}
				FILL_OTHER_LINES;
			}
		}
		return;
	case 3:
		{
			__m128d h, g, u0, u1, u2, u3;
			ALG_INIT_SCALE_ADD_UN_A;
			while(j < ej)
			{
				INITALG3(j + (p >> 1));
				h = _mm_set_pd(11 - Idiv2, 3 - Idiv2);
				for(ih = -Idiv2; ih < Idiv2; ih += 16)
				{
					g = h;
					G3;
					w128itruncfastlog2fabs(g, add, scale, S, ih, un);
					h = _mm_add_pd(h, a);
				}
				FILL_OTHER_LINES;
			}
		}
		return;
	case 4:
		{
			__m128d h, g, u0, u1, u2, u3, u4;
			ALG_INIT_SCALE_ADD_UN_A;
			while(j < ej)
			{
				INITALG4(j + (p >> 1));
				h = _mm_set_pd(11 - Idiv2, 3 - Idiv2);
				for(ih = -Idiv2; ih < Idiv2; ih += 16)
				{
					g = h;
					G4;
					w128itruncfastlog2fabs(g, add, scale, S, ih, un);
					h = _mm_add_pd(h, a);
				}
				FILL_OTHER_LINES;
			}
		}
		return;
	case 5:
		{
			__m128d h, g, u0, u1, u2, u3, u4, u5;
			ALG_INIT_SCALE_ADD_UN_A;
			while(j < ej)
			{
				INITALG5(j + (p >> 1));
				h = _mm_set_pd(11 - Idiv2, 3 - Idiv2);
				for(ih = -Idiv2; ih < Idiv2; ih += 16)
				{
					g = h;
					G5;
					w128itruncfastlog2fabs(g, add, scale, S, ih, un);
					h = _mm_add_pd(h, a);
				}
				FILL_OTHER_LINES;
			}
		}
		return;
	case 6:
		{
			__m128d h, g, u0, u1, u2, u3, u4, u5, u6;
			ALG_INIT_SCALE_ADD_UN_A;
			while(j < ej)
			{
				INITALG6(j + (p >> 1));
				h = _mm_set_pd(11 - Idiv2, 3 - Idiv2);
				for(ih = -Idiv2; ih < Idiv2; ih += 16)
				{
					g = h;
					G6;
					w128itruncfastlog2fabs(g, add, scale, S, ih, un);
					h = _mm_add_pd(h, a);
				}
				FILL_OTHER_LINES;
			}
		}
		return;
  case 7 : {
    __m128d h, g, u0, u1, u2, u3, u4, u5, u6, u7;
    ALG_INIT_SCALE_ADD_UN_A;
    while (j < ej) {
      INITALG7(j + (p >> 1));
      h = _mm_set_pd(11 - Idiv2, 3 - Idiv2);
      for (ih = -Idiv2; ih < Idiv2; ih += 16) {
	g = h; G7;
	w128itruncfastlog2fabs(g, add, scale, S, ih, un);
	h = _mm_add_pd(h, a);
      }
      FILL_OTHER_LINES;
    }}
    return;
  case 8 : {
    __m128d h, g, u0, u1, u2, u3, u4, u5, u6, u7, u8;
    ALG_INIT_SCALE_ADD_UN_A;
    while (j < ej) {
      INITALG8(j + (p >> 1));
      h =_mm_set_pd(11 - Idiv2, 3 - Idiv2);
      for (ih = -Idiv2; ih < Idiv2; ih += 16) {
	g = h; G8;
	w128itruncfastlog2fabs(g, add, scale, S, ih, un);
	h = _mm_add_pd(h, a);
      }
      FILL_OTHER_LINES;
    }}
    return;
  case 9 : {
    __m128d h, g, u0, u1, u2, u3, u4, u5, u6, u7, u8, u9;
    ALG_INIT_SCALE_ADD_UN_A;
    while (j < ej) {
      INITALG9(j + (p >> 1));
      h =_mm_set_pd(11 - Idiv2, 3 - Idiv2);
      for (ih = -Idiv2; ih < Idiv2; ih += 16) {
	g = h; G9;
	w128itruncfastlog2fabs(g, add, scale, S, ih, un);
	h = _mm_add_pd(h, a);
      }
      FILL_OTHER_LINES;
    }}
    return;
  default: {
    __m128d h, g, u[d+1];
    ALG_INIT_SCALE_ADD_UN_A;
    while (j < ej) {
      INITALGD(j + (p >> 1));
      h =_mm_set_pd(11 - Idiv2, 3 - Idiv2);
      for (ih = -Idiv2; ih < Idiv2; ih += 16) {
	g = h; GD;
	w128itruncfastlog2fabs(g, add, scale, S, ih, un);
	h = _mm_add_pd(h, a);
      }
      FILL_OTHER_LINES;
    }}
    return;
  }
}
#undef FILL_OTHER_LINES


typedef struct {
	unsigned int deg;
	double *c;         /* array of deg+1 entries */
} double_poly_struct_t;

typedef double_poly_struct_t double_poly_t[1];
typedef double_poly_struct_t *double_poly_ptr;
typedef double_poly_struct_t *double_poly_ptr;

/* Initialize a polynomial of degree d */
void double_poly_init(double_poly_ptr p, unsigned int d)
{
	p->c = malloc((d + 1) * sizeof (double));
	p->deg = d;
	return;
}

/* Clear a polynomial */
void double_poly_clear(double_poly_ptr p)
{
	free(p->c);
}

/* Evaluate the polynomial p at point x */
double double_poly_eval(double_poly_ptr p, const double x)
{
	double r;
	unsigned int k;
	const double *f = p->c;
	const unsigned int deg = p->deg;

	switch(deg)
	{
		case 0: return f[0];
		case 1: return f[0]+x*f[1];
		case 2: return f[0]+x*(f[1]+x*f[2]);
		case 3: return f[0]+x*(f[1]+x*(f[2]+x*f[3]));
		case 4: return f[0]+x*(f[1]+x*(f[2]+x*(f[3]+x*f[4])));
		case 5: return f[0]+x*(f[1]+x*(f[2]+x*(f[3]+x*(f[4]+x*f[5]))));
		case 6: return f[0]+x*(f[1]+x*(f[2]+x*(f[3]+x*(f[4]+x*(f[5]+x*f[6])))));
		case 7: return f[0]+x*(f[1]+x*(f[2]+x*(f[3]+x*(f[4]+x*(f[5]+x*(f[6]+x*f[7]))))));
		case 8: return f[0]+x*(f[1]+x*(f[2]+x*(f[3]+x*(f[4]+x*(f[5]+x*(f[6]+x*(f[7]+x*f[8])))))));
		case 9: return f[0]+x*(f[1]+x*(f[2]+x*(f[3]+x*(f[4]+x*(f[5]+x*(f[6]+x*(f[7]+x*(f[8]+x*f[9]))))))));
		default: for (r = f[deg], k = deg - 1; k != UINT_MAX; r = r * x + f[k--]); return r;
	}
}

/* assuming g(a)*g(b) < 0, and g has a single root in [a, b],
 * refines that root by dichotomy with n iterations.
 * Assumes sa is of same sign as g(a).
 */
double double_poly_dichotomy(double_poly_ptr p, double a, double b, double sa, unsigned int n)
{
	double s;

	do
	{
		s = (a + b) * 0.5;
		if (double_poly_eval(p, s) * sa > 0)
			a = s;
		else
			b = s;
	}
	while (n-- > 0);
	return (a + b) * 0.5;
}

/* Stores the derivative of f in df. Assumes df different from f.
 * Assumes df has been initialized with degree at least f->deg - 1.
 */
void double_poly_derivative(double_poly_ptr df, double_poly_ptr f)
{
	unsigned int n;
	if(f->deg == 0)
	{
		df->deg = 0; /* How do we store deg -\infty polynomials? */
		df->c[0] = 0;
		return;
	}

	// at this point, f->deg >=1
	df->deg = f->deg-1;
	for(n = 0; n <= f->deg - 1; n++)
		df->c[n] = f->c[n+1] * (double)(n+1);
	return;
}

/* Revert the coefficients in-place: f(x) => f(1/x) * x^degree */
void double_poly_revert(double_poly_ptr f)
{
	unsigned int k;
	const unsigned int d = f->deg;
	for(k = 0; k <= d / 2; k++)
	{
		double tmp = f->c[k];
		f->c[k] = f->c[d - k];
		f->c[d - k] = tmp;
	}
	return;
}

static unsigned int recurse_roots(double_poly_ptr poly, double *roots,
	const unsigned int sign_changes, const double s)
{
	unsigned int new_sign_changes = 0;
	if(poly->deg <= 0)
	{
		/* A constant polynomial (degree 0 or -\infty) has no sign changes */
	}
	else if(poly->deg == 1)
	{
		/* A polynomial of degree 1 can have at most one sign change in (0, s),
		 * this happens iff poly(0) = poly[0] and poly(s) have different signs
		 */
		if(poly->c[0] * double_poly_eval(poly, s) < 0)
		{
			new_sign_changes = 1;
			roots[0] = -poly->c[0] / poly->c[1];
		}
	}
	else
	{
		/* invariant: sign_changes is the number of sign changes of the
		 * (k+1)-th derivative, with corresponding roots in roots[0]...
		 * roots[sign_changes-1], and roots[sign_changes] = s.
		 */
		double a = 0.0;
		double va = poly->c[0]; /* value of poly at x=0 */
		unsigned int l;

		for(l = 0; l <= sign_changes; l++)
		{
			/* b is a root of dg[k+1], or s, the end of the interval */
			const double b = (l < sign_changes) ? roots[l] : s;
			const double vb = double_poly_eval (poly, b);
			if(va * vb < 0) /* root in interval [va, vb] */
				roots[new_sign_changes++] = double_poly_dichotomy(poly, a, b, va, 20);
			a = b;
			va = vb;
		}
	}
	return new_sign_changes;
}

unsigned int double_poly_compute_roots(double *roots, double_poly_ptr poly, double s)
{
	const unsigned int d = poly->deg;
	double_poly_t *dg; /* derivatives of poly */
	unsigned int k, sign_changes;

	/* The roots of the zero polynomial are ill-defined. Bomb out */
	assert(d > 0 || poly->c[0] != 0.);

	/* Handle constant polynomials separately */
	if(d == 0)
		return 0; /* Constant non-zero poly -> no roots */

	dg = (double_poly_t *) malloc(d * sizeof(double_poly_t));

	dg[0]->deg = poly->deg;
	dg[0]->c = poly->c;

	for(k = 1; k < d; k++)
	{
		/* dg[k] is the k-th derivative, thus has degree d-k, i.e., d-k+1
		 * coefficients
		 */
		double_poly_init(dg[k], d - k);
		double_poly_derivative(dg[k], dg[k - 1]);
	}

	sign_changes = 0;
	for(k = d; k > 0; k--)
		sign_changes = recurse_roots(dg[k - 1], roots, sign_changes, s);

	for(k = 1; k < d; k++)
		double_poly_clear(dg[k]);
	free(dg);
	return sign_changes;
}

/* Put in fij[] the coefficients of f'(i) = F(a0*i+a1, b0*i+b1).
 * Assumes the coefficients of fij[] are initialized.
 */
void mpzpoly_homography(mpzpoly_t Fij, mpzpoly_t F, lat_t lat)
{
	int k, l;
	mpz_t *g; /* will contain the coefficients of (b0*i+b1)^l */
	mpz_t f0, tmp;
	mpz_t *f = F->c;
	int d = F->deg;
	mpz_t *fij = Fij->c;

	for(k = 0; k <= d; k++)
		mpz_set(fij[k], f[k]);

	Fij->deg = d;

	g = malloc((d + 1) * sizeof (mpz_t));
	for(k = 0; k <= d; k++)
		mpz_init(g[k]);
	mpz_init(f0);
	mpz_init(tmp);

	/* Let h(x) = quo(f(x), x), then F(x,y) = H(x,y)*x + f0*y^d, thus
	 * F(a0*i+a1, b0*i+b1) = H(a0*i+a1, b0*i+b1)*(a0*i+a1) + f0*(b0*i+b1)^d.
	 * We use that formula recursively.
	 */

	mpz_set_ui(g[0], 1); /* g = 1 */

	for(k = d - 1; k >= 0; k--)
	{
		/* invariant: we have already translated coefficients of degree > k,
		 * in f[k+1..d], and g = (b0*i+b1)^(d - (k+1)), with coefficients in
		 * g[0..d - (k+1)]:
		 *    f[k] <- a1*f[k+1]
		 *    ...
		 *    f[l] <- a0*f[l]+a1*f[l+1] for k < l < d
		 *    ...
		 *    f[d] <- a0*f[d]
		 */
		mpz_swap(f0, fij[k]); /* save the new constant coefficient */
		mpz_mul_si(fij[k], fij[k + 1], lat->a1);
		for(l = k + 1; l < d; l++)
		{
			mpz_mul_si(fij[l], fij[l], lat->a0);
			mpz_set_si(tmp, lat->a1);
			mpz_addmul(fij[l], fij[l + 1], tmp);
		}
		mpz_mul_si(fij[d], fij[d], lat->a0);

		/* now compute (b0*i+b1)^(d-k) from the previous (b0*i+b1)^(d-k-1):
		 *    g[d-k] = b0*g[d-k-1]
		 *    ...
		 *    g[l] = b1*g[l]+b0*g[l-1] for 0 < l < d-k
		 *    ...
		 *    g[0] = b1*g[0]
		 */
		mpz_mul_si(g[d - k], g[d - k - 1], lat->b0);
		for(l = d - k - 1; l > 0; l--)
		{
			mpz_mul_si(g[l], g[l], lat->b1);
			mpz_set_si(tmp, lat->b0);
			mpz_addmul(g[l], g[l-1], tmp);
		}
		mpz_mul_si(g[0], g[0], lat->b1);

		/* now g has degree d-k, and we add f0*g */
		for(l = k; l <= d; l++)
			mpz_addmul(fij[l], g[l - k], f0);
	}

	mpz_clear(tmp);
	mpz_clear(f0);
	for(k = 0; k <= d; k++)
		mpz_clear(g[k]);
	free(g);
	return;
}

/* return max |g(x)| for x in (0, s) where s can be negative,
   and g(x) = g[d]*x^d + ... + g[1]*x + g[0] */
static double get_maxnorm_aux(double_poly_t poly, double s)
{
	double_poly_t deriv;
	const int d = poly->deg;
	unsigned int k;

	if(d < 0)
	{
		return 0.;
	}
	else if(d == 0)
	{
		return poly->c[0];
	}

	double *roots = (double *) malloc(poly->deg * sizeof (double));

	/* Compute the derivative of polynomial */
	double_poly_init(deriv, d - 1);
	double_poly_derivative(deriv, poly);

	/* Look for extrema of the polynomial, i.e., for roots of the derivative */
	const unsigned int nr_roots = double_poly_compute_roots(roots, deriv, s);

	/* now abscissae of all extrema of poly are 0, roots[0], ..., roots[nr_roots-1], s */
	double gmax = fabs(poly->c[0]);
	for(k = 0; k <= nr_roots; k++)
	{
		double x = (k < nr_roots) ? roots[k] : s;
		double va = fabs(double_poly_eval (poly, x));
		if(va > gmax)
			gmax = va;
	}
	free(roots);
	double_poly_clear(deriv);
	return gmax;
}

/* Like get_maxnorm_aux(), but for interval [-s, s] */
static double get_maxnorm_aux_pm(double_poly_t poly, double s)
{
	double norm1 = get_maxnorm_aux(poly, s);
	double norm2 = get_maxnorm_aux(poly, -s);
	return (norm2 > norm1) ? norm2 : norm1;
}

/* returns the maximal norm of log2|F'(i,j)| for -I/2 <= i <= I/2, 0 <= j <= J.
 * Since F is homogeneous, we know M = max |F'(i,j)| = |f'(i/j) * j^d| is
 * attained on the border of the rectangle, i.e.:
 * (a) either on F'(I/2,j) for -J <= j <= J (right-hand-side border, and the
 *     mirrored image of the left-hand-side border);
 *     with F'(i,j) = rev(F)(j,i), we want the maximum of
 *     rev(F')(j, I/2) = rev(f')(j/(I/2)) * (I/2)^d for -J <= j <= J;
 *     assuming I = 2J, this is rev(f')(x) * J^d for -1 <= x <= 1.
 * (b) either on F'(i,J) for -I/2 <= i <= I/2 (top border)
 *     = f(i/J) * J^d; assuming I = 2J, f(x) * J^d for -1 <= x <= 1.
 * (d) or on F(i,0) for -I <= i <= I (lower border), but this maximum is f[d]*I^d,
 * and is attained in (a).
 */
static double get_maxnorm_alg(const double *coeff, const unsigned int d, int32_t I_half)
{
	const int debug = 0;
	unsigned int k;
	double norm, max_norm;

	double_poly_t poly;
	double_poly_init(poly, d);

	for(k = 0; k <= d; k++)
		poly->c[k] = coeff[k];

	/* (b1) determine the maximum of |F(x)| for -s <= x <= s */
	max_norm = get_maxnorm_aux_pm(poly, 1.);

	/* (a) determine the maximum of |g(y)| for -1 <= y <= 1, with g(y) = F(s,y) */
	double_poly_revert(poly);
	norm = get_maxnorm_aux_pm(poly, 1.);
	if(norm > max_norm)
		max_norm = norm;

	/* Both cases (a) and (b) want to multiply by J^d = (I/2)^d. Since J may
	 * not have been initialised yet, we use I.
	 */
	max_norm *= pow((double) I_half, (double) d);

	double_poly_clear(poly);

	return log2(max_norm);
}

/* Allocates:
 * cfg->fij[side]
 * cfg->fijd[side]
 */
void sieve_info_init_norm_data(gls_config_t cfg)
{
	int side, d;
	for(side = 0; side < 2; side++)
	{
      d = cfg->poly[side]->deg;
      mpzpoly_init(cfg->fij[side]);
      cfg->fijd[side] = (double *) malloc((d + 1) * sizeof(double));
    }
	return;
}

void sieve_info_clear_norm_data(gls_config_t cfg)
{
	int side;

	for(side = 0 ; side < 2 ; side++)
	{
        mpzpoly_clear(cfg->fij[side]);
        free(cfg->fijd[side]);
    }
	return;
}

#if 0
/* return largest possible J by simply bounding the Fij and Gij polynomials
 * using the absolute value of their coefficients
 */
static unsigned int sieve_info_update_norm_data_Jmax(gls_config_t cfg, int32_t I_half)
{
	int side, k;

	double Iover2 = (double) I_half;
	double Jmax = Iover2;

	for(side = 0; side < 2; side++)
	{
		double maxnorm = pow(2.0, cfg->logmax[side]), v, powIover2 = 1.;
		double_poly_t F;

		double_poly_init(F, cfg->poly[side]->deg);
		for(k = 0; k <= cfg->poly[side]->deg; k++)
		{
			/* reverse the coefficients since fij[k] goes with i^k but j^(d-k) */
			F->c[cfg->poly[side]->deg - k] = fabs(cfg->fijd[side][k]) * powIover2;
			powIover2 *= Iover2;
		}
		v = double_poly_eval(F, Jmax);
		if(v > maxnorm)
		{
			/* use dichotomy to determine largest Jmax */
			double a, b, c;
			a = 0.0;
			b = Jmax;
			while(trunc (a) != trunc (b))
			{
				c = (a + b) * 0.5;
				v = double_poly_eval(F, c);
				if(v < maxnorm)
					a = c;
				else
					b = c;
			}
			Jmax = trunc (a) + 1; /* +1 since we don't sieve for j = Jmax */
		}
		double_poly_clear(F);
	}

	return (unsigned int) Jmax;
}
#endif

/* this function initializes the scaling factors and report bounds on the
 * rational and algebraic sides
 */
void sieve_info_update_norm_data(gls_config_t cfg, lat_t lat, int q_side, mpz_t q, int32_t I_half)
{
	int i, side;
	double step, begin;
	double r, maxlog2;
	unsigned int inc;

	/* Update floating point version of both polynomials. They will be used in get_maxnorm_alg(). */
	for(side = 0; side < 2; side++)
	{
		mpzpoly_homography(cfg->fij[side], cfg->poly[side], lat);
		/* On the special-q side, divide all the coefficients of the transformed polynomial by q */
		if(q_side == side)
		{
			for(i = 0; i <= cfg->poly[side]->deg; i++)
			{
				assert(mpz_divisible_p(cfg->fij[side]->c[i], q));
				mpz_divexact(cfg->fij[side]->c[i], cfg->fij[side]->c[i], q);
			}
		}
		for(i = 0; i <= cfg->poly[side]->deg; i++)
			cfg->fijd[side][i] = mpz_get_d(cfg->fij[side]->c[i]);
	}

	/************************** rational side **********************************/

	/* Compute the maximum norm of the rational polynomial over the sieve
	 * region. The polynomial coefficient in fijd are already divided by q
	 * on the special-q side. */
	cfg->logmax[RPOLY_IDX] = get_maxnorm_alg(cfg->fijd[RPOLY_IDX], cfg->poly[RPOLY_IDX]->deg, I_half);

	/* we increase artificially 'logmax', to allow larger values of J */
	cfg->logmax[RPOLY_IDX] += 1.;

	/* we know that |G(a,b)| < 2^(rat->logmax) when si->ratq = 0,
	 * and |G(a,b)/q| < 2^(rat->logmax) when si->ratq <> 0
	 */

	maxlog2 = cfg->logmax[RPOLY_IDX];
	fprintf(output, "# Rat. side: log2(maxnorm)=%1.2f logbase=%1.6f",
			maxlog2, exp2 (maxlog2 / ((double) UCHAR_MAX - GUARD)));
	/* we want to map 0 <= x < maxlog2 to GUARD <= y < UCHAR_MAX,
	 * thus y = GUARD + x * (UCHAR_MAX-GUARD)/maxlog2
	 */
	cfg->scale[RPOLY_IDX] = ((double) UCHAR_MAX - GUARD) / maxlog2;
	step = 1. / cfg->scale[RPOLY_IDX];
	begin = -step * GUARD;

	for(inc = 0; inc < 257; begin += step)
		cfg->cexp2[RPOLY_IDX][inc++] = exp2(begin);
	/* we want to select relations with a cofactor of less than r bits on the
	 * rational side
	 */
	r = fmin(cfg->lambda[RPOLY_IDX] * (double) cfg->lpb[RPOLY_IDX], maxlog2 - GUARD / cfg->scale[RPOLY_IDX]);
	cfg->bound[RPOLY_IDX] = (u_int8_t) (r * cfg->scale[RPOLY_IDX] + GUARD);
	fprintf(output, " bound=%u\n", cfg->bound[RPOLY_IDX]);
	double max_rlambda = (maxlog2 - GUARD / cfg->scale[RPOLY_IDX]) / cfg->lpb[RPOLY_IDX];
	if(cfg->lambda[RPOLY_IDX] > max_rlambda)
	{
		fprintf(output, "# Warning, rlambda>%.1f does not make sense (capped to limit)\n", max_rlambda);
	}

	/************************** algebraic side *********************************/

	/* Compute the maximum norm of the algebraic polynomial over the sieve
	 * region. The polynomial coefficient in fijd are already divided by q
	 * on the special-q side.
	 */
	cfg->logmax[APOLY_IDX] = get_maxnorm_alg(cfg->fijd[APOLY_IDX], cfg->poly[APOLY_IDX]->deg, I_half);

	/* we know that |F(a,b)/q| < 2^(alg->logmax) when si->ratq = 0,
	 * and |F(a,b)| < 2^(alg->logmax) when si->ratq <> 0
	 */

	/* we increase artificially 'logmax', to allow larger values of J */
	cfg->logmax[APOLY_IDX] += 2.0;

	/* on the algebraic side, we want that the non-reports on the rational
	 * side, which are set to 255, remain larger than the report bound 'r',
	 * even if the algebraic norm is totally smooth. For this, we artificially
	 * increase by 'r' the maximal range.
	 * If lambda * lpb < logmax, which is the usual case, then non-reports on
	 * the rational side will start from logmax + lambda * lpb or more, and
	 * decrease to lambda * lpb or more, thus above the threshold as we want.
	 * If logmax < lambda * lpb, non-reports on the rational side will start
	 * from 2*logmax or more, and decrease to logmax or more.
	 */
	r = fmin(cfg->lambda[APOLY_IDX] * (double) cfg->lpb[APOLY_IDX], cfg->logmax[APOLY_IDX]);
	maxlog2 = cfg->logmax[APOLY_IDX] + r;


	fprintf(output, "# Alg. side: log2(maxnorm)=%1.2f logbase=%1.6f",
			cfg->logmax[APOLY_IDX], exp2(maxlog2 / ((double) UCHAR_MAX - GUARD)));
	/* we want to map 0 <= x < maxlog2 to GUARD <= y < UCHAR_MAX,
	 * thus y = GUARD + x * (UCHAR_MAX-GUARD)/maxlog2
	 */
	cfg->scale[APOLY_IDX] = ((double) UCHAR_MAX - GUARD) / maxlog2;
	step = 1. / cfg->scale[APOLY_IDX];
	begin = -step * GUARD;
	for(inc = 0; inc < 257; begin += step)
		cfg->cexp2[APOLY_IDX][inc++] = exp2(begin);
	/* we want to report relations with a remaining log2-norm after sieving of
	 * at most lambda * lpb, which corresponds in the y-range to
	 * y >= GUARD + lambda * lpb * scale
	 */
	cfg->bound[APOLY_IDX] = (unsigned char) (r * cfg->scale[APOLY_IDX] + GUARD);
	fprintf(output, " bound=%u\n", cfg->bound[APOLY_IDX]);

	double max_alambda = (cfg->logmax[APOLY_IDX]) / cfg->lpb[APOLY_IDX];
	if(cfg->lambda[APOLY_IDX] > max_alambda)
	{
		fprintf(output, "# Warning, alambda>%.1f does not make sense (capped to limit)\n", max_alambda);
	}
	return;
}
