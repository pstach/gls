/*
 * makefb.c
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <gmp.h>
#include "gls_config.h"
#include "fb.h"

static inline u_int32_t mp_modsub_1(u_int32_t a, u_int32_t b, u_int32_t n)
{
	u_int32_t ret;

	ret = a - b;
	if(ret > a)
		ret += n;
	return ret;
}

static inline u_int32_t mp_modadd_1(u_int32_t a, u_int32_t b, u_int32_t n)
{
	return mp_modsub_1(a, n - b, n);
}

static inline u_int32_t mp_mod64(u_int64_t a, u_int32_t n)
{
	return (u_int32_t) (a % (u_int64_t) n);
}

static inline u_int32_t mp_modmul_1(u_int32_t a, u_int32_t b, u_int32_t n)
{
	u_int64_t tmp;

	tmp = (u_int64_t) a * (u_int64_t) b;
	return mp_mod64(tmp, n);
}

static inline u_int32_t mp_expo_1(u_int32_t a, u_int32_t b, u_int32_t n)
{
	u_int32_t res = 1;
	while(b)
	{
		if(b & 1)
			res = mp_modmul_1(res, a, n);
		a = mp_modmul_1(a, a, n);
		b = b >> 1;
	}
	return res;
}

static inline int32_t mp_legendre_1(u_int32_t a, u_int32_t p)
{
	u_int32_t x, y, tmp;
	int32_t out = 1;

	x = a;
	y = p;
	while(x)
	{
		while((x & 1) == 0)
		{
			x = x / 2;
			if((y & 7) == 3 || (y & 7) == 5)
				out = -out;
		}

		tmp = x;
		x = y;
		y = tmp;

		if((x & 3) == 3 && (y & 3) == 3)
			out = -out;

		x = x % y;
	}
	if(y == 1)
		return out;
	return 0;
}

static inline u_int32_t mp_modinv_1(u_int32_t a, u_int32_t p)
{
	u_int32_t ps1, ps2, dividend, divisor, rem, q, t;
	u_int32_t parity;

	q = 1; rem = a; dividend = p; divisor = a;
	ps1 = 1; ps2 = 0; parity = 0;

	while (divisor > 1)
	{
		rem = dividend - divisor;
		t = rem - divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t;
		if (rem >= divisor) {
			q = dividend / divisor;
			rem = dividend % divisor;
			q *= ps1;
		} } } } } } } } }

		q += ps2;
		parity = ~parity;
		dividend = divisor;
		divisor = rem;
		ps2 = ps1;
		ps1 = q;
	}

	if (parity == 0)
		return ps1;
	else
		return p - ps1;
}

static inline u_int32_t mp_modsqrt_1(u_int32_t a, u_int32_t p)
{

	u_int32_t a0 = a;

	if((p & 7) == 3 || (p & 7) == 7)
	{
		return mp_expo_1(a0, (p+1)/4, p);
	}
	else if((p & 7) == 5)
	{
		u_int32_t x, y;

		if(a0 >= p)
		a0 = a0 % p;
		x = mp_expo_1(a0, (p+3)/8, p);

		if(mp_modmul_1(x, x, p) == a0)
			return x;

		y = mp_expo_1(2, (p-1)/4, p);

		return mp_modmul_1(x, y, p);
	}
	else
	{
		u_int32_t d0, d1, a1, s, t, m;
		u_int32_t i;

		if(a0 == 1)
			return 1;

		for(d0 = 2; d0 < p; d0++)
		{
			if(mp_legendre_1(d0, p) != -1)
				continue;

			t = p - 1;
			s = 0;
			while(!(t & 1))
			{
				s++;
				t = t / 2;
			}

			a1 = mp_expo_1(a0, t, p);
			d1 = mp_expo_1(d0, t, p);

			for(i = 0, m = 0; i < s; i++)
			{
				u_int32_t ad;

				ad = mp_expo_1(d1, m, p);
				ad = mp_modmul_1(ad, a1, p);
				ad = mp_expo_1(ad, (u_int32_t)(1) << (s-1-i), p);
				if(ad == (p - 1))
					m += (1 << i);
			}

			a1 = mp_expo_1(a0, (t+1)/2, p);
			d1 = mp_expo_1(d1, m/2, p);
			return mp_modmul_1(a1, d1, p);
		}
	}

	printf("modsqrt_1 failed\n");
	exit(-1);
	return 0;
}

/* representation of polynomials with finite-field coefficients */
typedef struct {
	u_int32_t coef[2 * MAX_POLY_DEGREE + 1];
	u_int32_t degree;
} _poly_t;
typedef _poly_t poly_t[1];

static inline void poly_cp(poly_t dest, poly_t src)
{
	dest[0] = src[0];
	return;
}

static inline void poly_fix_degree(poly_t op)
{
	int32_t i = op->degree;

	while ((i > 0) && (op->coef[i] == 0))
		i--;
	op->degree = i;
	return;
}

static inline void poly_make_monic(poly_t res, poly_t a, u_int32_t p)
{
	u_int32_t i;
	u_int32_t d = a->degree;
	u_int32_t msw = a->coef[d];

	if (msw != 1)
	{
		msw = mp_modinv_1(msw, p);
		res->degree = d;
		res->coef[d] = 1;
		for (i = 0; i < d; i++)
			res->coef[i] = mp_modmul_1(msw, a->coef[i], p);
	}
	else
	{
		poly_cp(res, a);
	}
	return;
}

static void poly_mod(poly_t res, poly_t op, poly_t _mod, u_int32_t p)
{
	/* divide the polynomial 'op' by the polynomial '_mod'
	   and write the remainder to 'res'. All polynomial
	   coefficients are reduced modulo 'p' */

	int32_t i;
	u_int32_t msw;
	poly_t tmp, mod;

	if(_mod->degree == 0)
	{
		memset(res, 0, sizeof(res[0]));
		return;
	}
	poly_cp(tmp, op);
	poly_make_monic(mod, _mod, p);

	while(tmp->degree >= mod->degree)
	{
		/* tmp <-- tmp - msw * mod * x^{deg(tmp)- deg(mod)} */

		msw = tmp->coef[tmp->degree];

		tmp->coef[tmp->degree] = 0;
		for(i = mod->degree-1; i >= 0; i--)
		{
			u_int32_t c = mp_modmul_1(msw, mod->coef[i], p);
			u_int32_t j = tmp->degree - (mod->degree - i);
			tmp->coef[j] = mp_modsub_1(tmp->coef[j], c, p);
		}
		poly_fix_degree(tmp);
	}
	poly_cp(res, tmp);
	return;
}

static void poly_modmul(poly_t res, poly_t a, poly_t b, poly_t mod, u_int32_t p)
{
	u_int32_t i, j;
	poly_t prod;

	for (i = 0; i <= a->degree; i++)
		prod->coef[i] = mp_modmul_1(a->coef[i], b->coef[0], p);

	for (i = 1; i <= b->degree; i++)
	{
		for (j = 0; j < a->degree; j++)
		{
			u_int32_t c = mp_modmul_1(a->coef[j], b->coef[i], p);
			prod->coef[i+j] = mp_modadd_1(prod->coef[i+j], c, p);
		}
		prod->coef[i+j] = mp_modmul_1(a->coef[j], b->coef[i], p);
	}

	prod->degree = a->degree + b->degree;
	poly_fix_degree(prod);
	poly_mod(res, prod, mod, p);
	return;
}

/*------------------------------------------------------------------*/
	/* The following routines are highly performance-critical
	   for factor base generation.

	   These deal with modular multiplication and squaring
	   of polynomials. The operands and the modulus are
	   all packed contiguously into an array of uint32's,
	   to minimize pointer arithmetic */

#define NUM_POLY_COEFFS (MAX_POLY_DEGREE+1)
#define OP1(i) buf[i]
#define MOD(i) buf[NUM_POLY_COEFFS+i]

	/* The following implements word-based modular multiplication
	   and squaring of polynomials, with each coefficient reduced
	   modulo a prime p. Because there are no carries from one word
	   to the next, many simplifications are possible. To multiply
	   A[] and B[] mod N[], with each quantity a polynomial containing
	   d coefficients, the ordinary word-based method computes:

	   for (words B[i] starting at d and working backwards) {
	  	 accum[] = (accum[] << (one word)) mod N[]
	  	 accum[] = (accum[] + A[] * B[i]) mod N[]  (*)
	   }

	   In the finite field case, A[] * B[i] still has d words, and
	   the shifted up quantity has d+1 words. N[] is assumed monic
	   so the remainder operation amounts to subtracting N[] * (top
	   word of accumulator). This means that going into the multiply-
	   accumulate operation (*) we know the factor by which N[] is
	   multiplied, so the multiply-accumulate and the subtraction
	   of N[] can happen at the same time.

	   Another optimization is to allow the coefficents of the
	   accumulator to reach p^2 in size instead of p; this reduces the
	   number of 32-bit mod operations to only 2*d instead of d*(d+2).
	   There is a mod operation every time the top word of accum[]
	   is converted from mod p^2 to mod p, and there are d mods when
	   the final answer is copied from the accumulator */

/*------------------------------------------------------------------*/
static inline u_int32_t mul_mac(u_int32_t a, u_int32_t b, u_int32_t c, u_int32_t q, u_int32_t _mod, u_int32_t p, u_int64_t psq)
{
	/* For 32-bit a, b, q, c, _mod and 64-bit psq compute

	   (a * b + c - q * _mod) % p

	   psq is the square of p, and the 32-bit quantities
	   are all assumed less than p

	   Note that this is intended for a high-performance
	   processor with conditional move instructions. If the compiler
	   does not inline, or use these instructions, or realize that
	   many inputs are cached from previous calls then the overall
	   performance will be dismal. */

#if defined(GCC_ASM32A) && \
	!(defined(__GNUC__) && __GNUC__ < 3 ) && \
	defined(NDEBUG)

	u_int32_t ans;
	ASM_G(
	    "movl %2, %%eax             \n\t"
	    "mull %1                    \n\t"
	    "addl %3, %%eax             \n\t"
	    "adcl $0, %%edx             \n\t"
	    "movl %%eax, %%esi          \n\t"
	    "movl %%edx, %%edi          \n\t"
	    "movl %4, %%eax             \n\t"
	    "mull %5                    \n\t"
	    "subl %%eax, %%esi          \n\t"
	    "sbbl %%edx, %%edi          \n\t"
	    "movl $0, %%eax             \n\t"
	    "movl $0, %%edx             \n\t"
	    "cmovbl %7, %%eax           \n\t"
	    "cmovbl 4+%7, %%edx         \n\t"
	    "addl %%esi, %%eax          \n\t"
	    "adcl %%edi, %%edx          \n\t"
	    "divl %6                    \n\t"
	    "movl %%edx, %0             \n"
	    :"=g"(ans)
	    :"g"(a), "g"(b), "g"(c), "g"(_mod),
	     "g"(q), "g"(p), "g"(psq)
	    :"%eax", "%esi", "%edi", "%edx", "cc");

	return ans;

#elif defined(MSC_ASM32A)

	u_int32_t ans;
	ASM_M
	{
		mov	eax,b
		mul	a
		add	eax,c
		adc	edx,0
		mov	esi,eax
		mov	edi,edx
		mov	eax,_mod
		mul	q
		sub	esi,eax
		sbb	edi,edx
		mov	eax,0
		mov	edx,0
		lea	ecx,[psq]
		cmovb	eax,[ecx]
		cmovb	edx,[ecx+4]
		add	eax,esi
		adc	edx,edi
		div	p
		mov	ans,edx
	}

	return ans;

#else
	u_int64_t ans, tmp;

	/* for a,b,c < p the following will never exceed psq,
	   so no correction is needed */
	ans = (u_int64_t)a * (u_int64_t)b + (u_int64_t)c;

	tmp = (u_int64_t)q * (u_int64_t)_mod;
	ans = ans - tmp + (tmp > ans ? psq : 0);
	return mp_mod64(ans, p);
#endif
}

static inline u_int64_t sqr_mac(u_int32_t a, u_int32_t b, u_int64_t c, u_int32_t q, u_int32_t _mod, u_int64_t psq)
{

	/* For 32-bit a, b, q, _mod and 64-bit c, psq compute

	   (a * b + c - q * _mod) % psq

	   psq is the square of a prime, c is less than psq
	   and the 32-bit quantities are all assumed less
	   than the prime */

	u_int64_t ans;

#if defined(GCC_ASM32A) && \
	!(defined(__GNUC__) && __GNUC__ < 3 ) && \
	defined(NDEBUG)

	ASM_G(
	    "movl %1, %%eax               \n\t"
	    "mull %2                      \n\t"
	    "movl %6, %%esi               \n\t"
	    "movl 4+%6, %%edi             \n\t"
	    "subl %3, %%esi               \n\t"
	    "sbbl 4+%3, %%edi             \n\t"
	    "subl %%esi, %%eax            \n\t"
	    "sbbl %%edi, %%edx            \n\t"
	    "movl $0, %%esi               \n\t"
	    "movl $0, %%edi               \n\t"
	    "cmovbl %6, %%esi             \n\t"
	    "cmovbl 4+%6, %%edi           \n\t"
	    "addl %%eax, %%esi            \n\t"
	    "adcl %%edx, %%edi            \n\t"
	    "movl %5, %%eax               \n\t"
	    "mull %4                      \n\t"
	    "subl %%eax, %%esi            \n\t"
	    "sbbl %%edx, %%edi            \n\t"
	    "movl $0, %%eax               \n\t"
	    "movl $0, %%edx               \n\t"
	    "cmovbl %6, %%eax             \n\t"
	    "cmovbl 4+%6, %%edx           \n\t"
	    "addl %%esi, %%eax            \n\t"
	    "adcl %%edi, %%edx            \n\t"
	    "movl %%eax, %0               \n\t"
#if !defined(_ICL_WIN_)
	    "movl %%edx, 4+%0             \n"
#else   /* possible bug in Intel compiler ? */
	    "addl $4, %0                  \n\t"
	    "movl %%edx, %0               \n"
#endif
	    :"=g"(ans)
	    :"g"(a), "g"(b), "g"(c), "g"(q),
	     "g"(_mod), "g"(psq)
	    :"%eax", "%esi", "%edi", "%edx", "cc");

#elif defined(MSC_ASM32A)
	ASM_M
	{
		mov	eax, a
		mul	b
		lea	ecx,[psq]
		mov	esi,[ecx]
		mov	edi,[ecx+4]
		lea	ecx,[c]
		sub	esi,[ecx]
		sbb	edi,[ecx+4]
		sub	eax,esi
		sbb	edx,edi
		mov	esi,0
		mov	edi,0
		lea	ecx,[psq]
		cmovb	esi,[ecx]
		cmovb	edi,[ecx+4]
		add	esi,eax
		adc	edi,edx
		mov	eax,_mod
		mul	q
		sub	esi,eax
		sbb	edi,edx
		mov	eax,0
		mov	edx,0
		cmovb	eax,[ecx]
		cmovb	edx,[ecx+4]
		add	eax,esi
		adc	edx,edi
		lea	ecx,[ans]
		mov	[ecx],eax
		mov	[ecx+4],edx
	}
#else
	u_int64_t tmp;
	ans = (u_int64_t)a * (u_int64_t)b;
	tmp = psq - c;
	ans = ans - tmp + (tmp > ans ? psq : 0);
	tmp = (u_int64_t)q * (u_int64_t)_mod;
	ans = ans - tmp + (tmp > ans ? psq : 0);
#endif
	return ans;
}

static inline u_int64_t sqr_mac0(u_int32_t a, u_int32_t b, u_int32_t q, u_int32_t _mod, u_int64_t psq)
{
	/* For 32-bit a, b, q, _mod, compute

	   (a * b - q * _mod) % psq

	   psq is the square of a prime and the 32-bit quantities
	   are all assumed less than the prime */

	u_int64_t ans;

#if defined(GCC_ASM32A) && \
	!(defined(__GNUC__) && __GNUC__ < 3 ) && \
	defined(NDEBUG)

	ASM_G(
	    "movl %1, %%eax               \n\t"
	    "mull %2                      \n\t"
	    "movl %%eax, %%esi            \n\t"
	    "movl %%edx, %%edi            \n\t"
	    "movl %4, %%eax               \n\t"
	    "mull %3                      \n\t"
	    "subl %%eax, %%esi            \n\t"
	    "sbbl %%edx, %%edi            \n\t"
	    "movl $0, %%eax               \n\t"
	    "movl $0, %%edx               \n\t"
	    "cmovbl %5, %%eax             \n\t"
	    "cmovbl 4+%5, %%edx           \n\t"
	    "addl %%esi, %%eax            \n\t"
	    "adcl %%edi, %%edx            \n\t"
	    "movl %%eax, %0               \n\t"
	    "movl %%edx, 4+%0             \n"
	    :"=g"(ans)
	    :"g"(a), "g"(b), "g"(q), "g"(_mod), "g"(psq)
	    :"%eax", "%esi", "%edi", "%edx", "cc");

#elif defined(MSC_ASM32A)
	ASM_M
	{
		mov	eax,a
		mul	b
		mov	esi,eax
		mov	edi,edx
		mov	eax,_mod
		mul	q
		sub	esi,eax
		sbb	edi,edx
		mov	eax,0
		mov	edx,0
		lea	ecx,[psq]
		cmovb	eax,[ecx]
		cmovb	edx,[ecx+4]
		add	eax,esi
		adc	edx,edi
		lea	ecx,[ans]
		mov	[ecx],eax
		mov	[ecx+4],edx
	}
#else
	u_int64_t tmp;
	ans = (u_int64_t)a * (u_int64_t)b;
	tmp = (u_int64_t)q * (u_int64_t)_mod;
	ans = ans - tmp + (tmp > ans ? psq : 0);
#endif
	return ans;
}

/*------------------------------------------------------------------*/
static void poly_expo_modmul(u_int32_t *buf, u_int32_t dm, u_int32_t shift, u_int32_t p, u_int64_t psq)
{
	/* OP1 = OP1 * (x - shift) mod MOD
	   OP1 and MOD are of degree dm */

	u_int32_t q;
	u_int32_t zero = 0;

	q = OP1(dm-1);
	switch(dm-1) {
	case 7: OP1(7) = mul_mac(OP1(7), shift, OP1(6), q, MOD(7), p, psq);
	case 6: OP1(6) = mul_mac(OP1(6), shift, OP1(5), q, MOD(6), p, psq);
	case 5: OP1(5) = mul_mac(OP1(5), shift, OP1(4), q, MOD(5), p, psq);
	case 4: OP1(4) = mul_mac(OP1(4), shift, OP1(3), q, MOD(4), p, psq);
	case 3: OP1(3) = mul_mac(OP1(3), shift, OP1(2), q, MOD(3), p, psq);
	case 2: OP1(2) = mul_mac(OP1(2), shift, OP1(1), q, MOD(2), p, psq);
	case 1: OP1(1) = mul_mac(OP1(1), shift, OP1(0), q, MOD(1), p, psq);
	case 0: OP1(0) = mul_mac(OP1(0), shift,   zero, q, MOD(0), p, psq);
		break;
	}
}

/*------------------------------------------------------------------*/
static void poly_expo_square(u_int32_t *buf, u_int32_t dm, u_int32_t p, u_int64_t psq)
{
	/* OP1 = OP1 * OP1 mod MOD
	   OP1 and MOD are both of degree dm */

	u_int32_t i;
	u_int32_t q;
	u_int64_t acc[NUM_POLY_COEFFS];

	for (i = 0; i < dm; i++)
		acc[i] = (u_int64_t) (OP1(i)) * (u_int64_t)(OP1(dm-1));

	for (i = dm - 2; (int32_t)i >= 0; i--) {
		q = mp_mod64(acc[dm-1], p);
		switch(dm-1) {
  		case 7: acc[7] = sqr_mac(OP1(7), OP1(i), acc[6],
  							q, MOD(7), psq);
		case 6: acc[6] = sqr_mac(OP1(6), OP1(i), acc[5],
							q, MOD(6), psq);
		case 5: acc[5] = sqr_mac(OP1(5), OP1(i), acc[4],
							q, MOD(5), psq);
		case 4: acc[4] = sqr_mac(OP1(4), OP1(i), acc[3],
							q, MOD(4), psq);
		case 3: acc[3] = sqr_mac(OP1(3), OP1(i), acc[2],
							q, MOD(3), psq);
		case 2: acc[2] = sqr_mac(OP1(2), OP1(i), acc[1],
							q, MOD(2), psq);
		case 1: acc[1] = sqr_mac(OP1(1), OP1(i), acc[0],
							q, MOD(1), psq);
		case 0: acc[0] = sqr_mac0(OP1(0), OP1(i), q, MOD(0), psq);
			break;
		}
	}

	for (i = 0; i < dm; i++)
		OP1(i) = mp_mod64(acc[i], p);
}

/*------------------------------------------------------------------*/
static void poly_xpow(poly_t res, u_int32_t shift, u_int32_t n,
			poly_t mod, u_int32_t p) {

	/* Modular exponentiation of polynomials with
	   finite-field coefficients, i.e. res = (x-shift) ^ n % mod
	   with all polynomial coefficients reduced modulo p.
	   n is assumed nonzero */

	poly_t modnorm;
	u_int32_t msw;
	u_int32_t i, d;
	u_int32_t buf[2*NUM_POLY_COEFFS] = {0};
	u_int64_t psq;

	poly_make_monic(modnorm, mod, p);
	d = modnorm->degree;

	OP1(0) = shift;
	OP1(1) = 1;
	for (i = 0; i <= d; i++)
		MOD(i) = modnorm->coef[i];

	msw = 0x80000000;
	while (!(n & msw)) {
		msw >>= 1;
	}
	msw >>= 1;

	psq = (u_int64_t)p * (u_int64_t)p;

	/* use left-to-right binary exponentiation, not
	   the right-to-left variety. For factor base generation
	   the base always has degree less than modnorm, and the
	   left-to-right method preserves that, saving time during
	   modular multiplication */

	while (msw) {
		poly_expo_square(buf, d, p, psq);
		if (n & msw) {
			poly_expo_modmul(buf, d, shift, p, psq);
		}
		msw >>= 1;
	}

	res->degree = d;
	for (i = 0; i <= d; i++)
		res->coef[i] = OP1(i);
	poly_fix_degree(res);
}

/*------------------------------------------------------------------*/
static void poly_gcd(poly_t g_in, poly_t h_in, u_int32_t p) {

	poly_t g, h;

	/* make sure the first GCD iteration actually
	   performs useful work */

	if (g_in->degree > h_in->degree) {
		poly_cp(g, g_in);
		poly_cp(h, h_in);
	}
	else {
		poly_cp(h, g_in);
		poly_cp(g, h_in);
	}

	while ((h->degree > 0) || (h->coef[h->degree])) {
		poly_t r;
		poly_mod(r, g, h, p);
		poly_cp(g, h);
		poly_cp(h, r);
	}
	if (g->degree == 0)
		g->coef[0] = 1;
	poly_cp(g_in, g);
}

/*------------------------------------------------------------------*/
static void get_zeros_rec(u_int32_t *zeros, u_int32_t shift,
			u_int32_t *num_zeros, poly_t f, u_int32_t p) {

	/* get the zeros of a poly, f, that is known to split
	   completely over Z/pZ. Many thanks to Bob Silverman
	   for a neat implementation of Cantor-Zassenhaus splitting */

	poly_t g, xpow;
	u_int32_t degree1, degree2;

	/* base cases of the recursion: we can find the roots
	   of linear and quadratic polynomials immediately */

	if (f->degree == 1) {
		u_int32_t w = f->coef[1];
		if (w != 1) {
			w = mp_modinv_1(w, p);
			zeros[(*num_zeros)++] = mp_modmul_1(p - f->coef[0],w,p);
		}
		else {
			zeros[(*num_zeros)++] = (f->coef[0] == 0 ? 0 :
							p - f->coef[0]);
		}
		return;
	}
	else if (f->degree == 2) {

		/* if f is a quadratic polynomial, then it will
		   always have two distinct nonzero roots or else
		   we wouldn't have gotten to this point. The two
		   roots are the solution of a general quadratic
		   equation, mod p */

		u_int32_t d = mp_modmul_1(f->coef[0], f->coef[2], p);
		u_int32_t root1 = p - f->coef[1];
		u_int32_t root2 = root1;
		u_int32_t ainv = mp_modinv_1(
				mp_modadd_1(f->coef[2], f->coef[2], p),
				p);

		d = mp_modsub_1(mp_modmul_1(f->coef[1], f->coef[1], p),
				mp_modmul_1(4, d, p),
				p);
		d = mp_modsqrt_1(d, p);

		root1 = mp_modadd_1(root1, d, p);
		root2 = mp_modsub_1(root2, d, p);
		zeros[(*num_zeros)++] = mp_modmul_1(root1, ainv, p);
		zeros[(*num_zeros)++] = mp_modmul_1(root2, ainv, p);
		return;
	}

	/* For an increasing sequence of integers 's', compute
	   the polynomial gcd((x-s)^(p-1)/2 - 1, f). If the result is
	   not g = 1 or g = f, this is a nontrivial splitting
	   of f. References require choosing s randomly, but however
	   s is chosen there is a 50% chance that it will split f.
	   Since only 0 <= s < p is valid, we choose each s in turn;
	   choosing random s allows the possibility that the same
	   s gets chosen twice (mod p), which would waste time */

	while (shift < p) {
		poly_xpow(xpow, shift, (p-1)/2, f, p);

		poly_cp(g, xpow);
		g->coef[0] = mp_modsub_1(g->coef[0], 1, p);
		poly_fix_degree(g);

		poly_gcd(g, f, p);

		if (g->degree > 0)
			break;
		shift++;
	}

	/* f was split; repeat the splitting process on
	   the two halves of f. The linear factors of f are
	   either somewhere in x^((p-1)/2) - 1, in
	   x^((p-1)/2) + 1, or 'shift' itself is a linear
	   factor. Test each of these possibilities in turn.
	   In the first two cases, begin trying values of s
	   strictly greater than have been tried thus far */

	degree1 = g->degree;
	get_zeros_rec(zeros, shift + 1, num_zeros, g, p);

	poly_cp(g, xpow);
	g->coef[0] = mp_modadd_1(g->coef[0], 1, p);
	poly_fix_degree(g);
	poly_gcd(g, f, p);
	degree2 = g->degree;

	if (degree2 > 0)
		get_zeros_rec(zeros, shift + 1, num_zeros, g, p);

	if (degree1 + degree2 < f->degree)
		zeros[(*num_zeros)++] = (shift == 0 ? 0 : p - shift);
}

/*------------------------------------------------------------------*/
static void poly_reduce_mod_p(poly_t res, mpzpoly_t _f, u_int32_t p)
{
	u_int32_t i;

	res->degree = _f->deg;
	for (i = 0; i <= _f->deg; i++)
		res->coef[i] = mpz_fdiv_ui(_f->c[i], p);
	poly_fix_degree(res);
	return;
}

/*------------------------------------------------------------------*/
u_int32_t poly_get_zeros(u_int32_t *zeros, mpzpoly_t _f, u_int32_t p, u_int32_t count_only)
{
    /* Find all roots of multiplicity 1 for polynomial _f,
	   when the coefficients of _f are reduced mod p.
	   The leading coefficient of _f mod p is returned

	   Make count_only nonzero if only the number of roots
	   and not their identity matters; this is much faster */

	poly_t g, f;
	u_int32_t i, j, num_zeros;

	/* reduce the coefficients mod p */

	poly_reduce_mod_p(f, _f, p);

	/* bail out if the polynomial is zero */

	if (f->degree == 0)
		return 0;

	/* pull out roots of zero. We do this early to
	   avoid having to handle degree-1 polynomials
	   in later code */

	num_zeros = 0;
	if (f->coef[0] == 0) {
		for (i = 1; i <= f->degree; i++) {
			if (f->coef[i])
				break;
		}
		for (j = i; i <= f->degree; i++) {
			f->coef[i - j] = f->coef[i];
		}
		f->degree = i - j - 1;
		zeros[num_zeros++] = 0;
	}

	/* handle trivial cases */

	if (f->degree == 0) {
		return num_zeros;
	}
	else if (f->degree == 1) {
		u_int32_t w = f->coef[1];

		if (count_only)
			return num_zeros + 1;

		if (w != 1) {
			w = mp_modinv_1(w, p);
			zeros[num_zeros++] = mp_modmul_1(p - f->coef[0],
						w, p);
		}
		else {
			zeros[num_zeros++] = (f->coef[0] == 0 ?
						0 : p - f->coef[0]);
		}
		return num_zeros;
	}

	/* the rest of the algorithm assumes p is odd, which
	   will not work for p=2. Fortunately, in that case
	   there are only two possible roots, 0 and 1. The above
	   already tried 0, so try 1 here */

	if (p == 2) {
		u_int32_t parity = 0;
		for (i = 0; i <= f->degree; i++)
			parity ^= f->coef[i];
		if (parity == 0)
			zeros[num_zeros++] = 1;
		return num_zeros;
	}

	/* Compute g = gcd(f, x^(p-1) - 1). The result is
	   a polynomial that is the product of all the linear
	   factors of f. A given factor only occurs once in
	   this polynomial */

	poly_xpow(g, 0, p-1, f, p);
	g->coef[0] = mp_modsub_1(g->coef[0], 1, p);
	poly_fix_degree(g);
	poly_gcd(g, f, p);

	/* no linear factors, no service */

	if (g->degree < 1 || count_only)
		return num_zeros + g->degree;

	/* isolate the linear factors */

	get_zeros_rec(zeros, 0, &num_zeros, g, p);
	return num_zeros;
}

/*------------------------------------------------------------------*/
u_int32_t poly_get_zeros_and_mult(u_int32_t *zeros, u_int32_t *mult, mpzpoly_t _f, u_int32_t p)
{
	u_int32_t i;
	u_int32_t num_roots;
	poly_t f;

	num_roots = poly_get_zeros(zeros, _f, p, 0);
	if (num_roots == 0)
		return num_roots;

	poly_reduce_mod_p(f, _f, p);
	for (i = 0; i < num_roots; i++)
		mult[i] = 0;
	if (f->degree == num_roots)
		return num_roots;

	for (i = 0; i < num_roots; i++) {

		poly_t g, r;
		u_int32_t root = zeros[i];

		g->degree = 2;
		g->coef[0] = mp_modmul_1(root, root, p);
		g->coef[1] = p - mp_modadd_1(root, root, p);
		g->coef[2] = 1;

		poly_mod(r, f, g, p);
		if (r->degree == 0)
			mult[i] = 1;
	}
	return num_roots;
}

/*------------------------------------------------------------------*/
static void poly_xpow_pd(poly_t res, u_int32_t p, u_int32_t d, poly_t f) {

	/* compute x^(p^d) mod f */

	u_int32_t i;
	mpz_t exponent;
	poly_t x;

	mpz_init_set_ui(exponent, p);
	mpz_pow_ui(exponent, exponent, d);

	x->degree = 1;
	x->coef[0] = 0;
	x->coef[1] = 1;

	poly_cp(res, x);
	for (i = mpz_sizeinbase(exponent, 2) - 2; (int32_t)i >= 0; i--) {
		poly_modmul(res, res, res, f, p);
		if (mpz_tstbit(exponent, i))
			poly_modmul(res, res, x, f, p);
	}

	mpz_clear(exponent);
}

u_int32_t is_irreducible(mpzpoly_t poly, u_int32_t p)
{

	/* this uses Proposition 3.4.4 of H. Cohen, "A Course
	   in Computational Algebraic Number Theory". The tests
	   below are much simpler than trying to factor 'poly' */

	u_int32_t i;
	poly_t f, tmp;

	poly_reduce_mod_p(f, poly, p);
	poly_make_monic(f, f, p);

	/* in practice, the degree of f will be 8 or less,
	   and we want to compute GCDs for all prime numbers
	   that divide the degree. For this limited range
	   the loop below avoids duplicated code */

	for (i = 2; i < f->degree; i++) {
		if (f->degree % i)
			continue;

		/* for degree d, compute x^(p^(d/i)) - x */

		poly_xpow_pd(tmp, p, f->degree / i, f);
		if (tmp->degree == 0) {
			tmp->degree = 1;
			tmp->coef[1] = p - 1;
		}
		else {
			tmp->coef[1] = mp_modsub_1(tmp->coef[1],
						(u_int32_t)1, p);
			poly_fix_degree(tmp);
		}

		/* this must be relatively prime to f */

		poly_gcd(tmp, f, p);
		if (tmp->degree > 0 || tmp->coef[0] != 1) {
			return 0;
		}
	}

	/* final test: x^(p^d) mod f must equal x */

	poly_xpow_pd(tmp, p, f->degree, f);
	if (tmp->degree == 1 && tmp->coef[0] == 0 && tmp->coef[1] == 1)
		return 1;
	return 0;
}

int ui_cmp(const void *p1, const void *p2)
{
	return (*(u_int32_t *) p1 - *(u_int32_t *) p2);
}

int main(int argc, char *argv[])
{
	int poly_idx;
	gls_config_t cfg;
	fb_t fb;
	u_int32_t i, n_roots;
	u_int64_t fb_idx, fb_alloc;
	mpz_t P, roots[MAX_POLY_DEGREE];
	u_int32_t roots_ui[MAX_POLY_DEGREE];
	char fname[4096];
	off_t align;
	int fd;

	if(argc < 2)
	{
		fprintf(stderr, "usage: %s <polyconfig>\n", argv[0]);
		return 1;
	}

	gls_config_init(cfg);
	if(polyfile_read(cfg, argv[1]) < 0)
		return 1;

	mpz_init(P);
	for(i = 0; i < sizeof(roots) / sizeof(roots[0]); i++)
		mpz_init(roots[i]);

	for(poly_idx = 0; poly_idx < POLY_CNT; poly_idx++)
	{
		fb_init(fb);

		fb_idx = 0;
		fb_alloc = 0;
		for(mpz_set_ui(P, 2); mpz_cmp_ui(P, MAX_SMALL_PRIME) < 0 && mpz_cmp_ui(P, cfg->lim[poly_idx]) < 0; mpz_nextprime(P, P))
		{
			n_roots = poly_get_zeros(roots_ui, cfg->poly[poly_idx], mpz_get_ui(P), 0);
			if(!n_roots)
				continue;
			qsort(roots_ui, n_roots, sizeof(roots_ui[0]), ui_cmp);
//#define TEST_ROOTFINDER 1
#ifdef TEST_ROOTFINDER
			for(i = 0; i < n_roots; i++)
			{
				mpz_t tmp1, tmp2;
				mpz_init_set_ui(tmp1, roots_ui[i]);
				mpz_init(tmp2);
				mpzpoly_eval_mod(tmp2, cfg->poly[poly_idx], tmp1, P);
				if(mpz_cmp_ui(tmp2, 0) != 0)
				{
					printf("BAD ROOT (%u,%u)\n", roots_ui[i], mpz_get_ui(P));
					exit(-1);
				}
				mpz_clear(tmp1);
				mpz_clear(tmp2);
			}
#endif

			fb->n_small += n_roots;
			if(fb->n_small > fb_alloc)
			{
				fb_alloc += 4096;
				fb->r_small = (u_int32_t *) realloc(fb->r_small, sizeof(u_int32_t) * fb_alloc);
				fb->p_small = (u_int32_t *) realloc(fb->p_small, sizeof(u_int32_t) * fb_alloc);

				if(!fb->r_small || !fb->p_small)
				{
					perror("realloc");
					exit(1);
				}
			}
			for(i = 0; i < n_roots; i++)
			{
				fb->r_small[fb_idx] = roots_ui[i];
				fb->p_small[fb_idx] = mpz_get_ui(P);
				fb_idx++;
			}
		}

		fb_idx = 0;
		fb_alloc = 0;
		for(; mpz_cmp_ui(P, cfg->lim[poly_idx]) < 0; mpz_nextprime(P, P))
		{
			n_roots = mpzpoly_get_roots(roots, cfg->poly[poly_idx], P);
			if(!n_roots)
				continue;

			fb->n_large += n_roots;
			if(fb->n_large > fb_alloc)
			{
				fb_alloc += 4096;
				fb->r_large = (u_int64_t *) realloc(fb->r_large, sizeof(u_int64_t) * fb_alloc);
				fb->p_large = (u_int64_t *) realloc(fb->p_large, sizeof(u_int64_t) * fb_alloc);

				if(!fb->r_large || !fb->p_large)
				{
					perror("realloc");
					exit(1);
				}
			}
			for(i = 0; i < n_roots; i++)
			{
				fb->r_large[fb_idx] = mpz_get_ui(roots[i]);
				fb->p_large[fb_idx] = mpz_get_ui(P);
				fb_idx++;
			}
		}

		memset(fname, 0, sizeof(fname));
		snprintf(fname, sizeof(fname) - 1, "%s.fb.%u", argv[1], poly_idx);

		if(fb_filesave(fb, fname) < 0)
			return 1;
//#define TEST_SAVELOAD 1
#ifdef TEST_SAVELOAD
		{
			fb_t fb2;

			fb_init(fb2);
			if(fb_fileread(fb2, fname) < 0)
				return 1;

			if(fb->n_small != fb2->n_small || fb->n_large != fb2->n_large)
			{
				printf("COUNT MISMATCH\n");
				return 1;
			}
			if(memcmp(fb->p_small, fb2->p_small, sizeof(u_int32_t) * fb->n_small) != 0)
			{
				printf("SMALL MISMATCH\n");
				return 1;
			}
			if(memcmp(fb->p_large, fb2->p_large, sizeof(u_int64_t) * fb->n_large) != 0)
			{
				printf("LARGE MISMATCH\n");
				return 1;
			}
			fb_clear(fb2);
		}
#endif
		fb_clear(fb);
	}
	return 0;
}
