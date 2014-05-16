/*
 * ul64.h
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */

#ifndef UL64_H_
#define UL64_H_
#include <gmp.h>
#include "ul128.h"

typedef u_int64_t ul64[1];

typedef struct mod64_s {
	ul64 n;
	u_int64_t np;
	ul64 rsq;
} mod64[1];

static void ul64_init(ul64 dst) { return; }
static void ul64_clear(ul64 dst) { return; }

/* setters/getters */
static inline void ul64_set_ui(ul64 dst, u_int64_t src)
{
	dst[0] = src;
	return;
}

static inline u_int64_t ul64_get_ui(ul64 src)
{
	return src[0];
}

static inline void ul64_set(ul64 dst, ul64 src)
{
	dst[0] = src[0];
	return;
}

static inline int ul64_cmp(ul64 src1, ul64 src2)
{
	if(src1[0] > src2[0])
		return 1;
	if(src1[0] < src2[0])
		return -1;
	return 0;
}

static inline int ul64_cmp_ui(ul64 src1, u_int64_t src2)
{
	if(src1[0] > src2)
		return 1;
	if(src1[0] < src2)
		return -1;
	return 0;
}

/* mpz routines */
static inline void mpz_set_ul64(mpz_t dst, ul64 src)
{
	mpz_set_ui(dst, src[0]);
	return;
}

static inline void mpz_get_ul64(ul64 dst, mpz_t src)
{
	dst[0] = mpz_get_ui(src);
	return;
}

static inline void ul64_print(ul64 x)
{
	printf("%llu", x[0]);
	return;
}

static inline void ul64_setbit(ul64 dst, u_int32_t bit)
{
	dst[0] |= (1ULL << bit);
	return;
}

static inline int ul64_tstbit(ul64 dst, u_int32_t bit)
{
	return (dst[0] & (1ULL << bit));
}

/* arithmetic routines */
static inline void ul64_add(ul64 dst, ul64 src1, ul64 src2)
{
	dst[0] = src1[0] + src2[0];
	return;
}

static inline void ul64_sub(ul64 dst, ul64 src1, ul64 src2)
{
	dst[0] = src1[0] - src2[0];
	return;
}

static inline void ul64_mul(ul64 dst, ul64 src1, ul64 src2)
{
	dst[0] = src1[0] * src2[0];
	return;
}

static inline void ul64_mul_ul128(ul128 dst, ul64 src1, ul64 src2)
{
	__asm__(
		"mul %3\n\t"
		: "=a" (dst[0]), "=d" (dst[1])
		: "a" (src1[0]), "r" (src2[0])
	);
	return;
}

static inline void ul64_divrem(ul64 q, ul64 r, ul64 a, ul64 n)
{
	__asm__(
		"divq %4\n\t"
		: "=a" (q[0]), "=d" (r[0])
		: "a" (a[0]), "d"(0), "rm" (n[0])
		: "cc"
	);
	return;
}

static inline void ul64_mod(ul64 r, ul64 a, ul64 n)
{
	ul64 t;

	ul64_divrem(t, r, a, n);
	return;
}

static inline int ul64_bscan_fwd(ul64 x)
{
	int ret;
	if(!x[0])
		return -1;
	__asm__("bsf %1,%0\n\t"
		: "=r"(ret)
		: "r"(x[0]));
	return ret;
}

static inline int ul64_bscan_rev(ul64 x)
{
	int ret;
	if(!x[0])
		return -1;
	__asm__("bsr %1,%0\n\t"
		: "=r"(ret)
		: "r"(x[0]));
	return ret;
}

static inline void ul64_lshift(ul64 dst, ul64 src, int shift)
{
	dst[0] = src[0] << shift;
	return;
}

static inline void ul64_rshift(ul64 dst, ul64 src, int shift)
{
	dst[0] = src[0] >> shift;
	return;
}

extern void ul64_gcd(ul64 dst, ul64 x, ul64 y);

/* modular arithmetic routines */
static inline void ul64_modadd(ul64 dst, ul64 src1, ul64 src2, mod64 n)
{
    u_int64_t t = src1[0] - n->n[0], tr = src1[0] + src2[0];

    __asm__(
    	"add %2,%1\n\t"
    	"cmovc %1,%0\n\t"
    	: "+r"(tr), "+&r"(t)
    	: "g"(src2[0]) : "cc"
    );
    dst[0] = tr;
    return;
}

static inline void ul64_modsub(ul64 dst, ul64 src1, ul64 src2, mod64 n)
{
    u_int64_t tr, t = src1[0];
    __asm__(
    	"sub %2,%1\n\t"        /* t -= b */
    	"lea (%1,%3,1),%0\n\t" /* tr = t + m */
    	"cmovnc %1,%0\n\t"      /* if (a >= b) tr = t */
    	: "=&r"(tr), "+&r"(t)
    	: "g"(src2[0]), "r"(n->n[0]) : "cc"
    );
    dst[0] = tr;
    return;
}

static inline void ul64_modmul(ul64 dst, ul64 src1, ul64 src2, mod64 n)
{
    u_int64_t m, t;

    /* calculate src1*src2*(2^64)-1 (mod n) */
    __asm__(
    		"# ul64_modmul\n\t"
    		"mov %2,%%rax\n\t"
            "mulq %1\n\t" // rdx:rax = a * b
            "mov %%rax,%%r10\n\t" // r10 = low(a*b)
            "mov %%rdx,%%r11\n\t" // r11 = high(a*b)
            "mulq %3\n\t" // rdx:rax = low(a*b) * mod->np
            "mulq %4\n\t" // rdx:rax = low(a*b) * mod->np * mod->n
            "add %%r10,%%rax\n\t"
            "adc %%r11,%%rdx\n\t"
            "mov %%rdx,%%rax\n\t"
            "sub %4,%%rax\n\t"
            "cmovnc %%rax,%%rdx\n\t"
            : /* 0 */ "+d"(t)
            : /* 1,2 */ "r"(src1[0]), "g"(src2[0]), /* 3,4 */ "r"(n->np), "r"(n->n[0])
            : "rax", "r10", "r11"
    );
    dst[0] = t;
	return;
}

extern void ul64_moddiv2(ul64 dst, ul64 src, mod64 n);
extern void ul64_moddiv3(ul64 dst, ul64 src, mod64 n);
extern void ul64_moddiv5(ul64 dst, ul64 src, mod64 n);
extern void ul64_moddiv7(ul64 dst, ul64 src, mod64 n);
extern void ul64_moddiv11(ul64 dst, ul64 src, mod64 n);
extern void ul64_moddiv13(ul64 dst, ul64 src, mod64 n);
extern int ul64_modinv(ul64 dst, ul64 src, ul64 n);

/* modulus init routines */
static inline void mod64_init(mod64 dst) { return; }
static inline void mod64_clear(mod64 dst) { return; }

static inline void mod64_set(mod64 dst, ul64 n)
{
	u_int64_t tmp;

	ul64_set(dst->n, n);

	tmp = 2 + n[0];
	tmp = tmp * (2 + n[0] * tmp);
	tmp = tmp * (2 + n[0] * tmp);
	tmp = tmp * (2 + n[0] * tmp);
	tmp = tmp * (2 + n[0] * tmp);
	tmp = tmp * (2 + n[0] * tmp);
	dst->np = tmp;

    /* calculate (2^64)^2 (mod n) */
	__asm__(
		"# mod64_set calc rsq\n\t"
		"mov $1,%%rdx\n\t"
		"xor %%rax,%%rax\n\t"
		"divq %1\n\t"
		"xor %%rax,%%rax\n\t"
		"divq %1\n\t"
		: "+d"(dst->rsq[0])
		: "r"(dst->n[0])
		: "rax"
    );
	return;
}

static inline void ul64_to_montgomery(ul64 dst, ul64 src, mod64 mod)
{
	ul64_modmul(dst, src, mod->rsq, mod);
	return;
}

static inline void ul64_from_montgomery(ul64 dst, ul64 src, mod64 mod)
{
	ul64 tmp;

	tmp[0] = 1;
	ul64_modmul(dst, src, tmp, mod);
	return;
}

/* randomization routines */
static inline void ul64_rand(ul64 dst)
{
	u_int32_t *ptr;

	ptr = (u_int32_t *) &dst[0];
	ptr[0] = random();
	ptr[1] = random();
	return;
}

static inline void ul64_modrand(ul64 dst, ul64 n)
{
	ul64 tmp;
	ul64_rand(dst);
	ul64_divrem(tmp, dst, dst, n);
	return;
}

#endif /* UL64_H_ */
