/*
 * ul128.h
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */

#ifndef UL128_H_
#define UL128_H_
#include <gmp.h>

typedef u_int64_t ul256[4];

typedef u_int64_t ul128[2];

typedef struct mod128_s {
	ul128 n;
	u_int64_t np; /* np = -n^-1 (mod 2^64) */
	ul128 rsq; /* rsq = 2^256 (mod n) */
} mod128[1];

static void ul128_init(ul128 dst) { return; }
static void ul128_clear(ul128 dst) { return; }
static void ul256_init(ul128 dst) { return; }
static void ul256_clear(ul128 dst) { return; }

/* setters/getters */
static inline void ul128_set_ui(ul128 dst, u_int64_t src)
{
	dst[0] = src;
	dst[1] = 0;
	return;
}

static inline u_int64_t ul128_get_ui(ul128 src)
{
	return src[0];
}

static inline void ul128_set(ul128 dst, ul128 src)
{
	dst[0] = src[0];
	dst[1] = src[1];
	return;
}

static inline int ul128_cmp(ul128 src1, ul128 src2)
{
	if(src1[1] > src2[1])
		return 1;
	if(src1[1] < src2[1])
		return -1;
	if(src1[0] > src2[0])
		return 1;
	if(src1[0] < src2[0])
		return -1;
	return 0;
}

static inline int ul128_cmp_ui(ul128 src1, u_int64_t src2)
{
	if(src1[1])
		return 1;
	if(src1[0] > src2)
		return 1;
	if(src1[0] < src2)
		return -1;
	return 0;
}

/* mpz routines */
static inline void mpz_set_ul128(mpz_t dst, ul128 src)
{
	mpz_set_ui(dst, src[1]);
	mpz_mul_2exp(dst, dst, 64);
	mpz_add_ui(dst, dst, src[0]);
	return;
}

static inline void mpz_get_ul128(ul128 dst, mpz_t src)
{
	dst[0] = 0;
	dst[1] = 0;

	switch(src->_mp_size)
	{
	default:
		dst[1] = src->_mp_d[1];
		/* no break */
	case 1:
		dst[0] = src->_mp_d[0];
		break;
	case 0:
		break;
	}
	return;
}

static inline void mpz_set_ul256(mpz_t dst, ul256 src)
{
	mpz_set_ui(dst, src[3]);
	mpz_mul_2exp(dst, dst, 64);
	mpz_add_ui(dst, dst, src[2]);
	mpz_mul_2exp(dst, dst, 64);
	mpz_add_ui(dst, dst, src[1]);
	mpz_mul_2exp(dst, dst, 64);
	mpz_add_ui(dst, dst, src[0]);
	return;
}

static inline void mpz_get_ul256(ul256 dst, mpz_t src)
{
	dst[0] = 0;
	dst[1] = 0;
	dst[2] = 0;
	dst[3] = 0;
	if(src->_mp_size >= 1)
		dst[0] = src->_mp_d[0];
	if(src->_mp_size >= 2)
		dst[1] = src->_mp_d[1];
	if(src->_mp_size >= 3)
		dst[2] = src->_mp_d[2];
	if(src->_mp_size >= 4)
		dst[3] = src->_mp_d[3];
	return;
}

static inline void ul128_print(ul128 x)
{
	mpz_t tmp;

	mpz_init(tmp);
	mpz_set_ui(tmp, x[1]);
	mpz_mul_2exp(tmp, tmp, 64);
	mpz_add_ui(tmp, tmp, x[0]);
	gmp_printf("%Zd", tmp);
	mpz_clear(tmp);
	return;
}

static inline void ul128_setbit(ul128 dst, u_int32_t bit)
{
	if(bit >= 128)
		return;
	dst[bit / 64] |= (1ULL << (bit % 64));
	return;
}

static inline int ul128_tstbit(ul128 dst, u_int32_t bit)
{
	if(bit >= 128)
		return 0;
	return (dst[bit / 64] & (1ULL << (bit % 64)));
}

/* arithmetic routines */
static inline void ul128_add(ul128 dst, ul128 src1, ul128 src2)
{
	ul128 tr;
	ul128_set(tr, src1);
	__asm__(
		"add %2,%0\n\t"
		"adc %3,%1\n\t"
		: "+r" (tr[0]), "+&r" (tr[1])
		: "r" (src2[0]), "r" (src2[1])
		: "cc"
	);
	ul128_set(dst, tr);
	return;
}

static inline void ul128_sub(ul128 dst, ul128 src1, ul128 src2)
{
	ul128 tr;
	ul128_set(tr, src1);
	__asm__(
		"sub %2,%0\n\t"
		"sbb %3,%1\n\t"
		: "+r" (tr[0]), "+&r" (tr[1])
		: "r" (src2[0]), "r" (src2[1])
		: "cc"
	);
	ul128_set(dst, tr);
	return;
}

static inline void ul128_mul(ul128 dst, ul128 src1, ul128 src2)
{
	ul128 tr;
	__asm__(
		"mov %2,%%rax\n\t"
		"mulq %4\n\t"
		"mov %%rax,%0\n\t"
		"mov %%rdx,%1\n\t"
		"mov %3,%%rax\n\t"
		"mulq %4\n\t"
		"add %%rax,%1\n\t"
		"mov %2,%%rax\n\t"
		"mulq %5\n\t"
		"add %%rax,%1\n\t"
		: "+r" (tr[0]), "+&r" (tr[1])
		: "g" (src1[0]), "g" (src1[1]), "g" (src2[0]), "g" (src2[1])
		: "rax", "rdx", "cc"
	);
	ul128_set(dst, tr);
	return;
}

static inline void ul128_mul_ul256(ul256 dst, ul128 src1, ul128 src2)
{
	dst[0] = dst[1] = dst[2] = dst[3] = 0;
	__asm__(
		"mov %4,%%rax\n\t"
		"mulq %6\n\t"
		"mov %%rax,%0\n\t"
		"mov %%rdx,%1\n\t"
		"mov %5,%%rax\n\t"
		"mulq %6\n\t"
		"add %%rax,%1\n\t"
		"adc %%rdx,%2\n\t"
		"mov %4,%%rax\n\t"
		"mulq %7\n\t"
		"add %%rax,%1\n\t"
		"adc %%rdx,%2\n\t"
		"adc $0x0,%3\n\t"
		"mov %5,%%rax\n\t"
		"mulq %7\n\t"
		"add %%rax,%2\n\t"
		"adc %%rdx,%3\n\t"
		: "+r" (dst[0]), "+&r" (dst[1]), "+&r" (dst[2]), "+&r" (dst[3])
		: "g" (src1[0]), "g" (src1[1]), "g" (src2[0]), "g" (src2[1])
		: "rax", "rdx", "cc"
	);
	return;
}

static inline void ul128_divrem(ul128 q, ul128 r, ul128 a, ul128 n)
{
	/* TODO - replace */
	mpz_t b, c, d, e;

	mpz_init(b);
	mpz_init(c);
	mpz_init(d);
	mpz_init(e);
	mpz_set_ul128(d, a);
	mpz_set_ul128(e, n);
	mpz_fdiv_qr(b, c, d, e);
	mpz_get_ul128(q, b);
	mpz_get_ul128(r, c);
	mpz_clear(b);
	mpz_clear(c);
	mpz_clear(d);
	mpz_clear(e);
	return;
}

static inline void ul128_mod(ul128 r, ul128 a, ul128 n)
{
	ul128 t;
	ul128_divrem(t, r, a, n);
	return;
}

/* count trailing zeros */
static inline int ul128_bscan_fwd(ul128 x)
{
	u_int64_t ret;
	if(x[0])
	{
		__asm__("bsfq %1,%0\n\t"
			: "=r"(ret)
			: "r"(x[0]));
	}
	else
	{
		if(!x[1])
			return -1;
		__asm__("bsfq %1,%0\n\t"
			: "=r"(ret)
			: "r"(x[1]));
		ret += 64;
	}
	return ret;
}

/* count leading zeros */
static inline int ul128_bscan_rev(ul128 x)
{
	u_int64_t ret;
	if(x[1])
	{
		__asm__("bsrq %1,%0\n\t"
			: "=r"(ret)
			: "r"(x[1]));
		return 63 - ret;
	}
	else
	{
		if(!x[0])
			return -1;
		__asm__("bsrq %1,%0\n\t"
			: "=r"(ret)
			: "r"(x[0]));
		ret = 64 + 63 - ret;
	}
	return ret;
}

static inline void ul128_lshift(ul128 dst, ul128 src, int shift)
{
	dst[1] = (src[1] << shift) | (src[0] >> (64 - shift));
	dst[0] = src[0] << shift;
	return;
}

static inline void ul128_rshift(ul128 dst, ul128 src, int shift)
{
	dst[0] = (src[0] >> shift) | (src[1] << (64 - shift));
	dst[1] = src[1] >> shift;
	return;
}

extern void ul128_gcd(ul128 dst, ul128 x, ul128 y);

/* modular arithmetic routines */
static inline void ul128_modadd(ul128 dst, ul128 src1, ul128 src2, mod128 n)
{
	ul128 tr;

    tr[0] = src1[0];
    tr[1] = src1[1];
    __asm__(
    	"add %2,%0\n\t"
    	"adc %3,%1\n\t"
    	"mov %0,%%rax\n\t"
    	"mov %1,%%rdx\n\t"
    	"sub %4,%%rax\n\t"
    	"sbb %5,%%rdx\n\t"
    	"cmovnb %%rax,%0\n\t"
    	"cmovnb %%rdx,%1\n\t"
    	: "+r"(tr[0]), "+&r"(tr[1])
    	: "g"(src2[0]), "g" (src2[1]), "g"(n->n[0]), "g"(n->n[1])
    	: "rax", "rdx", "cc"
    );
    dst[0] = tr[0];
    dst[1] = tr[1];
    return;
}

static inline void ul128_modsub(ul128 dst, ul128 src1, ul128 src2, mod128 n)
{
    ul128 tr1, tr2;

    ul128_sub(tr1, src1, src2);
    ul128_add(tr2, tr1, n->n);
    if(ul128_cmp(tr2, n->n) > 0)
    	ul128_set(dst, tr1);
    else
    	ul128_set(dst, tr2);
    return;
}

static inline void ul128_modmul(ul128 dst, ul128 src1, ul128 src2, mod128 mod)
{
	u_int64_t T[3], M[3], m;

	/* T = src1[0] * src2 */
	__asm__(
		"# T = src1[0] * src2\n\t"
		"mov %9,%%rax\n\t"
		"mulq %7\n\t"
		"mov %%rax,%0\n\t"
		"mov %%rdx,%1\n\t"
		"mov %10,%%rax\n\t"
		"mulq %7\n\t"
		"xor %2,%2\n\t"
		"add %%rax,%1\n\t"
		"adc %%rdx,%2\n\t"
		"# m = T[0] * mod->np\n\t"
		"mov %0,%%rax\n\t"
		"mulq %13\n\t"
		"mov %%rax,%6\n\t"
		"# M = m * mod->n\n\t"
		"mulq %11\n\t"
		"mov %%rax,%3\n\t"
		"mov %%rdx,%4\n\t"
		"mov %6,%%rax\n\t"
		"mulq %12\n\t"
		"xor %5,%5\n\t"
		"add %%rax,%4\n\t"
		"adc %%rdx,%5\n\t"
		"# T = T + M\n\t"
		"add %3,%0\n\t"
		"adc %4,%1\n\t"
		"adc %5,%2\n\t"
		"# T = T / 2^64\n\t"
		"mov %1,%0\n\t"
		"mov %2,%1\n\t"
		"xor %2,%2\n\t"
		"# T = T + src1[1] * src2\n\t"
		"mov %9,%%rax\n\t"
		"mulq %8\n\t"
		"add %%rax,%0\n\t"
		"adc %%rdx,%1\n\t"
		"adc $0,%2\n\t"
		"mov %10,%%rax\n\t"
		"mulq %8\n\t"
		"add %%rax,%1\n\t"
		"adc %%rdx,%2\n\t"
		"# m = T[0] * mod->np\n\t"
		"mov %0,%%rax\n\t"
		"mulq %13\n\t"
		"mov %%rax,%6\n\t"
		"# M = m * mod->n\n\t"
		"mulq %11\n\t"
		"mov %%rax,%3\n\t"
		"mov %%rdx,%4\n\t"
		"mov %6,%%rax\n\t"
		"mulq %12\n\t"
		"xor %5,%5\n\t"
		"add %%rax,%4\n\t"
		"adc %%rdx,%5\n\t"
		"# T = T + M\n\t"
		"add %3,%0\n\t"
		"adc %4,%1\n\t"
		"adc %5,%2\n\t"
		"# T = T / 2^64\n\t"
		"mov %1,%0\n\t"
		"mov %2,%1\n\t"
		"xor %2,%2\n\t"
		: /* 0,1,2 */"+r"(T[0]), "+&r"(T[1]), "+&r"(T[2]),
		  /* 3,4,5 */ "+&r"(M[0]), "+&r"(M[1]), "+&r"(M[2]),
		  /* 6 */ "+&r"(m)
		: /* 7,8,9,10 */ "g"(src1[0]), "g"(src1[1]), "g"(src2[0]), "g"(src2[1]),
		  /* 11,12,13 */ "g"(mod->n[0]), "g"(mod->n[1]), "g"(mod->np)
		: "rax", "rdx", "cc"
	);
	dst[0] = T[0];
	dst[1] = T[1];
	if(ul128_cmp(dst, mod->n) >= 0)
		ul128_sub(dst, dst, mod->n);
	return;
}

extern void ul128_moddiv2(ul128 dst, ul128 src, mod128 n);
extern void ul128_moddiv3(ul128 dst, ul128 src, mod128 n);
extern void ul128_moddiv5(ul128 dst, ul128 src, mod128 n);
extern void ul128_moddiv7(ul128 dst, ul128 src, mod128 n);
extern void ul128_moddiv11(ul128 dst, ul128 src, mod128 n);
extern void ul128_moddiv13(ul128 dst, ul128 src, mod128 n);
extern int ul128_modinv(ul128 dst, ul128 src, ul128 n);

/* modulus init routines */
static inline void mod128_init(mod128 dst) { return; }
static inline void mod128_clear(mod128 dst) { return; }

static inline void mod128_set(mod128 dst, ul128 n)
{
	u_int64_t tmp;
	mpz_t gn, gtmp;

	ul128_set(dst->n, n);

	tmp = 2 + n[0];
	tmp = tmp * (2 + n[0] * tmp);
	tmp = tmp * (2 + n[0] * tmp);
	tmp = tmp * (2 + n[0] * tmp);
	tmp = tmp * (2 + n[0] * tmp);
	tmp = tmp * (2 + n[0] * tmp);
	dst->np = tmp;

	mpz_init(gn);
	mpz_init(gtmp);
	mpz_set_ul128(gn, n);
	mpz_setbit(gtmp, 256);
	mpz_mod(gtmp, gtmp, gn);
	mpz_get_ul128(dst->rsq, gtmp);
	mpz_clear(gtmp);
	mpz_clear(gn);
/*
	printf("N= ");
	ul128_print(dst->n);
	printf("\n");
	printf("np= %llu\n", dst->np);
	printf("rsq= ");
	ul128_print(dst->rsq);
	printf("\n");
*/
	return;
}

static inline void ul128_to_montgomery(ul128 dst, ul128 src, mod128 mod)
{
	ul128_modmul(dst, src, mod->rsq, mod);
	return;
}

static inline void ul128_from_montgomery(ul128 dst, ul128 src, mod128 mod)
{
	ul128 tmp;

	tmp[1] = 0;
	tmp[0] = 1;
	ul128_modmul(dst, src, tmp, mod);
	return;
}

/* randomization routines */
static inline void ul128_rand(ul128 dst)
{
	u_int32_t *ptr;

	ptr = (u_int32_t *) &dst[0];
	ptr[0] = random();
	ptr[1] = random();
	ptr[2] = random();
	ptr[3] = random();
	return;
}

static inline void ul128_modrand(ul128 dst, ul128 n)
{
	ul128 tmp;
	ul128_rand(dst);
	ul128_divrem(tmp, dst, dst, n);
	return;
}

#endif /* UL128_H_ */
