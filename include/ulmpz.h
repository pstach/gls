/*
 * ulmpz.h
 *
 *  Created on: Aug 14, 2013
 *      Author: pstach
 */

#ifndef ULMPZ_H_
#define ULMPZ_H_
#include <gmp.h>

typedef struct modmpz_s {
	mpz_t n;
	u_int64_t np; /* np = -n^-1 (mod 2^64) */
	mpz_t rsq; /* rsq = R^2 (mod n) */
	u_int32_t r;
} modmpz_t[1];

static inline void mpz_printer(mpz_t x)
{
	gmp_printf("%Zd", x);
	return;
}

/* count trailing zeros */
static inline int mpz_bscan_fwd(mpz_t x)
{
	return mpz_scan1(x, 0);
}

/* count leading zeros */
static inline int mpz_bscan_rev(mpz_t x)
{
	int ret;

	for(ret = mpz_sizeinbase(x, 2); ret >= 0; ret--)
	{
		if(mpz_tstbit(x, ret))
			break;
	}
	return ret;
}

/* modular arithmetic routines */
static inline void mpz_modadd(mpz_t dst, mpz_t src1, mpz_t src2, modmpz_t n)
{
	mpz_add(dst, src1, src2);
	if(mpz_cmp(dst, n->n) >= 0)
		mpz_sub(dst, dst, n->n);
    return;
}

static inline void mpz_modsub(mpz_t dst, mpz_t src1, mpz_t src2, modmpz_t n)
{
	mpz_sub(dst, src1, src2);
	if(mpz_cmp_ui(dst, 0) < 0)
		mpz_add(dst, dst, n->n);
	return;
}

static inline void mpz_modmul(mpz_t dst, mpz_t src1, mpz_t src2, modmpz_t mod)
{
	int i;
	u_int64_t m;
	mpz_t T;

	if(!src1->_mp_size || !src2->_mp_size)
	{
		mpz_set_ui(dst, 0);
		return;
	}

	mpz_init(T);

	mpz_mul_ui(T, src1, src2->_mp_d[0]);
	/* m = T[0] * np */
	m = T->_mp_d[0] * mod->np;
	/* T = T + m * N */
	mpz_addmul_ui(T, mod->n, m);
	/* T = T / 2^64 */
	mpz_div_2exp(T, T, sizeof(m) * 8);

	for(i = 1; i < mod->n->_mp_size; i++)
	{
		/* T = T + src1 * src2[i] */
		if(i < src2->_mp_size)
			mpz_addmul_ui(T, src1, src2->_mp_d[i]);
		/* m = T[0] * np */
		m = T->_mp_d[0] * mod->np;
		/* T = T + m * N */
		mpz_addmul_ui(T, mod->n, m);
		/* T = T / 2^64 */
		mpz_div_2exp(T, T, sizeof(m) * 8);
	}

	mpz_sub(dst, T, mod->n);
	if(dst->_mp_size < 0)
		mpz_set(dst, T);
	mpz_clear(T);
	return;
}

extern void mpz_moddiv2(mpz_t dst, mpz_t src, modmpz_t n);
extern void mpz_moddiv3(mpz_t dst, mpz_t src, modmpz_t n);
extern void mpz_moddiv5(mpz_t dst, mpz_t src, modmpz_t n);
extern void mpz_moddiv7(mpz_t dst, mpz_t src, modmpz_t n);
extern void mpz_moddiv11(mpz_t dst, mpz_t src, modmpz_t n);
extern void mpz_moddiv13(mpz_t dst, mpz_t src, modmpz_t n);

static inline void mpzmod_init(modmpz_t dst)
{
	mpz_init(dst->n);
	mpz_init(dst->rsq);
	return;
}
static inline void mpzmod_clear(modmpz_t dst)
{
	mpz_clear(dst->n);
	mpz_clear(dst->rsq);
	return;
}

static inline void mpzmod_set(modmpz_t dst, mpz_t n)
{
	u_int64_t tmp;

	mpz_set(dst->n, n);
	dst->r = (mpz_sizeinbase(n, 2) + 63) & ~63;

	tmp = 2 + mpz_get_ui(n);
	tmp = tmp * (2 + mpz_get_ui(n) * tmp);
	tmp = tmp * (2 + mpz_get_ui(n) * tmp);
	tmp = tmp * (2 + mpz_get_ui(n) * tmp);
	tmp = tmp * (2 + mpz_get_ui(n) * tmp);
	tmp = tmp * (2 + mpz_get_ui(n) * tmp);
	dst->np = tmp;

	mpz_set_ui(dst->rsq, 0);
	mpz_setbit(dst->rsq, (dst->r * 2));
	//printf("r: %d rsq: %d n: %d\n", dst->r, mpz_sizeinbase(dst->rsq, 2), mpz_sizeinbase(n, 2));
	mpz_mod(dst->rsq, dst->rsq, n);
	return;
}

static inline void mpz_to_montgomery(mpz_t dst, mpz_t src, modmpz_t n)
{
	mpz_modmul(dst, src, n->rsq, n);
	return;
}

static inline void mpz_from_montgomery(mpz_t dst, mpz_t src, modmpz_t n)
{
	mpz_t tmp;

	mpz_init_set_ui(tmp, 1);
	mpz_modmul(dst, src, tmp, n);
	mpz_clear(tmp);
	return;
}

extern void mpz_sqrtm(mpz_t q, mpz_t n, mpz_t p);

/* randomization routines */
static inline void mpz_rand(mpz_t dst)
{
	u_int32_t i, limb;
	u_int32_t *ptr;

	mpz_set_ui(dst, 0);
	limb = (random() % 16) + 2;
	for(i = 0; i < limb; i++)
	{
		mpz_mul_2exp(dst, dst, 32);
		mpz_add_ui(dst, dst, random());
	}
	return;
}

static inline void mpz_modrand(mpz_t dst, mpz_t n)
{
	mpz_t tmp;

	mpz_init(tmp);
	mpz_rand(tmp);
	mpz_mod(dst, tmp, n);
	mpz_clear(tmp);
	return;
}

#endif /* ULMPZ_H_ */
