/*
 * mpzpoly.h
 *
 *  Created on: Sep 24, 2013
 *      Author: pstach
 */

#ifndef MPZPOLY_H_
#define MPZPOLY_H_
#include <gmp.h>

#define MAX_POLY_DEGREE 8

typedef struct mpzpoly_s {
	u_int32_t deg;
	mpz_t c[MAX_POLY_DEGREE + 1];
} mpzpoly_t[1];


static inline void mpzpoly_init(mpzpoly_t p)
{
	int i;

	for(i = 0; i < sizeof(p->c) / sizeof(p->c[0]); i++)
		mpz_init(p->c[i]);
	p->deg = 0;
	return;
}

static inline void mpzpoly_clear(mpzpoly_t p)
{
	int i;

	for(i = 0; i < sizeof(p->c) / sizeof(p->c[0]); i++)
		mpz_clear(p->c[i]);
	return;
}

static inline void mpzpoly_print(mpzpoly_t p, mpz_t n)
{
	int i;

	for(i = p->deg; i > 0; i--)
		gmp_printf("Mod(%Zd,%Zd)*x^%d + ", p->c[i], n, i);
	gmp_printf("Mod(%Zd, %Zd)\n", p->c[i], n);
	return;
}

static inline void mpzpoly_set(mpzpoly_t dst, mpzpoly_t src)
{
	int i;

	dst->deg = src->deg;
	for(i = 0; i <= dst->deg; i++)
		mpz_set(dst->c[i], src->c[i]);
	return;
}

static inline void mpzpoly_fix_degree(mpzpoly_t p)
{
	for(; p->deg && mpz_cmp_ui(p->c[p->deg], 0) == 0; p->deg--);
	return;
}

static inline void mpzpoly_random(mpzpoly_t dst, mpz_t n, gmp_randstate_t rnd)
{
	u_int32_t i, bits;

	bits = mpz_sizeinbase(n, 2) * 2;
	for(i = 0; i < sizeof(dst->c) / sizeof(dst->c[0]); i++)
	{
		mpz_urandomb(dst->c[i], rnd, bits);
		mpz_mod(dst->c[i], dst->c[i], n);
	}
	dst->deg = (sizeof(dst->c) / sizeof(dst->c[0])) - 1;
	mpzpoly_fix_degree(dst);
	return;
}

static inline void mpzpoly_eval_mod(mpz_t dst, mpzpoly_t f, mpz_t x, mpz_t n)
{
	int i;
	mpz_t acc, tmp, e;

	mpz_init(acc);
	mpz_init(tmp);
	mpz_init(e);

	mpz_set(acc, f->c[0]);
	mpz_set_ui(e, 1);

	for(i = 1; i <= f->deg; i++)
	{
		mpz_mul(e, e, x);
		mpz_mod(e, e, n);
		mpz_mul(tmp, e, f->c[i]);
		mpz_add(acc, acc, tmp);
		mpz_mod(acc, acc, n);
	}
	mpz_set(dst, acc);

	mpz_clear(acc);
	mpz_clear(tmp);
	mpz_clear(e);
	return;
}

static inline void mpzpoly_eval_ab(mpz_t dst, mpzpoly_t f, mpz_t a, mpz_t b)
{
	int i;
	mpz_t tmp;
	mpz_t ae[MAX_POLY_DEGREE + 1], be[MAX_POLY_DEGREE + 1];

	mpz_init(tmp);
	for(i = 0; i < MAX_POLY_DEGREE; i++)
	{
		mpz_init(ae[i]);
		mpz_init(be[i]);
	}

	mpz_set_ui(ae[0], 1);
	mpz_set_ui(be[0], 1);
	mpz_set(ae[1], a);
	mpz_set(be[1], b);
	for(i = 2; i <= f->deg; i++)
	{
		mpz_mul(ae[i], ae[i - 1], a);
		mpz_mul(be[i], be[i - 1], b);
	}

	mpz_set_ui(dst, 0);
	for(i = 0; i <= f->deg; i++)
	{
		mpz_mul(tmp, f->c[i], ae[i]);
		mpz_mul(tmp, tmp, be[f->deg - i]);
		mpz_add(dst, dst, tmp);
	}

	mpz_clear(tmp);
	for(i = 0; i < MAX_POLY_DEGREE; i++)
	{
		mpz_clear(ae[i]);
		mpz_clear(be[i]);
	}
	return;
}

static inline void mpzpoly_make_monic(mpzpoly_t dst, mpzpoly_t src, mpz_t n)
{
	if(mpz_cmp_ui(src->c[src->deg], 1) != 0)
	{
		u_int32_t i;
		mpz_t inv;

		mpz_init(inv);
		mpz_invert(inv, src->c[src->deg], n);
		dst->deg = src->deg;
		mpz_set_ui(dst->c[dst->deg], 1);
		for(i = 0; i < dst->deg; i++)
		{
			mpz_mul(dst->c[i], src->c[i], inv);
			mpz_mod(dst->c[i], dst->c[i], n);
		}
		mpz_clear(inv);
	}
	else
	{
		mpzpoly_set(dst, src);
	}
	return;
}

static inline void mpzpoly_add(mpzpoly_t dst, mpzpoly_t src1, mpzpoly_t src2)
{
	int i;

	if(src1->deg < src2->deg)
	{
		for(i = 0; i <= src1->deg; i++)
			mpz_add(dst->c[i], src1->c[i], src2->c[i]);
		for(; i <= src2->deg; i++)
			mpz_set(dst->c[i], src2->c[i]);
		dst->deg = src2->deg;
	}
	else
	{
		for(i = 0; i <= src2->deg; i++)
			mpz_add(dst->c[i], src1->c[i], src2->c[i]);
		for(; i <= src1->deg; i++)
			mpz_set(dst->c[i], src1->c[i]);
		dst->deg = src1->deg;
	}
	return;
}

static inline void mpzpoly_sub(mpzpoly_t dst, mpzpoly_t src1, mpzpoly_t src2)
{
	int i;

	if(src1->deg < src2->deg)
	{
		for(i = 0; i <= src1->deg; i++)
			mpz_sub(dst->c[i], src1->c[i], src2->c[i]);
		for(; i <= src2->deg; i++)
			mpz_neg(dst->c[i], src2->c[i]);
		dst->deg = src2->deg;
	}
	else
	{
		for(i = 0; i <= src2->deg; i++)
			mpz_sub(dst->c[i], src1->c[i], src2->c[i]);
		for(; i <= src1->deg; i++)
			mpz_set(dst->c[i], src1->c[i]);
		dst->deg = src1->deg;
	}
	return;
}

static inline void mpzpoly_mod(mpzpoly_t dst, mpzpoly_t src, mpz_t n)
{
	int i;

	for(i = 0; i <= src->deg; i++)
		mpz_mod(dst->c[i], src->c[i], n);
	dst->deg = src->deg;
	mpzpoly_fix_degree(dst);
	return;
}

static inline void mpzpoly_modadd(mpzpoly_t dst, mpzpoly_t src1, mpzpoly_t src2, mpz_t n)
{
	int i;

	if(src1->deg < src2->deg)
	{
		for(i = 0; i <= src1->deg; i++)
		{
			mpz_add(dst->c[i], src1->c[i], src2->c[i]);
			mpz_mod(dst->c[i], dst->c[i], n);
		}
		for(; i <= src2->deg; i++)
			mpz_set(dst->c[i], src2->c[i]);
		dst->deg = src2->deg;
	}
	else
	{
		for(i = 0; i <= src2->deg; i++)
		{
			mpz_add(dst->c[i], src1->c[i], src2->c[i]);
			mpz_mod(dst->c[i], dst->c[i], n);
		}
		for(; i <= src1->deg; i++)
			mpz_set(dst->c[i], src1->c[i]);
		dst->deg = src1->deg;
	}
	mpzpoly_fix_degree(dst);
	return;
}

static inline void mpzpoly_modsub(mpzpoly_t dst, mpzpoly_t src1, mpzpoly_t src2, mpz_t n)
{
	int i;

	if(src1->deg < src2->deg)
	{
		for(i = 0; i <= src1->deg; i++)
		{
			mpz_sub(dst->c[i], src1->c[i], src2->c[i]);
			mpz_mod(dst->c[i], dst->c[i], n);
		}
		for(; i <= src2->deg; i++)
		{
			mpz_neg(dst->c[i], src2->c[i]);
			mpz_mod(dst->c[i], dst->c[i], n);
		}
		dst->deg = src2->deg;
	}
	else
	{
		for(i = 0; i <= src2->deg; i++)
		{
			mpz_sub(dst->c[i], src1->c[i], src2->c[i]);
			mpz_mod(dst->c[i], dst->c[i], n);
		}
		for(; i <= src1->deg; i++)
		{
			mpz_set(dst->c[i], src1->c[i]);
			mpz_mod(dst->c[i], dst->c[i], n);
		}
		dst->deg = src1->deg;
	}
	mpzpoly_fix_degree(dst);
	return;
}

static inline void mpzpoly_modmul_mpz(mpzpoly_t dst, mpzpoly_t src1, mpz_t src2, mpz_t n)
{
	int i;

	for(i = 0; i <= src1->deg; i++)
	{
		mpz_mul(dst->c[i], src1->c[i], src2);
		mpz_mod(dst->c[i], dst->c[i], n);
	}
	dst->deg = src1->deg;
	mpzpoly_fix_degree(dst);
	return;
}

extern void mpzpoly_mod_poly(mpzpoly_t dst, mpzpoly_t src, mpzpoly_t _mod, mpz_t n);
extern void mpzpoly_modmul_poly(mpzpoly_t dst, mpzpoly_t src1, mpzpoly_t src2, mpzpoly_t mod, mpz_t n);
extern void mpzpoly_gcd(mpzpoly_t dst, mpzpoly_t g_in, mpzpoly_t h_in, mpz_t n);
extern void mpzpoly_xpow(mpzpoly_t dst, mpz_t shift, mpz_t n, mpzpoly_t mod, mpz_t p);
extern u_int32_t mpzpoly_get_roots(mpz_t *roots, mpzpoly_t _f, mpz_t p);

#endif /* MPZPOLY_H_ */
