/*
 * mpzpoly.c
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */
#include <stdio.h>
#include <stdlib.h>
#include "ulmpz.h"
#include "mpzpoly.h"

void mpzpoly_mod_poly(mpzpoly_t dst, mpzpoly_t src, mpzpoly_t _mod, mpz_t n)
{
	int i, j;
	mpz_t c;
	mpzpoly_t tmp, mod;

	if(_mod->deg == 0)
	{
		dst->deg = 0;
		mpz_set_ui(dst->c[0], 0);
		return;
	}

	mpz_init(c);
	mpzpoly_init(tmp);
	mpzpoly_init(mod);

	mpzpoly_set(tmp, src);
	mpzpoly_make_monic(mod, _mod, n);

	while(tmp->deg >= mod->deg)
	{
		for(i = 0; i < mod->deg; i++)
		{
			j = tmp->deg - (mod->deg - i);
			mpz_mul(c, tmp->c[tmp->deg], mod->c[i]);
			mpz_sub(tmp->c[j], tmp->c[j], c);
			mpz_mod(tmp->c[j], tmp->c[j], n);
		}
		mpz_set_ui(tmp->c[tmp->deg], 0);
		mpzpoly_fix_degree(tmp);
	}
	mpzpoly_set(dst, tmp);
	mpzpoly_clear(tmp);
	mpzpoly_clear(mod);
	mpz_clear(c);
	return;
}

/* NOTE: assumes mod is monic */
void mpzpoly_modmul_poly(mpzpoly_t dst, mpzpoly_t src1, mpzpoly_t src2, mpzpoly_t mod, mpz_t n)
{
	int i, j;
	int idx;
	mpz_t tmp, buf[(sizeof(src1->c) / sizeof(src1->c[0])) * 2 - 1];

	if(mod->deg == 0)
	{
		dst->deg = 0;
		mpz_set_ui(dst->c[0], 0);
		return;
	}

	if(mpz_cmp_ui(mod->c[mod->deg], 1) != 0)
	{
		printf("NOT MONIC!\n");
		exit(0);
	}
	mpz_init(tmp);
	for(i = 0; i < sizeof(buf) / sizeof(buf[0]); i++)
		mpz_init_set_ui(buf[i], 0);

	for(i = 0; i <= src1->deg; i++)
	for(j = 0; j <= src2->deg; j++)
	{
		mpz_mul(tmp, src1->c[i], src2->c[j]);
		mpz_add(buf[i + j], buf[i + j], tmp);
		mpz_mod(buf[i + j], buf[i + j], n);
	}

	for(i = src1->deg + src2->deg; i >= mod->deg; i--)
	{
		if(mpz_cmp_ui(buf[i], 0) == 0)
			continue;
		for(j = 0; j < mod->deg; j++)
		{
			idx = i - (mod->deg - j);
			mpz_mul(tmp, buf[i], mod->c[j]);
			mpz_sub(buf[idx], buf[idx], tmp);
			mpz_mod(buf[idx], buf[idx], n);
		}
		mpz_set_ui(buf[i], 0);
	}

	for(i = 0; i < mod->deg; i++)
		mpz_set(dst->c[i], buf[i]);
	dst->deg = mod->deg - 1;
	mpzpoly_fix_degree(dst);

	mpz_clear(tmp);
	for(i = 0; i < sizeof(buf) / sizeof(buf[0]); i++)
		mpz_clear(buf[i]);
	return;
}

void mpzpoly_gcd(mpzpoly_t dst, mpzpoly_t g_in, mpzpoly_t h_in, mpz_t p)
{
	mpzpoly_t g, h, r;

	mpzpoly_init(g);
	mpzpoly_init(h);
	mpzpoly_init(r);

	if(g_in->deg > h_in->deg)
	{
		mpzpoly_set(g, g_in);
		mpzpoly_set(h, h_in);
	}
	else
	{
		mpzpoly_set(h, g_in);
		mpzpoly_set(g, h_in);
	}

	while(h->deg || mpz_cmp_ui(h->c[0], 0) != 0)
	{
		mpzpoly_mod_poly(r, g, h, p);
		/*
		printf("G= ");
		mpzpoly_print(g, p);
		printf("H= ");
		mpzpoly_print(h, p);
		printf("R= ");
		mpzpoly_print(r, p);
		printf("if(G%%H!=R,quit)\n");
		*/
		mpzpoly_set(g, h);
		mpzpoly_set(h, r);
	}

	if(g->deg == 0)
		mpz_set_ui(g->c[0], 1);
	mpzpoly_set(dst, g);
	/*
	printf("OUT= ");
	mpzpoly_print(dst, p);
	*/
	mpzpoly_clear(g);
	mpzpoly_clear(h);
	mpzpoly_clear(r);
	return;
}

void mpzpoly_xpow(mpzpoly_t dst, mpz_t shift, mpz_t n, mpzpoly_t mod, mpz_t p)
{
	mpzpoly_t modnorm, a, g, h;
	int i;

	mpzpoly_init(modnorm);
	mpzpoly_init(a);
	mpzpoly_init(g);
	mpzpoly_init(h);

	mpzpoly_make_monic(modnorm, mod, p);

	a->deg = 1;
	mpz_set_ui(a->c[1], 1);
	mpz_set(a->c[0], shift);

	mpzpoly_set(g, a);

	for(i = mpz_sizeinbase(n, 2) - 2; i >= 0; i--)
	{
		mpzpoly_modmul_poly(h, g, g, modnorm, p);
		if(mpz_tstbit(n, i))
			mpzpoly_modmul_poly(h, h, a, modnorm, p);
		mpzpoly_set(g, h);
	}

	mpzpoly_set(dst, g);
	mpzpoly_clear(modnorm);
	mpzpoly_clear(a);
	mpzpoly_clear(g);
	mpzpoly_clear(h);
	return;
#if 0
	mpzpoly_t modnorm, x, y;
	int i, bits;

	mpzpoly_init(modnorm);
	mpzpoly_init(x);
	mpzpoly_init(y);

	mpzpoly_make_monic(modnorm, mod, p);

	x->deg = 1;
	mpz_set_ui(x->c[1], 1);
	mpz_set(x->c[0], shift);

	y->deg = 0;
	mpz_set_ui(y->c[0], 1);

	bits = mpz_sizeinbase(n, 2) - 1;
	for(i = 0; i < bits; i++)
	{
		if(mpz_tstbit(n, i))
			mpzpoly_modmul_poly(y, y, x, modnorm, p);
		mpzpoly_modmul_poly(x, x, x, modnorm, p);
	}
	mpzpoly_modmul_poly(y, y, x, modnorm, p);

	mpzpoly_set(dst, y);
	mpzpoly_clear(modnorm);
	mpzpoly_clear(x);
	mpzpoly_clear(y);
	return;
#endif
}

static void mpzpoly_get_roots_recur(mpz_t *roots, mpz_t shift_in, u_int32_t *n_roots, mpzpoly_t f, mpz_t p)
{
	/* get the zeros of a poly, f, that is known to split
	 * completely over Z/pZ. Many thanks to Bob Silverman
	 * for a neat implementation of Cantor-Zassenhaus splitting
	 */
	mpz_t tmp, shift;
	mpzpoly_t g, xpow;
	int deg1, deg2;

	/* base cases of the recursion: we can find the roots
	 * of linear and quadratic polynomials immediately
	 */
	if(f->deg == 1)
	{
		if(mpz_cmp_ui(f->c[1], 1) != 0)
		{
		//if (w != 1) {
			mpz_invert(f->c[1], f->c[1], p);
			mpz_sub(f->c[0], p, f->c[0]);
			mpz_mul(roots[*n_roots], f->c[0], f->c[1]);
			mpz_mod(roots[*n_roots], roots[*n_roots], p);
		}
		else
		{
			if(mpz_cmp_ui(f->c[0], 0) == 0)
				mpz_set_ui(roots[*n_roots], 0);
			else
				mpz_sub(roots[*n_roots], p, f->c[0]);
		}
		(*n_roots)++;
		return;
	}

	if(f->deg == 2)
	{
		mpz_t d, root1, root2, ainv;

		/* if f is a quadratic polynomial, then it will
		 * always have two distinct nonzero roots or else
		 * we wouldn't have gotten to this point. The two
		 * roots are the solution of a general quadratic
		 * equation, mod p
		 */

		mpz_init(d);
		mpz_init(root1);
		mpz_init(root2);
		mpz_init(ainv);

		mpz_mul(d, f->c[0], f->c[2]);
		mpz_mod(d, d, p);

		mpz_sub(root1, p, f->c[1]);
		mpz_set(root2, root1);

		mpz_mul_2exp(ainv, f->c[2], 1);
		if(mpz_cmp(ainv, p) >= 0)
			mpz_sub(ainv, ainv, p);
		mpz_invert(ainv, ainv, p);

		mpz_mul(f->c[1], f->c[1], f->c[1]);
		mpz_mod(f->c[1], f->c[1], p);
		mpz_mul_2exp(d, d, 2);
		mpz_sub(d, f->c[1], d);
		mpz_mod(d, d, p);

		mpz_sqrtm(d, d, p);

		mpz_add(root1, root1, d);
		if(mpz_cmp(root1, p) >= 0)
			mpz_sub(root1, root1, p);
		mpz_sub(root2, root2, d);
		if(mpz_cmp_ui(root2, 0) < 0)
			mpz_add(root2, root2, p);

		mpz_mul(root1, root1, ainv);
		mpz_mod(roots[(*n_roots)++], root1, p);
		mpz_mul(root2, root2, ainv);
		mpz_mod(roots[(*n_roots)++], root2, p);

		mpz_clear(d);
		mpz_clear(root1);
		mpz_clear(root2);
		mpz_clear(ainv);
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

	mpz_init(tmp);
	mpz_init_set(shift, shift_in);
	mpzpoly_init(g);
	mpzpoly_init(xpow);

	mpz_sub_ui(tmp, p, 1);
	mpz_div_2exp(tmp, tmp, 1);

	while(mpz_cmp(shift, p) < 0)
	{
		mpzpoly_xpow(xpow, shift, tmp, f, p);
		mpzpoly_set(g, xpow);
		mpz_sub_ui(g->c[0], g->c[0], 1);
		if(mpz_cmp_ui(g->c[0], 0) < 0)
			mpz_add(g->c[0], g->c[0], p);

		mpzpoly_fix_degree(g);
		mpzpoly_gcd(g, g, f, p);

		if(g->deg > 0)
			break;
		mpz_add_ui(shift, shift, 1);
	}

	/* f was split; repeat the splitting process on
	   the two halves of f. The linear factors of f are
	   either somewhere in x^((p-1)/2) - 1, in
	   x^((p-1)/2) + 1, or 'shift' itself is a linear
	   factor. Test each of these possibilities in turn.
	   In the first two cases, begin trying values of s
	   strictly greater than have been tried thus far */

	deg1 = g->deg;

	mpz_add_ui(tmp, shift, 1);
	mpzpoly_get_roots_recur(roots, tmp, n_roots, g, p);

	mpzpoly_set(g, xpow);

	mpz_add_ui(g->c[0], g->c[0], 1);
	if(mpz_cmp(g->c[0], p) >= 0)
		mpz_sub(g->c[0], g->c[0], p);
	mpzpoly_fix_degree(g);

	mpzpoly_gcd(g, g, f, p);

	deg2 = g->deg;

	if(deg2 > 0)
		mpzpoly_get_roots_recur(roots, tmp, n_roots, g, p);

	if (deg1 + deg2 < f->deg)
	{
		if(mpz_cmp_ui(shift, 0) == 0)
			mpz_set_ui(roots[*n_roots], 0);
		else
			mpz_sub(roots[*n_roots], p, shift);
		(*n_roots)++;
	}

	mpz_clear(tmp);
	mpz_clear(shift);
	mpzpoly_clear(g);
	mpzpoly_clear(xpow);
	return;
}

u_int32_t mpzpoly_get_roots(mpz_t *roots, mpzpoly_t _f, mpz_t p)
{
	/* Find all roots of multiplicity 1 for polynomial _f,
	 * when the coefficients of _f are reduced mod p.
	 * The leading coefficient of _f mod p is returned
	 *
	 * Make count_only nonzero if only the number of roots
	 * and not their identity matters; this is much faster
	 */
	mpz_t pm1, shift;
	mpzpoly_t g, f;
	u_int32_t i, j, n_roots;

	n_roots = 0;
	mpz_init(pm1);
	mpz_init(shift);
	mpzpoly_init(g);
	mpzpoly_init(f);

	/* reduce the coefficients mod p */
	mpzpoly_mod(f, _f, p);
	/* FIXME - projective roots */
	/*
	printf("f->deg: %d\n", f->deg);
	printf("_f->deg: %d\n", _f->deg);
	*/
	/* bail out if the polynomial is zero */
	if(f->deg == 0)
		goto cleanup;

	/* pull out roots of zero. We do this early to
	 * avoid having to handle degree-1 polynomials
	 * in later code
	 */
	if(mpz_cmp_ui(f->c[0], 0) == 0)
	{
		for(i = 1; i <= f->deg && mpz_cmp_ui(f->c[i], 0) == 0; i++);
		for(j = i; i <= f->deg; i++)
			mpz_set(f->c[i - j], f->c[i]);
		f->deg = i - j - 1;
		mpz_set_ui(roots[n_roots++], 0);
	}

	/* handle trivial cases */
	if(f->deg == 0)
		goto cleanup;

	if(f->deg == 1)
	{
		if(mpz_cmp_ui(f->c[1], 1) != 0)
		{
			mpz_invert(f->c[1], f->c[1], p);
			mpz_sub(roots[n_roots], p, f->c[0]);
			mpz_mul(roots[n_roots], roots[n_roots], f->c[1]);
			mpz_mod(roots[n_roots], roots[n_roots], p);
			n_roots++;
			goto cleanup;
		}

		if(mpz_cmp_ui(f->c[0], 0) == 0)
			mpz_set_ui(roots[n_roots], 0);
		else
			mpz_sub(roots[n_roots], p, f->c[0]);
		n_roots++;
		goto cleanup;
	}

	/* the rest of the algorithm assumes p is odd, which
	 * will not work for p=2. Fortunately, in that case
	 * there are only two possible roots, 0 and 1. The above
	 * already tried 0, so try 1 here
	 */

	if(mpz_cmp_ui(p, 2) == 0)
	{
		u_int32_t parity = 0;
		for(i = 0; i <= f->deg; i++)
			parity ^= mpz_get_ui(f->c[i]);
		if(parity == 0)
			mpz_set_ui(roots[n_roots++], 1);
		goto cleanup;
	}

	/* Compute g = gcd(f, x^(p-1) - 1). The result is
	 * a polynomial that is the product of all the linear
	 * factors of f. A given factor only occurs once in
	 * this polynomial
	 */
	mpz_set_ui(shift, 0);
	mpz_sub_ui(pm1, p, 1);

	mpzpoly_xpow(g, shift, pm1, f, p);
	mpz_sub_ui(g->c[0], g->c[0], 1);
	if(mpz_cmp_ui(g->c[0], 0) < 0)
		mpz_add(g->c[0], g->c[0], p);
	mpzpoly_fix_degree(g);

	mpzpoly_gcd(g, g, f, p);

	/* no linear factors */
	if(g->deg < 1)
		goto cleanup;

	/* isolate the linear factors */
	mpz_set_ui(shift, 0);
	mpzpoly_get_roots_recur(roots, shift, &n_roots, g, p);

cleanup:
	mpz_clear(shift);
	mpz_clear(pm1);
	mpzpoly_clear(f);
	mpzpoly_clear(g);
	return n_roots;
}

