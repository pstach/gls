/*
 * ulmpz.c
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include "ulmpz.h"
#include "ulmpz_def.h"

void mpz_sqrtm(mpz_t dst, mpz_t a, mpz_t p)
{
	mpz_t a0;
	u_int32_t pmod8;

	if(mpz_legendre(a, p) != 1)
	{
		mpz_set_ui(dst, 0);
		return;
	}

	if(mpz_cmp_ui(a, 1) == 0)
	{
		mpz_set_ui(dst, 1);
		return;
	}

	pmod8 = mpz_get_ui(p) & 7;
	mpz_init_set(a0, a);

	if(pmod8 == 3 || pmod8 == 7)
	{
		mpz_t tmp;

		mpz_init(tmp);
		mpz_add_ui(tmp, p, 1);
		mpz_div_2exp(tmp, tmp, 2);
		mpz_powm(dst, a0, tmp, p);
		mpz_clear(tmp);
		mpz_clear(a0);
		return;
	}
	else if(pmod8 == 5)
	{
		mpz_t x, y;

		mpz_init(x);
		mpz_init(y);

		mpz_add_ui(x, p, 3);
		mpz_div_2exp(x, x, 3);
		mpz_powm(x, a, x, p);

		mpz_mul(y, x, x);
		mpz_mod(y, y, p);

		if(mpz_cmp(a0, y) == 0)
		{
			mpz_set(dst, x);
			mpz_clear(a0);
			mpz_clear(x);
			mpz_clear(y);
			return;
		}

		mpz_sub_ui(y, p, 1);
		mpz_div_2exp(y, y, 2);
		mpz_set_ui(a0, 2);
		mpz_powm(y, a0, y, p);

		mpz_mul(dst, x, y);
		mpz_mod(dst, dst, p);

		mpz_clear(a0);
		mpz_clear(x);
		mpz_clear(y);
		return;
	}
	else
	{
		u_int32_t i, s;
		mpz_t a1, d0, d1, m, t, ad, s1, pm1;

		mpz_init(a1);
		mpz_init(d0);
		mpz_init(d1);
		mpz_init(m);
		mpz_init(t);
		mpz_init(ad);
		mpz_init(s1);
		mpz_init(pm1);

		for(mpz_set_ui(d0, 2); mpz_cmp(d0, p) < 0; mpz_add_ui(d0, d0, 1))
		{
			if(mpz_legendre(d0, p) == -1)
				break;
		}

		s = mpz_scan1(p, 1);
		mpz_div_2exp(t, p, s);

		// a1 = mp_expo_1(a0, t, p);
		mpz_powm(a1, a0, t, p);
		// d1 = mp_expo_1(d0, t, p);
		mpz_powm(d1, d0, t, p);

		mpz_set_ui(m, 0);
		mpz_sub_ui(pm1, p, 1);

		for(i = 0; i < s; i++)
		{
			mpz_powm(ad, d1, m, p);
			mpz_mul(ad, ad, a1);
			mpz_mod(ad, ad, p);

			mpz_set_ui(s1, 0);
			mpz_setbit(s1, s - 1 - i);
			mpz_powm(ad, ad, s1, p);
			if(mpz_cmp(ad, pm1) == 0)
				mpz_setbit(m, i);
		}

		mpz_add_ui(t, t, 1);
		mpz_div_2exp(t, t, 1);
		mpz_powm(a1, a0, t, p);
		mpz_div_2exp(m, m, 1);
		mpz_powm(d1, d1, m, p);
		mpz_mul(dst, a1, d1);
		mpz_mod(dst, dst, p);

		mpz_clear(a0);
		mpz_clear(a1);
		mpz_clear(d0);
		mpz_clear(d1);
		mpz_clear(m);
		mpz_clear(t);
		mpz_clear(ad);
		mpz_clear(s1);
		mpz_clear(pm1);
	}
	return;
}

void mpz_moddiv2(mpz_t dst, mpz_t src, modmpz_t n)
{
	mpz_set(dst, src);
	if(mpz_get_ui(src) & 1)
		mpz_add(dst, dst, n->n);
	mpz_div_2exp(dst, dst, 1);
	return;
}


void mpz_moddiv3(mpz_t dst, mpz_t src, modmpz_t n)
{
	int i;
	mpz_t t;
	mp_limb_t m3, a3;

	if(!src->_mp_size)
	{
		mpz_set_ui(dst, 0);
		return;
	}

	for(m3 = 0, i = 0; i < n->n->_mp_size; i++)
		m3 += (n->n->_mp_d[i] & 0xff) + (n->n->_mp_d[i] >> 8);
	m3 = m3 % 3;
	if(!m3) /* 3 has no inverse mod n */
		return;
	for(a3 = 0, i = 0; i < src->_mp_size; i++)
		a3 += (src->_mp_d[i] & 0xff) + (src->_mp_d[i] >> 8);
	a3 = a3 % 3;

	mpz_init_set(t, src);

	if(a3 != 0)
	{
		if(a3 + m3 == 3) /* Hence a3 == 1, m3 == 2 or a3 == 2, m3 == 1 */
		{
			mpz_add(t, t, n->n);
		}
		else /* a3 == 1, m3 == 1 or a3 == 2, m3 == 2 */
		{
			mpz_add(t, t, n->n);
			mpz_add(t, t, n->n);
		}
	}

	mpz_div_ui(dst, t, 3);
	mpz_clear(t);
	return;
}

void mpz_moddiv5(mpz_t dst, mpz_t src, modmpz_t n)
{
	mpz_t t;

	mpz_init_set_ui(t, 5);
	mpz_from_montgomery(t, t, n);
	mpz_invert(t, t, n->n);
	mpz_modmul(dst, src, t, n);
	mpz_clear(t);
	return;
}

void mpz_moddiv7(mpz_t dst, mpz_t src, modmpz_t n)
{
	mpz_t t;

	mpz_init_set_ui(t, 7);
	mpz_from_montgomery(t, t, n);
	mpz_invert(t, t, n->n);
	mpz_modmul(dst, src, t, n);
	mpz_clear(t);
	return;
}

void mpz_moddiv11(mpz_t dst, mpz_t src, modmpz_t n)
{
	mpz_t t;

	mpz_init_set_ui(t, 11);
	mpz_from_montgomery(t, t, n);
	mpz_invert(t, t, n->n);
	mpz_modmul(dst, src, t, n);
	mpz_clear(t);
	return;
}

void mpz_moddiv13(mpz_t dst, mpz_t src, modmpz_t n)
{
	mpz_t t;

	mpz_init_set_ui(t, 13);
	mpz_from_montgomery(t, t, n);
	mpz_invert(t, t, n->n);
	mpz_modmul(dst, src, t, n);
	mpz_clear(t);
	return;
}
