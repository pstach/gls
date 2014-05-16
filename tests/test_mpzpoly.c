/*
 * test_mpzpoly.c
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include "mpzpoly.h"

int main(int argc, char *argv[])
{
	u_int32_t i, j, seed, n_roots, n_brute_roots;
	mpz_t x, n, tst1, tst2, tst3;
	mpzpoly_t a, b, c, d, e;
	mpz_t roots[sizeof(a->c) / sizeof(a->c[0])];
	mpz_t brute_roots[sizeof(a->c) / sizeof(a->c[0])];
	gmp_randstate_t rnd;

	seed = time(NULL) ^ (getpid() << 16);
	if(argc > 1)
		seed = strtoul(argv[1], NULL, 0);
	printf("SEED: 0x%08x\n", seed);
	srandom(seed);
	gmp_randinit_default(rnd);
	gmp_randseed_ui(rnd, seed);

	for(i = 0; i < sizeof(roots) / sizeof(roots[0]); i++)
		mpz_init(roots[i]);
	for(i = 0; i < sizeof(roots) / sizeof(roots[0]); i++)
		mpz_init(brute_roots[i]);

	mpz_init(x);
	mpz_init(n);
	mpz_init(tst1);
	mpz_init(tst2);
	mpz_init(tst3);
	mpzpoly_init(a);
	mpzpoly_init(b);
	mpzpoly_init(c);
	mpzpoly_init(d);
	mpzpoly_init(e);

	e->deg = 0;
	mpz_set_ui(e->c[0], 1);

	goto howdy;

	for(i = 0; i < 50000; i++)
	{
		mpz_urandomb(n, rnd, 128);
		mpz_nextprime(n, n);
		mpz_urandomb(x, rnd, 128);
		mpz_mod(x, x, n);

		mpzpoly_random(a, n, rnd);
		mpzpoly_random(b, n, rnd);

		mpzpoly_eval_mod(tst1, a, x, n);
		mpzpoly_eval_mod(tst2, b, x, n);
		mpz_add(tst1, tst1, tst2);
		mpz_mod(tst1, tst1, n);

		mpzpoly_add(c, a, b);

		mpzpoly_eval_mod(tst3, c, x, n);

		if(mpz_cmp(tst1, tst3) != 0)
		{
			printf("mpzpoly_add failed\n");
			exit(-1);
		}
	}
	printf("mpzpoly_add success\n");

	for(i = 0; i < 50000; i++)
	{
		mpz_urandomb(n, rnd, 128);
		mpz_nextprime(n, n);
		mpz_urandomb(x, rnd, 128);
		mpz_mod(x, x, n);

		mpzpoly_random(a, n, rnd);
		mpzpoly_random(b, n, rnd);

		mpzpoly_eval_mod(tst1, a, x, n);
		mpzpoly_eval_mod(tst2, b, x, n);
		mpz_sub(tst1, tst1, tst2);
		mpz_mod(tst1, tst1, n);

		mpzpoly_sub(c, a, b);

		mpzpoly_eval_mod(tst3, c, x, n);

		if(mpz_cmp(tst1, tst3) != 0)
		{
			printf("mpzpoly_sub failed\n");
			exit(-1);
		}
	}
	printf("mpzpoly_sub success\n");

	for(i = 0; i < 50000; i++)
	{
		mpz_urandomb(n, rnd, 128);
		mpz_nextprime(n, n);
		mpz_urandomb(x, rnd, 128);
		mpz_mod(x, x, n);

		mpzpoly_random(a, n, rnd);
		mpzpoly_random(b, n, rnd);

		mpzpoly_eval_mod(tst1, a, x, n);
		mpzpoly_eval_mod(tst2, b, x, n);
		mpz_add(tst1, tst1, tst2);
		mpz_mod(tst1, tst1, n);

		mpzpoly_modadd(c, a, b, n);

		mpzpoly_eval_mod(tst3, c, x, n);

		if(mpz_cmp(tst1, tst3) != 0)
		{
			printf("mpzpoly_modadd failed\n");
			exit(-1);
		}
	}
	printf("mpzpoly_modadd success\n");

	for(i = 0; i < 50000; i++)
	{
		mpz_urandomb(n, rnd, 128);
		mpz_nextprime(n, n);
		mpz_urandomb(x, rnd, 128);
		mpz_mod(x, x, n);

		mpzpoly_random(a, n, rnd);
		mpzpoly_random(b, n, rnd);

		mpzpoly_eval_mod(tst1, a, x, n);
		mpzpoly_eval_mod(tst2, b, x, n);
		mpz_sub(tst1, tst1, tst2);
		mpz_mod(tst1, tst1, n);

		mpzpoly_modsub(c, a, b, n);

		mpzpoly_eval_mod(tst3, c, x, n);

		if(mpz_cmp(tst1, tst3) != 0)
		{
			printf("mpzpoly_modsub failed\n");
			exit(-1);
		}
	}
	printf("mpzpoly_modsub success\n");

	for(i = 0; i < 50000; i++)
	{
		mpz_urandomb(n, rnd, 128);
		mpz_nextprime(n, n);
		mpz_urandomb(x, rnd, 128);
		mpz_mod(x, x, n);
		mpz_urandomb(tst3, rnd, 128);
		mpz_mod(tst3, tst3, n);

		mpzpoly_random(a, n, rnd);

		mpzpoly_eval_mod(tst1, a, x, n);
		mpz_mul(tst1, tst1, tst3);
		mpz_mod(tst1, tst1, n);


		mpzpoly_modmul_mpz(b, a, tst3, n);

		mpzpoly_eval_mod(tst2, b, x, n);

		if(mpz_cmp(tst1, tst2) != 0)
		{
			printf("mpzpoly_modmul_mpz failed\n");
			exit(-1);
		}
	}
	printf("mpzpoly_modmul_mpz test1 success\n");

	for(i = 0; i < 50000; i++)
	{
		mpz_urandomb(n, rnd, 128);
		mpz_nextprime(n, n);
		mpz_urandomb(x, rnd, 128);
		mpz_mod(x, x, n);
		mpz_urandomb(tst3, rnd, 128);
		mpz_mod(tst3, tst3, n);

		mpzpoly_random(a, n, rnd);

		mpzpoly_eval_mod(tst1, a, x, n);
		mpz_mul(tst1, tst1, tst3);
		mpz_mod(tst1, tst1, n);


		mpzpoly_modmul_mpz(b, a, tst3, n);

		mpzpoly_eval_mod(tst2, b, x, n);

		if(mpz_cmp(tst1, tst2) != 0)
		{
			printf("mpzpoly_modmul_mpz failed\n");
			exit(-1);
		}
	}
	printf("mpzpoly_modmul_mpz test2 success\n");

#if 0
	for(i = 0; i < 50000; i++)
	{
		mpz_urandomb(n, rnd, 16);
		mpz_nextprime(n, n);
		mpz_set_ui(n, 19);
		mpz_urandomb(x, rnd, 16);
		mpz_mod(x, x, n);

		mpzpoly_random(a, n, rnd);
		mpzpoly_random(b, n, rnd);
		a->deg = 3;
		b->deg = 3;
		mpzpoly_set(d, a);
		mpzpoly_gcd(c, a, b, n);
		printf("A= ");
		mpzpoly_print(a, n);
		printf("B= ");
		mpzpoly_print(b, n);
		printf("C= ");
		mpzpoly_print(c, n);
		printf("if(gcd(A,B)!=C,quit)\n");
	}

	for(i = 0; i < 50000; i++)
	{
		mpz_urandomb(n, rnd, 128);
		mpz_nextprime(n, n);
		mpz_urandomb(x, rnd, 128);
		mpz_mod(x, x, n);

		mpzpoly_random(a, n, rnd);
		mpz_set_ui(tst1, 0);
		mpz_sub_ui(tst2, n, 1);

		printf("A= ");
		mpzpoly_print(a, n);
		//mpzpoly_xpow(mpzpoly_t dst, mpz_t shift, mpz_t n, mpzpoly_t mod, mpz_t p);
		mpzpoly_xpow(b, tst1, tst2, a, n);
		gmp_printf("X= Mod(1,%Zd)*x^%Zd\n", n, tst2);
		printf("B= ");
		mpzpoly_print(b, n);
		printf("if(X%%A==B,quit)\n");
	}
#endif
	for(i = 0; i < 50000; i++)
	{
		do
		{
			mpz_urandomb(n, rnd, 128);
			mpz_nextprime(n, n);
		} while((mpz_get_ui(n) & 7) != 1);

		do
		{
			mpz_urandomb(x, rnd, 128);
			mpz_mod(x, x, n);
		} while(mpz_legendre(x, n) != 1);

		mpz_sqrtm(tst1, x, n);
		mpz_mul(tst2, tst1, tst1);
		mpz_mod(tst2, tst2, n);

		if(mpz_cmp(tst2, x) != 0)
		{
			printf("mpz_sqrtm failed\n");
			exit(0);
		}
	}
	printf("mpz_sqrtm success\n");
howdy:
	for(i = 0; i < 5000; i++)
	{
		mpz_urandomb(n, rnd, 12);
		mpz_nextprime(n, n);

		mpzpoly_random(a, n, rnd);

		n_roots = mpzpoly_get_roots(roots, a, n);

		for(n_brute_roots = 0, mpz_set_ui(tst1, 0); mpz_cmp(tst1, n) < 0; mpz_add_ui(tst1, tst1, 1))
		{
			mpzpoly_eval_mod(tst2, a, tst1, n);
			if(mpz_cmp_ui(tst2, 0) == 0)
				mpz_set(brute_roots[n_brute_roots++], tst1);
		}

		if(n_roots != n_brute_roots)
		{
			printf("n_roots != n_brute_roots\ndumping:\n");
			for(j = 0; j < n_roots; j++)
				gmp_printf("roots[%d] = %Zd\n", j, roots[j]);
			for(j = 0; j < n_brute_roots; j++)
				gmp_printf("brute_roots[%d] = %Zd\n", j, brute_roots[j]);
			exit(0);
		}
		for(j = 0; j < n_roots; j++)
		{
			mpzpoly_eval_mod(tst1, a, roots[j], n);
			if(mpz_cmp_ui(tst1, 0) != 0)
			{
				printf("mpzpoly_get_roots failed\ndumping: \n");
				for(j = 0; j < n_roots; j++)
					gmp_printf("roots[%d] = %Zd\n", j, roots[j]);
				for(j = 0; j < n_brute_roots; j++)
					gmp_printf("brute_roots[%d] = %Zd\n", j, brute_roots[j]);
				exit(-1);
			}
		}
	}
	printf("mpzpoly_get_roots test1 success\n");

	for(i = 0; i < 5000; i++)
	{
		mpz_urandomb(n, rnd, 128);
		mpz_nextprime(n, n);

		mpzpoly_random(a, n, rnd);

		n_roots = mpzpoly_get_roots(roots, a, n);

		for(j = 0; j < n_roots; j++)
		{
			mpzpoly_eval_mod(tst1, a, roots[j], n);
			if(mpz_cmp_ui(tst1, 0) != 0)
			{
				printf("mpzpoly_get_roots failed\ndumping: \n");
				for(j = 0; j < n_roots; j++)
					gmp_printf("roots[%d] = %Zd\n", j, roots[j]);
				exit(-1);
			}
		}
	}
	printf("mpzpoly_get_roots test1 success\n");

	for(i = 0; i < sizeof(roots) / sizeof(roots[0]); i++)
		mpz_clear(brute_roots[i]);
	for(i = 0; i < sizeof(roots) / sizeof(roots[0]); i++)
		mpz_clear(roots[i]);
	mpz_clear(x);
	mpz_clear(n);
	mpz_clear(tst1);
	mpz_clear(tst2);
	mpz_clear(tst3);
	mpzpoly_clear(a);
	mpzpoly_clear(b);
	mpzpoly_clear(c);
	mpzpoly_clear(d);
	mpzpoly_clear(e);
	return 0;
}


