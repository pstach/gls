/*
 * test_pm1.c
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */
#include <stdio.h>
#include <stdlib.h>
#include "pm1.h"

int main(int argc, char *argv[])
{
	int bt;
	u_int32_t i, seed;
	pm1_plan_t plan;

	seed = time(NULL) ^ (getpid() << 16);
	if(argc > 1)
		seed = strtoul(argv[1], NULL, 0);
	printf("SEED: 0x%08x\n", seed);
	srandom(seed);

	pm1_plan_init(&plan, 315, 2205);
#if 0
	for(i = 0; i < 500000; i++)
	{
		ul64 f, X;
		mod64 mod;

		ul64_init(f);
		ul64_init(X);
		mod64_init(mod);

		ul64_rand(mod->n);
		mod->n[0] |= 1;
		mod64_set(mod, mod->n);
		bt = pm1_stage1_ul64(f, X, mod, &plan);

		if(ul64_cmp_ui(f, 1) != 0)
		{
			/*
			printf("N= ");
			ul64_print(mod->n);
			printf("\nf= ");
			ul64_print(f);
			printf("\n");
			*/
			continue;
		}

		pm1_stage2_ul64(f, X, mod, &plan.stage2);
		if(ul64_cmp_ui(f, 1) != 0)
		{
			/*
			printf("N= ");
			ul64_print(mod->n);
			printf("\nf= ");
			ul64_print(f);
			printf("\n");
			*/
			continue;
		}
	}

	for(i = 0; i < 500000; i++)
	{
		ul128 f, X;
		mod128 mod;

		ul128_init(f);
		ul128_init(X);
		mod128_init(mod);

		ul128_rand(mod->n);
		ul128_setbit(mod->n, 0);
		mod128_set(mod, mod->n);
		bt = pm1_stage1_ul128(f, X, mod, &plan);

		if(ul128_cmp_ui(f, 1) != 0)
		{
			/*
			printf("N= ");
			ul128_print(mod->n);
			printf("\nf= ");
			ul128_print(f);
			printf("\n");
			*/
			continue;
		}

		pm1_stage2_ul128(f, X, mod, &plan.stage2);
		if(ul128_cmp_ui(f, 1) != 0)
		{
			/*
			printf("N= ");
			ul128_print(mod->n);
			printf("\nf= ");
			ul128_print(f);
			printf("\n");
			*/
			continue;
		}
	}
#endif
	for(i = 0; i < 500000; i++)
	{
		mpz_t f, X;
		modmpz_t mod;

		mpz_init(f);
		mpz_init(X);
		mpzmod_init(mod);

		mpz_rand(mod->n);
		mpz_setbit(mod->n, 0);
		mpzmod_set(mod, mod->n);
		bt = pm1_stage1_mpz(f, X, mod, &plan);

		if(mpz_cmp_ui(f, 1) != 0)
		{
			/*
			printf("N= ");
			mpz_printer(mod->n);
			printf("\nf= ");
			mpz_printer(f);
			printf("\n");
			*/
			continue;
		}

		pm1_stage2_mpz(f, X, mod, &plan.stage2);
		if(mpz_cmp_ui(f, 1) != 0)
		{
			/*
			printf("N= ");
			mpz_printer(mod->n);
			printf("\nf= ");
			mpz_printer(f);
			printf("\n");
			*/
			continue;
		}
	}

	pm1_plan_clear(&plan);
	return 0;
}

