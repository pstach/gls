/*
 * test_ecm.c
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */
#include <stdio.h>
#include <stdlib.h>
#include "ecm.h"

int main(int argc, char *argv[])
{
	int bt;
	u_int32_t i, seed;
	ecm_plan_t plan;

	seed = time(NULL) ^ (getpid() << 16);
	if(argc > 1)
		seed = strtoul(argv[1], NULL, 0);
	printf("SEED: 0x%08x\n", seed);
	srandom(seed);

	ecm_plan_init(&plan, 105, 3255, MONTY12, 2);

	for(i = 0; i < 500000; i++)
	{
		ul64 f, b;
		ellM64_point_t X;
		mod64 mod;

		ul64_init(f);
		ul64_init(b);
		ul64_init(X->x);
		ul64_init(X->z);
		mod64_init(mod);

		ul64_rand(mod->n);
		mod->n[0] |= 1;
		mod64_set(mod, mod->n);
		bt = ecm_stage1_ul64(f, X, b, mod, &plan);

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

		ecm_stage2_ul64(f, X, b, mod, &plan.stage2);
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
		ul128 f, b;
		ellM128_point_t X;
		mod128 mod;

		ul128_init(f);
		ul128_init(b);
		ul128_init(X->x);
		ul128_init(X->z);
		mod128_init(mod);

		ul128_rand(mod->n);
		ul128_setbit(mod->n, 0);
		mod128_set(mod, mod->n);
		bt = ecm_stage1_ul128(f, X, b, mod, &plan);

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

		ecm_stage2_ul128(f, X, b, mod, &plan.stage2);
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

	{
		mpz_t f, b;
		ellMmpz_point_t X;
		modmpz_t mod;

		mpz_init(f);
		mpz_init(b);
		mpz_init(X->x);
		mpz_init(X->z);
		mpzmod_init(mod);

		for(i = 0; i < 500000; i++)
		{
			mpz_rand(mod->n);
			mpz_setbit(mod->n, 0);
			mpzmod_set(mod, mod->n);
			bt = ecm_stage1_mpz(f, X, b, mod, &plan);

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

			ecm_stage2_mpz(f, X, b, mod, &plan.stage2);
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
	}

	ecm_plan_clear(&plan);
	return 0;
}

