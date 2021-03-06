/*
 * pp1_common.c
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include "cofact_plan.h"

#ifndef PP1_BACKTRACKING
#define PP1_BACKTRACKING 0
#endif

static inline void pp1_add(ul r, ul a, ul b, ul d, mod m)
{
	ul_modmul(r, a, b, m);
	ul_modsub(r, r, d, m);
	return;
}

static inline void pp1_double(ul r, ul a, ul two, mod m)
{
	ul_modmul(r, a, a, m);
	ul_modsub(r, r, two, m);
	return;
}

static void mod_V_ul(ul r, ul b, const unsigned long e, mod m)
{
	unsigned long mask;
	ul t, t1, two;

	ul_init(two);
	ul_set_ui(two, 2);
	ul_to_montgomery(two, two, m);

	if(e == 0UL)
	{
		ul_set(r, two);
		ul_clear(two);
		return;
	}

	/* Find highest set bit in e. */
	mask = 1ULL << 63;
	while((mask & e) == 0)
		mask >>= 1;

	/* Exponentiate */
	ul_init(t);
	ul_init(t1);

	ul_set(t, b); /* t = b = V_1 (b) */
	ul_modmul(t1, b, b, m);
	ul_modsub(t1, t1, two, m); /* t1 = b^2 - 2 = V_2 (b) */
	mask >>= 1;

	/* Here t = V_j (b) and t1 = V_{j+1} (b) for j = 1 */
	while (mask > 0UL)
	{
		if (e & mask)
		{
			/* j -> 2*j+1. Compute V_{2j+1} and V_{2j+2} */
			ul_modmul(t, t, t1, m);
			ul_modsub(t, t, b, m); /* V_j * V_{j+1} - V_1 = V_{2j+1} */
			ul_modmul(t1, t1, t1, m);
			ul_modsub(t1, t1, two, m); /* (V_{j+1})^2 - 2 = V_{2j+2} */
		}
		else
		{
			/* j -> 2*j. Compute V_{2j} and V_{2j+1} */
			ul_modmul(t1, t1, t, m);
			ul_modsub(t1, t1, b, m); /* V_j * V_{j+1} - V_1 = V_{2j+1}*/
			ul_modmul(t, t, t, m);
			ul_modsub(t, t, two, m);
		}
		mask >>= 1;
	}

	ul_set(r, t);
	ul_clear(t);
	ul_clear(t1);
	ul_clear(two);
	return;
}

void pp1_stage2(ul f, ul X, mod m, stage2_plan_t *plan)
{
	ul one, two;
	ul Xd, Xid, Xid1, a, a_bk, t;
	ul *Xj; /* FIXME - switch to preallocated */
	unsigned int k, l;

	ul_init(one);
	ul_init(two);
	ul_init(Xd);
	ul_init(t);

	ul_set_ui(one, 1);
	ul_to_montgomery(one, one, m);
	ul_modadd(two, one, one, m);
	ul_to_montgomery(X, X, m);

	{
		int i1, i5;
		ul ap1_0, ap1_1, ap5_0, ap5_1, X2, X6;

		ul_init(ap1_0);
		ul_init(ap1_1);
		ul_init(ap5_0);
		ul_init(ap5_1);
		ul_init(X2);
		ul_init(X6);

		ul_set(ap1_0, X); /* ap1_0 = V_1(X) = X */
		pp1_double(X2, X, two, m); /* X2 = V_2(X) = X^2 - 2 */
		pp1_add(X6, X2, X, X, m); /* V_3(X) = V_2(X) * V_1(X) - V_1(X) */
		pp1_add(ap5_0, X6, X2, X, m); /* V_5(X) = V_3(X) * V_2(X) - V_1(X) */
		pp1_double(X6, X6, two, m); /* V_6(X) = V_3(X)*V_3(X) - 2 */
		pp1_add(ap1_1, X6, X, ap5_0, m); /* V_7(X) = V_6(X) * V_1(X) - V_5(X) */
		pp1_add(ap5_1, X6, ap5_0, X, m); /* V_11(X) = V_6(X) * V_5(X) - V_1(X) */

	    /* Now we generate all the V_j(X) for j in S_1 */
		Xj = (ul *) malloc(plan->n_S1 * sizeof(ul));

		/* We treat the first two manually because those might correspond
		 * to ap1_0 = V_1(X) and ap5_0 = V_5(X)
		 */
		k = 0;
		if(plan->n_S1 > k && plan->S1[k] == 1)
		{
			ul_init(Xj[k]);
			ul_set(Xj[k++], ap1_0);
		}
		if(plan->n_S1 > k && plan->S1[k] == 5)
		{
			ul_init(Xj[k]);
			ul_set(Xj[k++], ap5_0);
		}

		i1 = 7;
		i5 = 11;
		while(k < plan->n_S1)
		{
			if(plan->S1[k] == i1)
			{
				ul_init(Xj[k]);
				ul_set(Xj[k], ap1_1);
				k++;
				continue;
			}

			if(plan->S1[k] == i5)
			{
				ul_init(Xj[k]);
				ul_set(Xj[k], ap5_1);
				k++;
				continue;
			}

			pp1_add(t, ap1_1, X6, ap1_0, m);
			ul_set(ap1_0, ap1_1);
			ul_set(ap1_1, t);
			i1 += 6;

			pp1_add(t, ap5_1, X6, ap5_0, m);
			ul_set(ap5_0, ap5_1);
			ul_set(ap5_1, t);
			i5 += 6;
		}

		/* Also compute Xd = V_d(X) while we've got V_6(X) */
		mod_V_ul(Xd, X6, plan->d / 6, m);

		ul_clear(ap1_0);
		ul_clear(ap1_1);
		ul_clear(ap5_0);
		ul_clear(ap5_1);
		ul_clear(X6);
		ul_clear(X2);
	}

	ul_init(Xid);
	ul_init(Xid1);
	ul_init(a);
	ul_init(a_bk);

	ul_set(a, one);
	ul_set(a_bk, a);
	l = 0;

	{
		/* Compute V_{i0 * d}(X) and V_{(i0 + 1) * d}(X) so we can
		 * compute the remaining V_{id}(X) via an arithmetic progression.
		 */

		/* TODO: init both with the same binary chain. */
	    /* TODO: do both with the same addition chain */
		mod_V_ul(Xid, Xd, plan->i0, m);
	    mod_V_ul(Xid1, Xd, plan->i0 + 1, m);

	    while(plan->pairs[l] != NEXT_PASS)
	    {
	    	while(plan->pairs[l] < NEXT_D && plan->pairs[l] < NEXT_PASS)
			{
				ul_modsub(t, Xid, Xj[plan->pairs[l]], m);
				ul_modmul(a, a, t, m);
				l++;
			}

			/* See if we got a == 0. If yes, restore previous 'a' value and end stage 2 */
			if(ul_cmp_ui(a, 0) == 0)
			{
				ul_set(a, a_bk);
				break;
			}
			ul_set(a_bk, a); /* Save new 'a' value */

			/* Advance i by 1 */
			if(plan->pairs[l] == NEXT_D)
			{
				pp1_add(t, Xid1, Xd, Xid, m);
				ul_set(Xid, Xid1);
				ul_set(Xid1, t);
				l++; /* Skip over NEXT_D */
			}
	    }
	    l++; /* Skip over NEXT_PASS */
	}

	ul_set(f, a);

	for(k = 0; k < plan->n_S1; k++)
		ul_clear(Xj[k]);

	free(Xj);
	ul_clear(Xd);
	ul_clear(Xid);
	ul_clear(Xid1);
	ul_clear(a);
	ul_clear(a_bk);
	ul_clear(t);

	ul_gcd(f, f, m->n);
	return;
}

static inline void pp1_stage1_bc(ul X, const u_int8_t *code, ul two, mod m)
{
	ul A, B, C, t, t2;

	ul_init(A);
	ul_init(B);
	ul_init(C);
	ul_init(t);
	ul_init(t2);

	ul_set(A, X);

	/* Implicit init of first subchain */
	ul_set(B, A);
	ul_set(C, A);

	pp1_double(A, A, two, m);

	while(1)
	{
		switch (*code++)
		{
		case 0: /* Swap A, B */
			ul_set(t, A);
			ul_set(A, B);
			ul_set(B, t);
            break;
		case 1:
			pp1_add(t, A, B, C, m);
			pp1_add(t2, t, A, B, m);
			pp1_add(B, B, t, A, m);
			ul_set(A, t2);
            break;
		case 2:
			pp1_add(B, A, B, C, m);
			pp1_double (A, A, two, m);
			break;
		case 3:
			pp1_add(t, B, A, C, m);
			ul_set(C, B);
			ul_set(B, t);
			break;
		case 4:
			pp1_add(B, B, A, C, m);
			pp1_double(A, A, two, m);
            break;
		case 5:
			pp1_add(C, C, A, B, m);
			pp1_double(A, A, two, m);
			break;
		case 6:
			pp1_double(t, A, two, m);
			pp1_add(t2, A, B, C, m);
			pp1_add(t2, t, t2, C, m);
			ul_set(C, t2);
			pp1_add(t2, t, A, A, m);
			ul_set(A, t2);
			ul_set(t, B);
			ul_set(B, C);
			ul_set(C, t);
			break;
		case 7:
			pp1_add(t, A, B, C, m);
			pp1_add(t2, t, A, B, m);
			ul_set(B, t2);
			pp1_double(t, A, two, m);
			ul_set(t2, A);
			pp1_add(A, A, t, t2, m);
			break;
		case 8:
			pp1_add(t, A, B, C, m);
			pp1_add(C, C, A, B, m);
			ul_set(t2, t);
			ul_set(t, B);
			ul_set(B, t2);
			pp1_double(t, A, two, m);
			pp1_add(t2, A, t, A, m);
			ul_set(A, t2);
			break;
		case 9:
			pp1_add(C, C, B, A, m);
			pp1_double(B, B, two, m);
			break;
		case 10:
			/* Combined final add of old subchain and init of new subchain */
			pp1_add(B, A, B, C, m);
            ul_set(C, B);
            pp1_double(A, B, two, m);
            break;
		case 11:
			/* Rule 3, then rule 0 */
			ul_set(t, A);
			pp1_add(A, B, A, C, m);
			ul_set(C, B);
			ul_set(B, t);
			break;
		case 13:
			/* Rule 3, then subchain end/start */
			pp1_add(t, B, A, C, m);
			pp1_add(C, A, t, B, m);
			ul_set(B, C);
			pp1_double(A, C, two, m);
			break;
		case 14:
			/* Rule 3, rule 0, rule 3 and rule 0, merged a bit */
			ul_set(t, B);
			pp1_add(B, B, A, C, m);
			ul_set(C, A);
			pp1_add(A, A, B, t, m);
			break;
		case 12: /* End of bytecode */
			goto end_of_bytecode;
		default:
			printf("pp1_stage1_bc - invalid bytecode\n");
			exit(-1);
			break;
		}
	}

end_of_bytecode:

	/* Implicit final add */
	pp1_add(A, A, B, C, m); /* Final add */

	ul_set(X, A);

	ul_clear(A);
	ul_clear(B);
	ul_clear(C);
	ul_clear(t);
	ul_clear(t2);
	return;
}

int pp1_stage1(ul f, ul X, mod m, pp1_plan_t *plan)
{
	ul b, two, save, t;
	unsigned int i;
	int bt = 0;

	ul_init(b);
	ul_init(two);
	ul_init(save);
	ul_init(t);

	ul_set_ui(two, 2);
	ul_to_montgomery(two, two, m);

	/* Compute 2/7 (mod N) */
	ul_set(b, two);
	ul_moddiv7(b, b, m);

	pp1_stage1_bc(b, plan->bc, two, m);

	/* Backtracking for the 2's in the exponent */
	ul_set(t, b);
	for(i = 0; i < plan->exp2; i++)
    {
		pp1_double(b, b, two, m);
#if PP1_BACKTRACKING
		if(ul_cmp(b, two) == 0)
        {
			ul_set(b, t);
			bt = 1;
			break;
        }
		ul_set(t, b);
#endif
    }

	ul_modsub(t, b, two, m);
	ul_gcd(f, t, m->n);
/*
	if(ul_cmp_ui(f, 1) == 0 && plan->stage2.B2 > plan->B1)
	{
		pp1_stage2(t, b, &(plan->stage2), two, m);
		mod_gcd (f, t, m);
	}
*/
	ul_clear(b);
	ul_clear(save);
	ul_clear(two);
	ul_clear(t);
	return bt;
}
