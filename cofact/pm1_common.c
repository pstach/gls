/*
 * pm1_common.c
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include "cofact_plan.h"

#define PM1_BACKTRACKING 1

int pm1_stage1(ul f, ul X, mod m, pm1_plan_t *plan)
{
	int i, bt;
	u_int64_t *E_ptr, e, mask;
	ul one, two, t, x;

	bt = 0;
	ul_init(one);
	ul_init(two);
	ul_init(t);
	ul_init(x);

	ul_set_ui(one, 1);
	ul_to_montgomery(one, one, m);
	ul_modadd(two, one, one, m);

	/* stage1 - compute 2^E (mod n) */
	ul_set(t, two);
	i = plan->E_n_words - 1;
	E_ptr = &plan->E[i];
	mask = plan->E_mask;
	mask >>= 1;

	for(; i >= 0; i--, E_ptr--)
	{
		e = *E_ptr;
		for(; mask; mask >>= 1)
		{
			ul_modmul(t, t, t, m);
			if(e & mask)
			{
				ul_modadd(t, t, t, m);
			}
		}
		mask = 1ULL << 63;
	}

	ul_set(x, t);

	for(i = 0; i < plan->exp2; i++)
	{
		ul_modmul(x, x, x, m);
#if PM1_BACKTRACKING
		if(ul_cmp(x, one) == 0)
		{
			ul_set(x, t);
			bt = 1;
			break;
		}
		ul_set(t, x);
#endif
	}

	ul_modsub(t, x, one, m);
	ul_gcd(f, t, m->n);

	if(ul_cmp_ui(f, 1) > 0 || plan->B1 >= plan->stage2.B2)
	{
		ul_clear(one);
		ul_clear(two);
		ul_clear(t);
		ul_clear(x);
		return 0;
	}

	/* Compute X = x + 1/x */
	ul_from_montgomery(x, x, m);
	ul_modinv(X, x, m->n);
	ul_modadd(X, X, x, m);
	ul_to_montgomery(X, X, m);

	ul_clear(one);
	ul_clear(two);
	ul_clear(t);
	ul_clear(x);
	return bt;
/*
	pp1_stage2(t, X, &plan->stage2, two, m);
	ul_gcd(f, t, m);

	pp1_stage2 (t, X, &(plan->stage2), two, m);
	  mod_gcd (f, t, m);

	  mod_clear (one, m);
	  mod_clear (two, m);
	  mod_clear (t, m);
	  mod_clear (X, m);
	  mod_clear (x, m);
	  return bt;
*/
	return bt;
}

