
#include "test_common.h"


#include "test_binop.h"
#include "test_modbinop.h"
#include "test_cmp.h"
#include "test_divrem.h"
#include "test_pm1.h"
#include "test_pp1.h"
#include "test_ecm.h"


int ul_test_all(struct state_t *state) {
	int ret = 0;

	printf("Running tests for ul%d\n", ul_bitsize);

	/* binary operations */
	{
		struct binop_test_t binops[] =
		{
#if TESTOP_SET & TESTOP_ADD
			{ .szKernName = ul_add_test, .mpz_op = &mpz_add, .op = '+', .op2 = NULL},
#endif
#if TESTOP_SET & TESTOP_SUB
			{ .szKernName = ul_sub_test, .mpz_op = &mpz_sub, .op = '-', .op2 = NULL},
#endif
#if TESTOP_SET & TESTOP_MUL
			{ .szKernName = ul_mul_test, .mpz_op = &mpz_mul, .op = '*', .op2 = NULL},
#endif
#if TESTOP_SET & TESTOP_MODINV
			{ .szKernName = ul_mod_inv_test, .mpz_op = &mpz_invert, .op = '\0', .op2 = "^-1 % "},
#endif
		};

		for (size_t i = 0; i < sizeof(binops) / sizeof(binops[0]); i++) {
			ret = ul_binop_test(state, &binops[i]);
			if (ret)
				goto CLEANUP;
		}
	}

#if TESTOP_SET & TESTOP_CMP
	/* comparisons */
	{
		ret = ul_compare_test(state);
		if (ret)
			goto CLEANUP;
	}
#endif

	/* modular binops */
	{
		struct mod_binop_test_t binops[] =
		{
#if TESTOP_SET & TESTOP_MODADD
			{ .szKernName = ul_mod_add_test, .mpz_op = &mpz_add, .op = '+'},
#endif
#if TESTOP_SET & TESTOP_MODSUB
			{ .szKernName = ul_mod_sub_test, .mpz_op = &mpz_sub, .op = '-'},
#endif
#if TESTOP_SET & TESTOP_MODMUL
			{ .szKernName = ul_mod_mul_test, .mpz_op = &mpz_mul, .op = '*'},
#endif
#if TESTOP_SET & TESTOP_MODMUL2
			{ .szKernName = ul_mod_mul_test2, .mpz_op = &mpz_mul, .op = '#'},
#endif
		};

		for (size_t i = 0; i < sizeof(binops) / sizeof(binops[0]); i++) {
			ret = ul_mod_binop_test(state, &binops[i]);
			if (ret)
				goto CLEANUP;
		}
	}

#if TESTOP_SET & TESTOP_DIVREM
	/* division/remainder */
	{
		ret = ul_divremop_test(state);
		if (ret)
			goto CLEANUP;
	}
#endif

#if TESTOP_SET & TESTOP_PM1
	/* P-1 cofactorization */
	{
		ret = ul_pm1op_test(state);
		if (ret)
			goto CLEANUP;
	}
#endif

#if TESTOP_SET & TESTOP_PP1
	/* P-1 cofactorization */
	{
		ret = ul_pp1op_test(state);
		if (ret)
			goto CLEANUP;
	}
#endif

#if TESTOP_SET & TESTOP_ECM
	/* elliptic curve cofactorization */
	{
		ret = ul_ecmop_test(state);
		if (ret)
			goto CLEANUP;
	}
#endif

	CLEANUP:
	return ret;
}


