
#include "test_common.h"
#include "ul64.h"
#include "ul128.h"

struct mod_binop_test_t {
	const char *szKernName;
	void (*mpz_op)(mpz_ptr, mpz_srcptr, mpz_srcptr);
	char op;
};

int ul_mod_binop_test(struct state_t *state, struct mod_binop_test_t *binop) {
	int ret = 0;
	cl_int err = 0;

	struct ul_test_state_t test_state;
	ret = ul_test_setup(state, binop->szKernName, &test_state, g_config.config_global_work_size);
	if (ret)
		goto CLEANUP;

	struct ul_test_state_t rand_seed_state;
	ret = ul_test_setup(state, ul_rand_seed_three, &rand_seed_state, test_state.global_work_size);
	if (ret)
		goto CLEANUP;


	cl_int seed_offset = 0;

	ocl_mem(devDst, ul, state->cxGPUContext, CL_MEM_WRITE_ONLY, test_state.global_work_size);
	ocl_mem(devSrc1, ul, state->cxGPUContext, CL_MEM_READ_WRITE, test_state.global_work_size);
	ocl_mem(devSrc2, ul, state->cxGPUContext, CL_MEM_READ_WRITE, test_state.global_work_size);
	ocl_mem(devn, ul, state->cxGPUContext, CL_MEM_READ_WRITE, test_state.global_work_size);

	heap_mem(hstDst, ul, test_state.global_work_size);
	heap_mem(hstSrc1, ul, test_state.global_work_size);
	heap_mem(hstSrc2, ul, test_state.global_work_size);
	heap_mem(hstn, ul, test_state.global_work_size);

	/* Seed the sources with rand */
	{
		srand(time(NULL));
		seed_offset = rand();

		int arg = 0;
		err = clSetKernelArg(rand_seed_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devSrc1);
		err |= clSetKernelArg(rand_seed_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devSrc2);
		err |= clSetKernelArg(rand_seed_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devn);
		err |= clSetKernelArg(rand_seed_state.ckKernel, arg++, sizeof(cl_int), (void*)&test_state.global_work_size);
		err |= clSetKernelArg(rand_seed_state.ckKernel, arg++, sizeof(cl_int), (void*)&seed_offset);
		if (CL_SUCCESS != err)
			FAIL_CL("clCreateBuffer", err);

		ret = ul_test_run_kernel(state, &rand_seed_state);
		if (ret)
			goto CLEANUP;

#ifdef TEST_CHECK_RAND_THREE
		err = clEnqueueReadBuffer(state->cqCommandQueue, devSrc1, CL_TRUE, 0, sizeof(ul) * test_state.global_work_size, hstSrc1, 0, NULL, NULL);
		err |= clEnqueueReadBuffer(state->cqCommandQueue, devSrc2, CL_TRUE, 0, sizeof(ul) * test_state.global_work_size, hstSrc2, 0, NULL, NULL);
		err |= clEnqueueReadBuffer(state->cqCommandQueue, devn, CL_TRUE, 0, sizeof(ul) * test_state.global_work_size, hstn, 0, NULL, NULL);
		if (CL_SUCCESS != err)
			FAIL_CL("clEnqueueReadBuffer", err);

		/* Check the invariants: n is odd, and src1, src2 are reduced */
		for (size_t i = 0; i < test_state.global_work_size; i++) {
			mpz_t s1, s2, n;
			mpz_init(s1);
			mpz_init(s2);
			mpz_init(n);

			mpz_set_ul(s1, &hstSrc1[i]);
			mpz_set_ul(s2, &hstSrc2[i]);
			mpz_set_ul(n, &hstn[i]);

			int s1n = mpz_cmp(s1, n);
			int s2n = mpz_cmp(s2, n);
			int n0 = mpz_tstbit(n, 0);
			if ((s1n >= 0) || (s2n >= 0) || !n0) {
				printf("ERROR:\n");
				if (s1n >= 0)
					gmp_printf("  s1n >= 0: %Zx >= %Zx\n", s1, n);
				if (s2n >= 0)
					gmp_printf("  s2n >= 0: %Zx >= %Zx\n", s2, n);
				if (!n0)
					gmp_printf("  !n0: %Zx\n", n);
			}

			mpz_clear(n);
			mpz_clear(s2);
			mpz_clear(s1);
		}
#endif /* TEST_CHECK_RAND_THREE */
	}

	/* Run test, fetch results */
	{
		int arg = 0;
		err = clSetKernelArg(test_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devDst);
		err |= clSetKernelArg(test_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devSrc1);
		err |= clSetKernelArg(test_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devSrc2);
		err |= clSetKernelArg(test_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devn);
		err |= clSetKernelArg(test_state.ckKernel, arg++, sizeof(cl_int), (void*)&test_state.global_work_size);
		if (CL_SUCCESS != err)
			FAIL_CL("clCreateBuffer", err);

		ret = ul_test_run_kernel(state, &test_state);
		if (ret)
			goto CLEANUP;

		err = clEnqueueReadBuffer(state->cqCommandQueue, devDst, CL_TRUE, 0, sizeof(ul) * test_state.global_work_size, hstDst, 0, NULL, NULL);
		err |= clEnqueueReadBuffer(state->cqCommandQueue, devSrc1, CL_TRUE, 0, sizeof(ul) * test_state.global_work_size, hstSrc1, 0, NULL, NULL);
		err |= clEnqueueReadBuffer(state->cqCommandQueue, devSrc2, CL_TRUE, 0, sizeof(ul) * test_state.global_work_size, hstSrc2, 0, NULL, NULL);
		err |= clEnqueueReadBuffer(state->cqCommandQueue, devn, CL_TRUE, 0, sizeof(ul) * test_state.global_work_size, hstn, 0, NULL, NULL);
		if (CL_SUCCESS != err)
			FAIL_CL("clEnqueueReadBuffer", err);
	}

	/* Analyze */
	{
#if defined(TEST_GMPD_A) && defined(TEST_GMPD_B)
		size_t aux_num_fail = 0;
#endif
		size_t num_fail = 0;
		for (size_t i = 0; i < test_state.global_work_size; i++) {
			int cmp = 0;

			mpz_t d, gmpd, s1, s2, n;
			mpz_init(d);
			mpz_init(gmpd);
			mpz_init(s1);
			mpz_init(s2);
			mpz_init(n);

			mpz_set_ul(d, &hstDst[i]);
			mpz_set_ul(s1, &hstSrc1[i]);
			mpz_set_ul(s2, &hstSrc2[i]);
			mpz_set_ul(n, &hstn[i]);

#ifdef TEST_CHECK_RAND_THREE
			{
				int s1n = mpz_cmp(s1, n);
				int s2n = mpz_cmp(s2, n);
				int n0 = mpz_tstbit(n, 0);
				if ((s1n >= 0) || (s2n >= 0) || !n0) {
					printf("ERROR:\n");
					if (s1n >= 0)
						gmp_printf("  s1n >= 0: %Zx >= %Zx\n", s1, n);
					if (s2n >= 0)
						gmp_printf("  s2n >= 0: %Zx >= %Zx\n", s2, n);
					if (!n0)
						gmp_printf("  !n0: %Zx\n", n);
				}
			}
#endif /* TEST_CHECK_RAND_THREE */

			/*
			 * In the case of Montgomery reduction, we perform
			 * the reduction here using two methods. The first,
			 * whose result is stored in gmpd_a, computes the
			 * reduction naively, explicitly computing
			 *             aR * bR / R mod n.
			 * The second method, whose result is stored in gmpd_b,
			 * computes according to the algorithm presented in
			 * Modern Computer Arithmetic. This algorithm should
			 * match the method used on the GPU side.
			 */
			if (binop->op == '*') {
				if (ul_bitsize == 64) {
					ul64 ul_d;
					ul64 ul_s1;
					ul64 ul_s2;
					ul64 ul_n;
					ul64_init(ul_d);
					ul64_init(ul_s1);
					ul64_init(ul_s2);
					ul64_init(ul_n);

					mpz_get_ul64(ul_s1, s1);
					mpz_get_ul64(ul_s2, s2);
					mpz_get_ul64(ul_n, n);

					mod64 ul_m;
					mod64_init(ul_m);
					mod64_set(ul_m, ul_n);

					ul64_modmul(ul_d, ul_s1, ul_s2, ul_m);
					mpz_set_ul64(gmpd, ul_d);

					mod64_clear(ul_m);

					ul64_clear(ul_d);
					ul64_clear(ul_s1);
					ul64_clear(ul_s2);
					ul64_clear(ul_n);

					/* fix for patrick's reduction bug */
#ifdef FIX_PATRICK
					if (mpz_cmp(d, gmpd)) {
						mpz_t mask;
						mpz_init(mask);
						mpz_setbit(mask, 64);
						mpz_sub_ui(mask, mask, 1);
						mpz_sub(gmpd, gmpd, n);
						mpz_and(gmpd, gmpd, mask);
						mpz_clear(mask);
					}
#endif
				}
				else if (ul_bitsize == 128) {
					ul128 ul_d;
					ul128 ul_s1;
					ul128 ul_s2;
					ul128 ul_n;
					ul128_init(ul_d);
					ul128_init(ul_s1);
					ul128_init(ul_s2);
					ul128_init(ul_n);

					mpz_get_ul128(ul_s1, s1);
					mpz_get_ul128(ul_s2, s2);
					mpz_get_ul128(ul_n, n);

					mod128 ul_m;
					mod128_init(ul_m);
					mod128_set(ul_m, ul_n);

					ul128_modmul(ul_d, ul_s1, ul_s2, ul_m);
					mpz_set_ul128(gmpd, ul_d);

					mod128_clear(ul_m);

					ul128_clear(ul_d);
					ul128_clear(ul_s1);
					ul128_clear(ul_s2);
					ul128_clear(ul_n);
				}
				else {
					binop->mpz_op(gmpd, s1, s2);
#ifdef TEST_GMPD_A
					mpz_t gmpd_a;
					mpz_init(gmpd_a);
					{
						mpz_t r;
						mpz_init(r);
						mpz_setbit(r, ul_bitsize);
						mpz_mod(r, r, n);

						mpz_t r_inv;
						mpz_init(r_inv);
						mpz_invert(r_inv, r, n);

						mpz_mul(gmpd_a, gmpd, r_inv);
						mpz_mod(gmpd_a, gmpd_a, n);

						mpz_clear(r);
						mpz_clear(r_inv);
					}
#endif /* TEST_GMPD_A */
#ifdef TEST_GMPD_B
					mpz_t gmpd_b;
					mpz_init(gmpd_b);
					{
						mpz_set(gmpd_b, gmpd);

						mpz_t beta;
						mpz_init(beta);
						mpz_setbit(beta, 32);

						mpz_t np;
						mpz_init(np);
						mpz_invert(np, n, beta);
						mpz_neg(np, np);
						mpz_mod(np, np, beta);

						for (int i = 0; i < ul_bitsize; i += 32) {
							mpz_t ci;
							mpz_init(ci);
							mpz_fdiv_q_2exp(ci, gmpd_b, i);
							mpz_mod(ci, ci, beta);

							mpz_t q;
							mpz_init(q);
							mpz_mul(q, np, ci);
							mpz_mod(q, q, beta);

							mpz_t qN;
							mpz_init(qN);
							mpz_mul(qN, q, n);
							mpz_mul_2exp(qN, qN, i);
							mpz_add(gmpd_b, gmpd_b, qN);

							mpz_clear(q);
							mpz_clear(ci);
							mpz_clear(qN);
						}

						mpz_fdiv_q_2exp(gmpd_b, gmpd_b, ul_bitsize);
						if (mpz_cmp(gmpd_b, n) >= 0)
							mpz_sub(gmpd_b, gmpd_b, n);

						mpz_clear(np);
						mpz_clear(beta);
					}
#endif /* TEST_GMPD_B */
#if defined(TEST_GMPD_A) && defined(TEST_GMPD_B)
					if (mpz_cmp(gmpd_a, gmpd_b) != 0) {
						aux_num_fail++;
						gmp_printf("ERR: %Zx != %Zx\n", gmpd_a, gmpd_b);
					}
#endif
#ifdef TEST_GMPD_A
					mpz_set(gmpd, gmpd_a);
					mpz_clear(gmpd_a);
#elif defined(TEST_GMPD_B)
					mpz_set(gmpd, gmpd_b);
					mpz_clear(gmpd_b);
#endif
				}
			}
			else if (binop->op == '#') {
				mpz_mul(gmpd, s1, s2);
				mpz_mod(gmpd, gmpd, n);
			}
			else {
				binop->mpz_op(gmpd, s1, s2);
				if (mpz_cmp_ui(gmpd, 0) < 0)
					mpz_add(gmpd, gmpd, n);

				mpz_fdiv_r_2exp(gmpd, gmpd, ul_bitsize);
				mpz_mod(gmpd, gmpd, n);
			}

			cmp = mpz_cmp(d, gmpd);

			if (cmp) {
				num_fail ++;
				gmp_printf("** fail: %Zx %c %Zx = %Zx (mod %Zx) /= %Zx\n", s1, binop->op, s2, gmpd, n, d);
			}
#ifdef TEST_PRINT_GOOD
			else {
				gmp_printf(" good  : %ZX %c %ZX = %Zx (mod %Zx) = %Zx\n", s1, binop->op, s2, gmpd, n, d);
			}
#endif /* TEST_PRINT_GOOD */

			mpz_clear(d);
			mpz_clear(gmpd);
			mpz_clear(s1);
			mpz_clear(s2);
			mpz_clear(n);
		}

		double elapse = (double)(test_state.stop.tv_sec - test_state.start.tv_sec) +
						((double)(test_state.stop.tv_nsec - test_state.start.tv_nsec))/1000000000;
		printf("Test: %s\t%lu failures\tGlobal: %lu\tWall time: %f\tOps/s: %e\n",
				binop->szKernName,
				num_fail,
				test_state.global_work_size,
				elapse,
				test_state.global_work_size / elapse);
#if defined(TEST_GMPD_A) && defined(TEST_GMPD_B)
		if (binop->op == '*') {
			printf("Disagreements in our GMP Montgomery reducers: %lu\n", aux_num_fail);
		}
#endif
	}

	CLEANUP:

	ocl_mem_cleanup(devDst);
	ocl_mem_cleanup(devSrc1);
	ocl_mem_cleanup(devSrc2);
	ocl_mem_cleanup(devn);
	heap_mem_cleanup(hstDst);
	heap_mem_cleanup(hstSrc1);
	heap_mem_cleanup(hstSrc2);
	heap_mem_cleanup(hstn);

	ul_test_clear(&test_state);
	ul_test_clear(&rand_seed_state);
	return ret;
}
