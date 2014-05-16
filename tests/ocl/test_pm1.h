
#include "test_common.h"
#include "cofact_plan.h"
#include "ulmpz.h"
#include "ul64.h"
#include "ul128.h"


int pm1_stage1_mpz(mpz_t f, mpz_t X, modmpz_t m, pm1_plan_t *plan);
int pm1_stage1_ul64(ul64 f, ul64 X, mod64 m, pm1_plan_t *plan);
int pm1_stage1_ul128(ul128 f, ul128 X, mod128 m, pm1_plan_t *plan);

#define pm1_stage2_ul64 pp1_stage2_ul64
#define pm1_stage2_ul128 pp1_stage2_ul128
#define pm1_stage2_mpz pp1_stage2_mpz

void pm1_stage2_mpz(mpz_t f, mpz_t X, modmpz_t m, stage2_plan_t *plan);
void pm1_stage2_ul64(ul64 f, ul64 X, mod64 m, stage2_plan_t *plan);
void pm1_stage2_ul128(ul128 f, ul128 X, mod128 m, stage2_plan_t *plan);


int ul_pm1op_test(struct state_t *state) {
	int ret = 0;
	cl_int err = 0;

	size_t global_work_size = g_config.config_global_work_size;

	struct ul_test_state_t test_state_s1;
	ret = ul_test_setup(state, ul_pm1_stage1, &test_state_s1, global_work_size);
	if (ret)
		goto CLEANUP;

	struct ul_test_state_t test_state_r;
	ret = ul_test_setup(state, ul_pm1pp1_reorder, &test_state_r, global_work_size);
	if (ret)
		goto CLEANUP;

	struct ul_test_state_t test_state_s2;
	ret = ul_test_setup(state, ul_pm1_stage2, &test_state_s2, global_work_size);
	if (ret)
		goto CLEANUP;

    printf("global_work_size %ld\n", global_work_size);
    printf("pm1 s1: global %ld, local %ld\n", test_state_s1.global_work_size, test_state_s1.local_work_size);
    printf("pm1  r: global %ld, local %ld\n", test_state_r.global_work_size, test_state_r.local_work_size);
    printf("pm1 s2: global %ld, local %ld\n", test_state_s2.global_work_size, test_state_s2.local_work_size);

	cl_int seed_offset = random();

    ocl_mem(dev_perm, cl_int, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(dev_f, ul, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(_dev_f, ul, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(dev_X, ul, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(_dev_X, ul, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(dev_m, ul, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(_dev_m, ul, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(dev_plan, ocl_pm1_plan_t, state->cxGPUContext, CL_MEM_READ_ONLY, 1);
	ocl_mem(dev_plan_E, u_int64_t, state->cxGPUContext, CL_MEM_READ_ONLY, g_config.pm1_stage1_plan.E_n_words);
	ocl_mem(dev_bt, cl_int, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(_dev_bt, cl_int, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(dev_stage2_plan, ocl_stage2_plan_t, state->cxGPUContext, CL_MEM_READ_ONLY, 1);
	ocl_mem(dev_S1, u_int32_t, state->cxGPUContext, CL_MEM_READ_ONLY, g_config.pm1_stage1_plan.stage2.n_S1);
	ocl_mem(dev_pairs, u_int8_t, state->cxGPUContext, CL_MEM_READ_ONLY, g_config.pm1_stage1_plan.stage2.n_pairs);

	heap_mem(hst_perm, cl_int, global_work_size);
	heap_mem(hst_f, ul, global_work_size);
	heap_mem(hst_X, ul, global_work_size);
	heap_mem(hst_m, ul, global_work_size);
	heap_mem(hst_plan, ocl_pm1_plan_t, 1);
	heap_mem(hst_bt, cl_int, global_work_size);
	heap_mem(hst_stage2_plan, ocl_stage2_plan_t, 1);

	/* Copy the plan to the device */
	{
		hst_plan->B1 = g_config.pm1_stage1_plan.B1;
		hst_plan->B2 = g_config.pm1_stage1_plan.stage2.B2;
		hst_plan->exp2 = g_config.pm1_stage1_plan.exp2;
		hst_plan->E_mask = g_config.pm1_stage1_plan.E_mask;
		hst_plan->E_n_words = g_config.pm1_stage1_plan.E_n_words;

		hst_stage2_plan->B2 = g_config.pm1_stage1_plan.stage2.B2;
		hst_stage2_plan->d = g_config.pm1_stage1_plan.stage2.d;
		hst_stage2_plan->i0 = g_config.pm1_stage1_plan.stage2.i0;
		hst_stage2_plan->i1 = g_config.pm1_stage1_plan.stage2.i1;
		hst_stage2_plan->n_S1 = g_config.pm1_stage1_plan.stage2.n_S1;

		err = clEnqueueWriteBuffer(state->cqCommandQueue,
				dev_plan, CL_TRUE, 0,
				sizeof(ocl_pm1_plan_t),
				hst_plan, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(state->cqCommandQueue,
				dev_plan_E, CL_TRUE, 0,
				sizeof(u_int64_t) * g_config.pm1_stage1_plan.E_n_words,
				g_config.pm1_stage1_plan.E, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(state->cqCommandQueue,
				dev_stage2_plan, CL_TRUE, 0,
				sizeof(ocl_stage2_plan_t),
				hst_stage2_plan, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(state->cqCommandQueue,
				dev_S1, CL_TRUE, 0,
				sizeof(u_int32_t) * g_config.pm1_stage1_plan.stage2.n_S1,
				g_config.pm1_stage1_plan.stage2.S1, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(state->cqCommandQueue,
				dev_pairs, CL_TRUE, 0,
				sizeof(u_int8_t) * g_config.pm1_stage1_plan.stage2.n_pairs,
				g_config.pm1_stage1_plan.stage2.pairs, 0, NULL, NULL);
		if (CL_SUCCESS != err)
			FAIL_CL("clEnqueueWriteBuffer", err);
	}

	/* Run test, fetch results */
	{
		/* stage 1*/
		{
			int arg = 0;
			err = clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_f);
			err |= clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_X);
			err |= clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_m);
			err |= clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_plan);
			err |= clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_plan_E);
			err |= clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_bt);
			err |= clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_int), (void*)&global_work_size);
			err |= clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_int), (void*)&seed_offset);
			if (CL_SUCCESS != err)
				FAIL_CL("clCreateBuffer", err);

			ret = ul_test_run_kernel(state, &test_state_s1);
			if (ret)
				goto CLEANUP;
		}

		/* reoder */
		{
			int arg = 0;
            err = clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_perm);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_f);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_f);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_X);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_X);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_m);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_m);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_bt);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_bt);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_int), (void*)&global_work_size);
			if (CL_SUCCESS != err)
				FAIL_CL("clCreateBuffer", err);

			ret = ul_test_run_kernel(state, &test_state_r);
			if (ret)
				goto CLEANUP;
		}

		/* stage 2 */
		{
			int arg = 0;
			err = clSetKernelArg(test_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_f);
			err |= clSetKernelArg(test_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_X);
			err |= clSetKernelArg(test_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_m);
			err |= clSetKernelArg(test_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_stage2_plan);
			err |= clSetKernelArg(test_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_S1);
			err |= clSetKernelArg(test_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_pairs);
			err |= clSetKernelArg(test_state_s2.ckKernel, arg++, sizeof(cl_int), (void*)&global_work_size);
			if (CL_SUCCESS != err)
				FAIL_CL("clCreateBuffer", err);

			ret = ul_test_run_kernel(state, &test_state_s2);
			if (ret)
				goto CLEANUP;
		}


		/* fetch */
		{
            err = clEnqueueReadBuffer(state->cqCommandQueue, dev_perm, CL_TRUE, 0, sizeof(cl_int) * global_work_size, hst_perm, 0, NULL, NULL);
			err |= clEnqueueReadBuffer(state->cqCommandQueue, dev_f, CL_TRUE, 0, sizeof(ul) * global_work_size, hst_f, 0, NULL, NULL);
			err |= clEnqueueReadBuffer(state->cqCommandQueue, dev_X, CL_TRUE, 0, sizeof(ul) * global_work_size, hst_X, 0, NULL, NULL);
			err |= clEnqueueReadBuffer(state->cqCommandQueue, dev_m, CL_TRUE, 0, sizeof(ul) * global_work_size, hst_m, 0, NULL, NULL);
			err |= clEnqueueReadBuffer(state->cqCommandQueue, dev_bt, CL_TRUE, 0, sizeof(cl_int) * global_work_size, hst_bt, 0, NULL, NULL);
			if (CL_SUCCESS != err)
				FAIL_CL("clEnqueueReadBuffer", err);
		}
	}

	/* Analyze */
	{
		size_t num_fail = 0;
		for (size_t i = 0; i < global_work_size; i++) {
			int cmpf = 0;
			int cmpX = 0;

			cl_int bt = hst_bt[i];
			mpz_t f, X, m, gmpf, gmpX;
			mpz_init(f);
			mpz_init(X);
			mpz_init(m);
			mpz_init(gmpf);
			mpz_init(gmpX);

			mpz_set_ul(f, &hst_f[i]);
			mpz_set_ul(X, &hst_X[i]);
			mpz_set_ul(m, &hst_m[i]);

			modmpz_t mod;
			mpzmod_init(mod);
			mpzmod_set(mod, m);

			int gmp_bt = 0;
			switch (ul_bitsize) {
			case 64:
				{
					ul64 ul_f;
					ul64 ul_X;
					ul64 ul_m;
					ul64_init(ul_f);
					ul64_init(ul_X);

					mpz_get_ul64(ul_m, m);
					mod64 ul_mod;
					mod64_init(ul_mod);
					mod64_set(ul_mod, ul_m);

					ul64_set_ui(ul_f, 0);
					ul64_set_ui(ul_X, 0);
					gmp_bt = pm1_stage1_ul64(ul_f, ul_X, ul_mod, &g_config.pm1_stage1_plan);
					if(ul64_cmp_ui(ul_f, 1) != 0) {
					}
					else {
						pm1_stage2_ul64(ul_f, ul_X, ul_mod, &g_config.pm1_stage1_plan.stage2);
					}
					mpz_set_ul64(gmpf, ul_f);
					mpz_set_ul64(gmpX, ul_X);

					mod64_clear(ul_mod);
					ul64_clear(ul_m);
					ul64_clear(ul_f);
					ul64_clear(ul_X);
				}
				break;
			case 128:
				{
					ul128 ul_f;
					ul128 ul_X;
					ul128 ul_m;
					ul128_init(ul_f);
					ul128_init(ul_X);

					mpz_get_ul128(ul_m, m);
					mod128 ul_mod;
					mod128_init(ul_mod);
					mod128_set(ul_mod, ul_m);

					ul128_set_ui(ul_f, 0);
					ul128_set_ui(ul_X, 0);
					gmp_bt = pm1_stage1_ul128(ul_f, ul_X, ul_mod, &g_config.pm1_stage1_plan);
					if(ul128_cmp_ui(ul_f, 1) != 0) {
					}
					else {
						pm1_stage2_ul128(ul_f, ul_X, ul_mod, &g_config.pm1_stage1_plan.stage2);
					}
					mpz_set_ul128(gmpf, ul_f);
					mpz_set_ul128(gmpX, ul_X);

					mod128_clear(ul_mod);
					ul128_clear(ul_m);
					ul128_clear(ul_f);
					ul128_clear(ul_X);
				}
				break;
			default:
				gmp_bt = pm1_stage1_mpz(gmpf, gmpX, mod, &g_config.pm1_stage1_plan);
				if (mpz_cmp_ui(gmpf, 1) != 0) {
				}
				else {
					pm1_stage2_mpz(gmpf, gmpX, mod, &g_config.pm1_stage1_plan.stage2);
				}
			}

			cmpf = mpz_cmp(f, gmpf);
			if ((mpz_cmp_ui(f, 1) > 0) || g_config.pm1_stage1_plan.B1 >= g_config.pm1_stage1_plan.stage2.B2)
				cmpX = 0;
			else
				cmpX = mpz_cmp(X, gmpX);

			/* if 7 divides the modulus, the test is invalid and we skip it */
			int discard = 0;
			{
				mpz_t r;
				mpz_t seven;
				mpz_init(r);
				mpz_init(seven);

				mpz_set_ui(seven, 7);
				mpz_cdiv_r(r, m, seven);
				discard = (mpz_cmp_ui(r, 0) == 0);

				mpz_clear(seven);
				mpz_clear(r);
			}

			if (!discard)
			{
				if (cmpf || cmpX || bt != gmp_bt) {
					num_fail ++;
					gmp_printf("** fail (i=%d): mod %Zx\n", i, m);
					if (cmpf) {
						gmp_printf("         f: %Zx /= %Zx\n", f, gmpf);
						gmp_printf("         X: %Zx    %Zx\n", X, gmpX);
					}
					if (cmpX)
						gmp_printf("         X: %Zx /= %Zx\n", X, gmpX);
					if (bt != gmp_bt)
						gmp_printf("         bt: %x /= %x\n", bt, gmp_bt);
				}
				else {
					// gmp_printf("correct: f = %Zx, mod = %Zx\n", f, m);
				}
			}

			mpz_clear(f);
			mpz_clear(X);
			mpz_clear(m);
			mpz_clear(gmpf);
			mpz_clear(gmpX);
			mpzmod_clear(mod);
		}

		double elapse_s1 = (double)(test_state_s1.stop.tv_sec - test_state_s1.start.tv_sec) +
						((double)(test_state_s1.stop.tv_nsec - test_state_s1.start.tv_nsec))/1000000000;
		double elapse_r = (double)(test_state_r.stop.tv_sec - test_state_r.start.tv_sec) +
						((double)(test_state_r.stop.tv_nsec - test_state_r.start.tv_nsec))/1000000000;
		double elapse_s2 = (double)(test_state_s2.stop.tv_sec - test_state_s2.start.tv_sec) +
						((double)(test_state_s2.stop.tv_nsec - test_state_s2.start.tv_nsec))/1000000000;
		printf("Test: %s\tWall time: %f\tOps/s: %e\n",
				ul_pm1_stage1,
				elapse_s1,
				global_work_size / elapse_s1);
		printf("      %s\tWall time: %f\tOps/s: %e\n",
				ul_pm1pp1_reorder,
				elapse_r,
				global_work_size / elapse_r);
		printf("      %s\tWall time: %f\tOps/s: %e\n",
				ul_pm1_stage2,
				elapse_s2,
				global_work_size / elapse_s2);
		printf("      %lu failures\tGlobal: %lu\n",
				num_fail,
				global_work_size);
	}

	CLEANUP:

	ocl_mem_cleanup(dev_perm);
	ocl_mem_cleanup(dev_f);
	ocl_mem_cleanup(_dev_f);
	ocl_mem_cleanup(dev_X);
	ocl_mem_cleanup(_dev_X);
	ocl_mem_cleanup(dev_m);
	ocl_mem_cleanup(_dev_m);
	ocl_mem_cleanup(dev_plan);
	ocl_mem_cleanup(dev_plan_E);
	ocl_mem_cleanup(dev_bt);
	ocl_mem_cleanup(_dev_bt);
	ocl_mem_cleanup(dev_stage2_plan);
	ocl_mem_cleanup(dev_S1);
	ocl_mem_cleanup(dev_pairs);

	heap_mem_cleanup(hst_perm);
	heap_mem_cleanup(hst_f);
	heap_mem_cleanup(hst_X);
	heap_mem_cleanup(hst_m);
	heap_mem_cleanup(hst_plan);
	heap_mem_cleanup(hst_bt);

	ul_test_clear(&test_state_s1);
	ul_test_clear(&test_state_r);
	ul_test_clear(&test_state_s2);
	return ret;
}

