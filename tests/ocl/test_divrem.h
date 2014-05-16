
#include "test_common.h"


int ul_divremop_test(struct state_t *state) {
	int ret = 0;
	cl_int err = 0;

	struct ul_test_state_t rand_seed_state;
	ret = ul_test_setup(state, ul_rand_seed_two, &rand_seed_state, g_config.config_global_work_size);
	if (ret)
		goto CLEANUP;

	struct ul_test_state_t test_state;
	ret = ul_test_setup(state, ul_divrem_test, &test_state, g_config.config_global_work_size);
	if (ret)
		goto CLEANUP;



	cl_int seed_offset = 0;

	ocl_mem(devDst1, ul, state->cxGPUContext, CL_MEM_WRITE_ONLY, test_state.global_work_size);
	ocl_mem(devDst2, ul, state->cxGPUContext, CL_MEM_WRITE_ONLY, test_state.global_work_size);
	ocl_mem(devSrc1, ul, state->cxGPUContext, CL_MEM_READ_WRITE, test_state.global_work_size);
	ocl_mem(devSrc2, ul, state->cxGPUContext, CL_MEM_READ_WRITE, test_state.global_work_size);

	heap_mem(hstDst1, ul, test_state.global_work_size);
	heap_mem(hstDst2, ul, test_state.global_work_size);
	heap_mem(hstSrc1, ul, test_state.global_work_size);
	heap_mem(hstSrc2, ul, test_state.global_work_size);

	/* Seed the sources with rand */
	{
		srand(time(NULL));
		seed_offset = rand();

		int arg = 0;
		err = clSetKernelArg(rand_seed_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devSrc1);
		err |= clSetKernelArg(rand_seed_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devSrc2);
		err |= clSetKernelArg(rand_seed_state.ckKernel, arg++, sizeof(cl_int), (void*)&test_state.global_work_size);
		err |= clSetKernelArg(rand_seed_state.ckKernel, arg++, sizeof(cl_int), (void*)&seed_offset);
		if (CL_SUCCESS != err)
			FAIL_CL("clCreateBuffer", err);

		ret = ul_test_run_kernel(state, &rand_seed_state);
		if (ret)
			goto CLEANUP;
	}

	/* Run test, fetch results */
	{
		int arg = 0;
		err = clSetKernelArg(test_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devDst1);
		err |= clSetKernelArg(test_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devDst2);
		err |= clSetKernelArg(test_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devSrc1);
		err |= clSetKernelArg(test_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devSrc2);
		err |= clSetKernelArg(test_state.ckKernel, arg++, sizeof(cl_int), (void*)&test_state.global_work_size);
		if (CL_SUCCESS != err)
			FAIL_CL("clCreateBuffer", err);

		ret = ul_test_run_kernel(state, &test_state);
		if (ret)
			goto CLEANUP;

		err = clEnqueueReadBuffer(state->cqCommandQueue, devDst1, CL_TRUE, 0, sizeof(ul) * test_state.global_work_size, hstDst1, 0, NULL, NULL);
		err |= clEnqueueReadBuffer(state->cqCommandQueue, devDst2, CL_TRUE, 0, sizeof(ul) * test_state.global_work_size, hstDst2, 0, NULL, NULL);
		err |= clEnqueueReadBuffer(state->cqCommandQueue, devSrc1, CL_TRUE, 0, sizeof(ul) * test_state.global_work_size, hstSrc1, 0, NULL, NULL);
		err |= clEnqueueReadBuffer(state->cqCommandQueue, devSrc2, CL_TRUE, 0, sizeof(ul) * test_state.global_work_size, hstSrc2, 0, NULL, NULL);
		if (CL_SUCCESS != err)
			FAIL_CL("clEnqueueReadBuffer", err);
	}

	/* Analyze */
	{
		size_t num_fail = 0;
		for (size_t i = 0; i < test_state.global_work_size; i++) {
			int cmpd = 0;
			int cmpr = 0;

			mpz_t r, d, gmpd, gmpr, s1, s2;
			mpz_init(d);
			mpz_init(r);
			mpz_init(gmpd);
			mpz_init(gmpr);
			mpz_init(s1);
			mpz_init(s2);

			mpz_set_ul(d, &hstDst1[i]);
			mpz_set_ul(r, &hstDst2[i]);
			mpz_set_ul(s1, &hstSrc1[i]);
			mpz_set_ul(s2, &hstSrc2[i]);

			mpz_fdiv_qr(gmpd, gmpr, s1, s2);

			cmpd = mpz_cmp(d, gmpd);
			cmpr = mpz_cmp(r, gmpr);

			if (cmpd || cmpr) {
				num_fail ++;
				gmp_printf("** fail: %Zx / %Zx = (%Zx, %Zx) /= (%Zx, %Zx)\n", s1, s2, gmpd, gmpr, d, r);
			}

			mpz_clear(d);
			mpz_clear(r);
			mpz_clear(gmpd);
			mpz_clear(gmpr);
			mpz_clear(s1);
			mpz_clear(s2);
		}

		double elapse = (double)(test_state.stop.tv_sec - test_state.start.tv_sec) +
						((double)(test_state.stop.tv_nsec - test_state.start.tv_nsec))/1000000000;
		printf("Test: %s\t%lu failures\tGlobal: %lu\tWall time: %f\tOps/s: %e\n",
				ul_divrem_test,
				num_fail,
				test_state.global_work_size,
				elapse,
				test_state.global_work_size / elapse);
	}

	CLEANUP:

	ocl_mem_cleanup(devDst1);
	ocl_mem_cleanup(devDst2);
	ocl_mem_cleanup(devSrc1);
	ocl_mem_cleanup(devSrc2);
	heap_mem_cleanup(hstDst1);
	heap_mem_cleanup(hstDst2);
	heap_mem_cleanup(hstSrc1);
	heap_mem_cleanup(hstSrc2);

	ul_test_clear(&test_state);
	ul_test_clear(&rand_seed_state);
	return ret;
}

