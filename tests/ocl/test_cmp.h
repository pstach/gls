
#include "test_common.h"



int ul_compare_test(struct state_t *state) {
	int ret = 0;
	cl_int err = 0;

	struct ul_test_state_t rand_seed_state;
	ret = ul_test_setup(state, ul_rand_seed_two, &rand_seed_state, g_config.config_global_work_size);
	if (ret)
		goto CLEANUP;

	struct ul_test_state_t cmp_test_state;
	ret = ul_test_setup(state, ul_cmp_test, &cmp_test_state, g_config.config_global_work_size);
	if (ret)
		goto CLEANUP;

	struct ul_test_state_t cmp_ui_test_state;
	ret = ul_test_setup(state, ul_cmp_ui_test, &cmp_ui_test_state, g_config.config_global_work_size);
	if (ret)
		goto CLEANUP;

	cl_int seed_offset = 0;

	ocl_mem(devDst, cl_int, state->cxGPUContext, CL_MEM_WRITE_ONLY, cmp_test_state.global_work_size);
	ocl_mem(devSrc1, ul, state->cxGPUContext, CL_MEM_READ_WRITE, cmp_test_state.global_work_size);
	ocl_mem(devSrc2, ul, state->cxGPUContext, CL_MEM_READ_WRITE, cmp_test_state.global_work_size);
	ocl_mem(devSrc2_ui, u_int32_t, state->cxGPUContext, CL_MEM_READ_WRITE, cmp_test_state.global_work_size);

	heap_mem(hstDst, cl_int, cmp_test_state.global_work_size);
	heap_mem(hstSrc1, ul, cmp_test_state.global_work_size);
	heap_mem(hstSrc2, ul, cmp_test_state.global_work_size);
	heap_mem(hstSrc2_ui, u_int32_t, cmp_test_state.global_work_size);

	/* Seed the sources with rand */
	{
		srand(time(NULL));
		seed_offset = rand();

		int arg = 0;
		err = clSetKernelArg(rand_seed_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devSrc1);
		err |= clSetKernelArg(rand_seed_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devSrc2);
		err |= clSetKernelArg(rand_seed_state.ckKernel, arg++, sizeof(cl_int), (void*)&cmp_test_state.global_work_size);
		err |= clSetKernelArg(rand_seed_state.ckKernel, arg++, sizeof(cl_int), (void*)&seed_offset);
		if (CL_SUCCESS != err)
			FAIL_CL("clCreateBuffer", err);

		ret = ul_test_run_kernel(state, &rand_seed_state);
		if (ret)
			goto CLEANUP;

		ul_test_clear(&rand_seed_state);
	}

	/* ul_cmp */
	{
		/* Run test, fetch results */
		{
			int arg = 0;
			err = clSetKernelArg(cmp_test_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devDst);
			err |= clSetKernelArg(cmp_test_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devSrc1);
			err |= clSetKernelArg(cmp_test_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devSrc2);
			err |= clSetKernelArg(cmp_test_state.ckKernel, arg++, sizeof(cl_int), (void*)&cmp_test_state.global_work_size);
			if (CL_SUCCESS != err)
				FAIL_CL("clCreateBuffer", err);

			ret = ul_test_run_kernel(state, &cmp_test_state);
			if (ret)
				goto CLEANUP;

			err = clEnqueueReadBuffer(state->cqCommandQueue, devDst, CL_TRUE, 0, sizeof(cl_int) * cmp_test_state.global_work_size, hstDst, 0, NULL, NULL);
			err |= clEnqueueReadBuffer(state->cqCommandQueue, devSrc1, CL_TRUE, 0, sizeof(ul) * cmp_test_state.global_work_size, hstSrc1, 0, NULL, NULL);
			err |= clEnqueueReadBuffer(state->cqCommandQueue, devSrc2, CL_TRUE, 0, sizeof(ul) * cmp_test_state.global_work_size, hstSrc2, 0, NULL, NULL);
			if (CL_SUCCESS != err)
				FAIL_CL("clEnqueueReadBuffer", err);

			ul_test_clear(&cmp_test_state);
		}

		/* Analyze */
		{
			size_t num_fail = 0;
			for (size_t i = 0; i < cmp_test_state.global_work_size; i++) {
				cl_int cmp = 0;
				int gmpcmp = 0;

				mpz_t s1, s2;
				mpz_init(s1);
				mpz_init(s2);

				cmp = hstDst[i];
				mpz_set_ul(s1, &hstSrc1[i]);
				mpz_set_ul(s2, &hstSrc2[i]);

				gmpcmp = mpz_cmp(s1, s2);

				if (cmp != gmpcmp) {
					num_fail++;
					gmp_printf("** fail: %Zx ? %Zx = %d /= %d\n", s1, s2, gmpcmp, cmp);
				}

				mpz_clear(s1);
				mpz_clear(s2);
			}

			double elapse = (double)(cmp_test_state.stop.tv_sec - cmp_test_state.start.tv_sec) +
							((double)(cmp_test_state.stop.tv_nsec - cmp_test_state.start.tv_nsec))/1000000000;
			printf("Test: %s\t%lu failures\tGlobal: %lu\tWall time: %f\tOps/s: %e\n",
					ul_cmp_test,
					num_fail,
					cmp_test_state.global_work_size,
					elapse,
					cmp_test_state.global_work_size / elapse);
		}
	}

	/* ul_cmp_ui */
	{
		/* Run test, fetch results */
		{
			int arg = 0;
			err = clSetKernelArg(cmp_ui_test_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devDst);
			err |= clSetKernelArg(cmp_ui_test_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devSrc2_ui);
			err |= clSetKernelArg(cmp_ui_test_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devSrc1);
			err |= clSetKernelArg(cmp_ui_test_state.ckKernel, arg++, sizeof(cl_mem), (void*)&devSrc2);
			err |= clSetKernelArg(cmp_ui_test_state.ckKernel, arg++, sizeof(cl_int), (void*)&cmp_ui_test_state.global_work_size);
			if (CL_SUCCESS != err)
				FAIL_CL("clCreateBuffer", err);

			ret = ul_test_run_kernel(state, &cmp_ui_test_state);
			if (ret)
				goto CLEANUP;

			err = clEnqueueReadBuffer(state->cqCommandQueue, devDst, CL_TRUE, 0, sizeof(cl_int) * cmp_ui_test_state.global_work_size, hstDst, 0, NULL, NULL);
			err |= clEnqueueReadBuffer(state->cqCommandQueue, devSrc1, CL_TRUE, 0, sizeof(ul) * cmp_ui_test_state.global_work_size, hstSrc1, 0, NULL, NULL);
			err |= clEnqueueReadBuffer(state->cqCommandQueue, devSrc2, CL_TRUE, 0, sizeof(ul) * cmp_ui_test_state.global_work_size, hstSrc2, 0, NULL, NULL);
			err |= clEnqueueReadBuffer(state->cqCommandQueue, devSrc2_ui, CL_TRUE, 0, sizeof(u_int32_t) * cmp_ui_test_state.global_work_size, hstSrc2_ui, 0, NULL, NULL);
			if (CL_SUCCESS != err)
				FAIL_CL("clEnqueueReadBuffer", err);
		}

		/* Analyze */
		{
			size_t num_fail = 0;
			for (size_t i = 0; i < cmp_ui_test_state.global_work_size; i++) {
				cl_int cmp = 0;
				int gmpcmp = 0;

				mpz_t s1;
				unsigned long s2_ui;
				mpz_init(s1);

				cmp = hstDst[i];
				s2_ui = hstSrc2_ui[i];
				mpz_set_ul(s1, &hstSrc1[i]);

				gmpcmp = mpz_cmp_ui(s1, s2_ui);

				if (cmp != gmpcmp) {
					num_fail++;
					gmp_printf("** fail: %Zx ? %lx = %d /= %d\n", s1, s2_ui, gmpcmp, cmp);
				}

				mpz_clear(s1);
			}

			double elapse = (double)(cmp_ui_test_state.stop.tv_sec - cmp_ui_test_state.start.tv_sec) +
							((double)(cmp_ui_test_state.stop.tv_nsec - cmp_ui_test_state.start.tv_nsec))/1000000000;
			printf("Test: %s\t%lu failures\tGlobal: %lu\tWall time: %f\tOps/s: %e\n",
					ul_cmp_ui_test,
					num_fail,
					cmp_ui_test_state.global_work_size,
					elapse,
					cmp_ui_test_state.global_work_size / elapse);
		}
	}

	CLEANUP:

	ocl_mem_cleanup(devDst);
	ocl_mem_cleanup(devSrc1);
	ocl_mem_cleanup(devSrc2);
	ocl_mem_cleanup(devSrc2_ui);
	heap_mem_cleanup(hstDst);
	heap_mem_cleanup(hstSrc1);
	heap_mem_cleanup(hstSrc2);
	heap_mem_cleanup(hstSrc2_ui);

	ul_test_clear(&cmp_ui_test_state);
	ul_test_clear(&cmp_test_state);
	ul_test_clear(&rand_seed_state);
	return ret;
}
