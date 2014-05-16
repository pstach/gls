
#include "test_common.h"
#include "cofact_plan.h"
#include "ulmpz.h"
#include "ul64.h"
#include "ul128.h"

typedef struct
{
	mpz_t x;
	mpz_t z;
} ellM_point_mpz[1];

typedef struct
{
	ul64 x;
	ul64 z;
} ellM_point_ul64[1];

typedef struct
{
	ul128 x;
	ul128 z;
} ellM_point_ul128[1];

typedef struct
{
	ul x;
	ul z;
} ellM_point_ul;


int ecm_stage1_mpz(mpz_t f, ellM_point_mpz P, mpz_t b, modmpz_t m, ecm_plan_t *plan);
int ecm_stage1_ul64(ul64 f, ellM_point_ul64 P, ul64 b, mod64 m, ecm_plan_t *plan);
int ecm_stage1_ul128(ul128 f, ellM_point_ul128 P, ul128 b, mod128 m, ecm_plan_t *plan);

int ecm_stage2_mpz(mpz_t r, ellM_point_mpz P, mpz_t b, modmpz_t m, stage2_plan_t *plan);
int ecm_stage2_ul64(ul64 r, ellM_point_ul64 P, ul64 b, mod64 m, stage2_plan_t *plan);
int ecm_stage2_ul128(ul128 r, ellM_point_ul128 P, ul128 b, mod128 m, stage2_plan_t *plan);


int ul_ecmop_test(struct state_t *state) {
	int ret = 0;
	cl_int err = 0;

	size_t global_work_size = g_config.config_global_work_size;

	struct ul_test_state_t test_state_s1;
	ret = ul_test_setup(state, ul_ecm_stage1, &test_state_s1, global_work_size);
	if (ret)
		goto CLEANUP;

	struct ul_test_state_t test_state_r;
	ret = ul_test_setup(state, ul_ecm_reorder, &test_state_r, global_work_size);
	if (ret)
		goto CLEANUP;

	struct ul_test_state_t test_state_s2;
	ret = ul_test_setup(state, ul_ecm_stage2, &test_state_s2, global_work_size);
	if (ret)
		goto CLEANUP;

    printf("global_work_size %ld\n", global_work_size);
    printf("ecm s1: global %ld, local %ld\n", test_state_s1.global_work_size, test_state_s1.local_work_size);
    printf("ecm  r: global %ld, local %ld\n", test_state_r.global_work_size, test_state_r.local_work_size);
    printf("ecm s2: global %ld, local %ld\n", test_state_s2.global_work_size, test_state_s2.local_work_size);


	cl_int seed_offset = random();

    ocl_mem(dev_perm, cl_int, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);

	// algebraic data
	ocl_mem(dev_f, ul, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(_dev_f, ul, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(dev_X, ellM_point_ul, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(_dev_X, ellM_point_ul, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(dev_b, ul, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(_dev_b, ul, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(dev_m, ul, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(_dev_m, ul, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(dev_bt1, cl_int, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(_dev_bt1, cl_int, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	ocl_mem(dev_bt2, cl_int, state->cxGPUContext, CL_MEM_READ_WRITE, global_work_size);
	// stage 1 plan
	ocl_mem(dev_plan, ocl_ecm_plan_t, state->cxGPUContext, CL_MEM_READ_ONLY, 1);
	ocl_mem(dev_plan_bc, u_int8_t, state->cxGPUContext, CL_MEM_READ_ONLY, g_config.ecm_stage1_plan.bc_len);
	// stage 2 plan
	ocl_mem(dev_stage2_plan, ocl_stage2_plan_t, state->cxGPUContext, CL_MEM_READ_ONLY, 1);
	ocl_mem(dev_S1, u_int32_t, state->cxGPUContext, CL_MEM_READ_ONLY, g_config.ecm_stage1_plan.stage2.n_S1);
	ocl_mem(dev_pairs, u_int8_t, state->cxGPUContext, CL_MEM_READ_ONLY, g_config.ecm_stage1_plan.stage2.n_pairs);


	heap_mem(hst_perm, cl_int, global_work_size);
	heap_mem(hst_f, ul, global_work_size);
	heap_mem(hst_X, ellM_point_ul, global_work_size);
	heap_mem(hst_b, ul, global_work_size);
	heap_mem(hst_m, ul, global_work_size);
	heap_mem(hst_bt1, cl_int, global_work_size);
	heap_mem(hst_bt2, cl_int, global_work_size);
	heap_mem(hst_plan, ocl_ecm_plan_t, 1);
	heap_mem(hst_stage2_plan, ocl_stage2_plan_t, 1);

	/* Copy the plan to the device */
	{
		hst_plan->B1 = g_config.ecm_stage1_plan.B1;
		hst_plan->bc_len = g_config.ecm_stage1_plan.bc_len;
		hst_plan->exp2 = g_config.ecm_stage1_plan.exp2;
		hst_plan->sigma = g_config.ecm_stage1_plan.sigma;
		hst_plan->parameterization = g_config.ecm_stage1_plan.parameterization;

		hst_stage2_plan->B2 = g_config.ecm_stage1_plan.stage2.B2;
		hst_stage2_plan->d = g_config.ecm_stage1_plan.stage2.d;
		hst_stage2_plan->i0 = g_config.ecm_stage1_plan.stage2.i0;
		hst_stage2_plan->i1 = g_config.ecm_stage1_plan.stage2.i1;
		hst_stage2_plan->n_S1 = g_config.ecm_stage1_plan.stage2.n_S1;

		err = clEnqueueWriteBuffer(state->cqCommandQueue, dev_plan, CL_TRUE, 0, sizeof(ocl_ecm_plan_t), hst_plan, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(state->cqCommandQueue, dev_plan_bc, CL_TRUE, 0, sizeof(u_int8_t) * g_config.ecm_stage1_plan.bc_len, g_config.ecm_stage1_plan.bc, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(state->cqCommandQueue, dev_stage2_plan, CL_TRUE, 0, sizeof(ocl_stage2_plan_t), hst_stage2_plan, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(state->cqCommandQueue, dev_S1, CL_TRUE, 0, sizeof(u_int32_t) * g_config.ecm_stage1_plan.stage2.n_S1, g_config.ecm_stage1_plan.stage2.S1, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(state->cqCommandQueue, dev_pairs, CL_TRUE, 0, sizeof(u_int8_t) * g_config.ecm_stage1_plan.stage2.n_pairs, g_config.ecm_stage1_plan.stage2.pairs, 0, NULL, NULL);
		if (CL_SUCCESS != err)
			FAIL_CL("clEnqueueWriteBuffer", err);
	}

	/* Run test, fetch results */
	{
		/* stage 1 */
		{
			int arg = 0;
			err = clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_f);
			err |= clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_X);
			err |= clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_b);
			err |= clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_m);
			err |= clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_bt1);

			err |= clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_plan);
			err |= clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_plan_bc);

			err |= clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_int), (void*)&global_work_size);
			err |= clSetKernelArg(test_state_s1.ckKernel, arg++, sizeof(cl_int), (void*)&seed_offset);
			if (CL_SUCCESS != err)
				FAIL_CL("clCreateBuffer", err);

			ret = ul_test_run_kernel(state, &test_state_s1);
			if (ret)
				goto CLEANUP;
		}

		/* reorder */
		{
			int arg = 0;
            err = clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_perm);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_f);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_f);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_X);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_X);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_b);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_b);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_m);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_m);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&_dev_bt1);
			err |= clSetKernelArg(test_state_r.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_bt1);
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
			err |= clSetKernelArg(test_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_b);
			err |= clSetKernelArg(test_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_m);
			err |= clSetKernelArg(test_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_bt2);

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
			err |= clEnqueueReadBuffer(state->cqCommandQueue, dev_X, CL_TRUE, 0, sizeof(ellM_point_ul) * global_work_size, hst_X, 0, NULL, NULL);
			err |= clEnqueueReadBuffer(state->cqCommandQueue, dev_b, CL_TRUE, 0, sizeof(ul) * global_work_size, hst_b, 0, NULL, NULL);
			err |= clEnqueueReadBuffer(state->cqCommandQueue, dev_m, CL_TRUE, 0, sizeof(ul) * global_work_size, hst_m, 0, NULL, NULL);
			err |= clEnqueueReadBuffer(state->cqCommandQueue, dev_bt1, CL_TRUE, 0, sizeof(cl_int) * global_work_size, hst_bt1, 0, NULL, NULL);
			err |= clEnqueueReadBuffer(state->cqCommandQueue, dev_bt2, CL_TRUE, 0, sizeof(cl_int) * global_work_size, hst_bt2, 0, NULL, NULL);
			if (CL_SUCCESS != err)
				FAIL_CL("clEnqueueReadBuffer", err);
		}
	}

	/* Analyze */
	{
		size_t num_fail = 0;
		for (size_t i = 0; i < global_work_size; i++) {
			int cmpf = 0;
			int cmpb = 0;
			int cmpXx = 0;
			int cmpXz = 0;
            int cmpbt1 = 0;
            int cmpbt2 = 0;

            int bt1 = 0;
            int bt2 = 0;

			mpz_t f, b, gmpf, gmpb, m;
			ellM_point_mpz X, gmpX;
			mpz_init(f);
			mpz_init(b);
			mpz_init(gmpf);
			mpz_init(gmpb);
			mpz_init(m);
			mpz_init(X->x);
			mpz_init(X->z);
			mpz_init(gmpX->x);
			mpz_init(gmpX->z);

			mpz_set_ul(f, &hst_f[i]);
			mpz_set_ul(X->x, &hst_X[i].x);
			mpz_set_ul(X->z, &hst_X[i].z);
			mpz_set_ul(b, &hst_b[i]);
			mpz_set_ul(m, &hst_m[i]);

			switch (ul_bitsize) {
			case 64:
				{
					ul64 ul_f, ul_b, ul_m;
					ellM_point_ul64 ul_X;

					ul64_init(ul_f);
					ul64_init(ul_b);
					ul64_init(ul_m);
					ul64_init(ul_X->x);
					ul64_init(ul_X->z);

					mpz_get_ul64(ul_m, m);
					mod64 ul_mod;
					mod64_init(ul_mod);
					mod64_set(ul_mod, ul_m);

					ul64_set_ui(ul_f, 0);
					ul64_set_ui(ul_b, 0);
					ul64_set_ui(ul_X->x, 0);
					ul64_set_ui(ul_X->z, 0);

					bt1 = ecm_stage1_ul64(ul_f, ul_X, ul_b, ul_mod, &g_config.ecm_stage1_plan);
					if(ul64_cmp_ui(ul_f, 1) != 0) {
                        bt2 = 0;
					}
					else {
						bt2 = ecm_stage2_ul64(ul_f, ul_X, ul_b, ul_mod, &g_config.ecm_stage1_plan.stage2);
					}
					mpz_set_ul64(gmpf, ul_f);
					mpz_set_ul64(gmpb, ul_b);
					mpz_set_ul64(gmpX->x, ul_X->x);
					mpz_set_ul64(gmpX->z, ul_X->z);

					mod64_clear(ul_mod);

					ul64_clear(ul_f);
					ul64_clear(ul_b);
					ul64_clear(ul_m);
					ul64_clear(ul_X->x);
					ul64_clear(ul_X->z);
				}
				break;
			case 128:
				{
					ul128 ul_f, ul_b, ul_m;
					ellM_point_ul128 ul_X;

					ul128_init(ul_f);
					ul128_init(ul_b);
					ul128_init(ul_m);
					ul128_init(ul_X->x);
					ul128_init(ul_X->z);

					mpz_get_ul128(ul_m, m);
					mod128 ul_mod;
					mod128_init(ul_mod);
					mod128_set(ul_mod, ul_m);

					bt1 = ecm_stage1_ul128(ul_f, ul_X, ul_b, ul_mod, &g_config.ecm_stage1_plan);
					if(ul128_cmp_ui(ul_f, 1) != 0) {
                        bt2 = 0;
					}
					else {
						bt2 = ecm_stage2_ul128(ul_f, ul_X, ul_b, ul_mod, &g_config.ecm_stage1_plan.stage2);
					}
					mpz_set_ul128(gmpf, ul_f);
					mpz_set_ul128(gmpb, ul_b);
					mpz_set_ul128(gmpX->x, ul_X->x);
					mpz_set_ul128(gmpX->z, ul_X->z);

					mod128_clear(ul_mod);

					ul128_clear(ul_f);
					ul128_clear(ul_b);
					ul128_clear(ul_m);
					ul128_clear(ul_X->x);
					ul128_clear(ul_X->z);
				}
				break;
			default:
				{
					modmpz_t mod;
					mpzmod_init(mod);
					mpzmod_set(mod, m);
					bt1 = ecm_stage1_mpz(gmpf, gmpX, gmpb, mod, &g_config.ecm_stage1_plan);
					if(mpz_cmp_ui(gmpf, 1) != 0) {
                        bt2 = 0;
					}
					else {
						bt2 = ecm_stage2_mpz(gmpf, gmpX, gmpb, mod, &g_config.ecm_stage1_plan.stage2);
					}
					mpzmod_clear(mod);
				}
			}

			/* make sure our modulus doesn't have any forbidden factors */
			int abort_fail = 0;
			{
				mpz_t r;
				mpz_init(r);

				mpz_t _3;
				mpz_init(_3);
				mpz_set_ui(_3, 3);
				mpz_cdiv_r(r, m, _3);
				mpz_clear(_3);
				abort_fail |= (mpz_cmp_ui(r, 0) == 0);

				mpz_t _13;
				mpz_init(_13);
				mpz_set_ui(_13, 13);
				mpz_cdiv_r(r, m, _13);
				mpz_clear(_13);
				abort_fail |= (mpz_cmp_ui(r, 0) == 0);

				mpz_t _37;
				mpz_init(_37);
				mpz_set_ui(_37, 37);
				mpz_cdiv_r(r, m, _37);
				mpz_clear(_37);
				abort_fail |= (mpz_cmp_ui(r, 0) == 0);

				mpz_t _28;
				mpz_init(_28);
				mpz_set_ui(_28, 28);
				mpz_cdiv_r(r, m, _28);
				mpz_clear(_28);
				abort_fail |= (mpz_cmp_ui(r, 0) == 0);

				mpz_clear(r);
			}

			cmpf = mpz_cmp(f, gmpf);
			cmpb = mpz_cmp(b, gmpb);
			cmpXx = mpz_cmp(X->x, gmpX->x);
			cmpXz = mpz_cmp(X->z, gmpX->z);
            cmpbt1 = (bt1 == hst_bt1[i]) ? 0 : 1;
            cmpbt2 = (bt2 == hst_bt2[i]) ? 0 : 1;
			if (!abort_fail && (cmpf || cmpb || cmpXx || cmpXz || cmpbt1 || cmpbt2)) {
				num_fail ++;
				gmp_printf("** fail (i=%d): mod %Zx\n", i, m);
				if (cmpf)
					gmp_printf("         f: %Zx /= %Zx\n", f, gmpf);
				if (cmpb)
					gmp_printf("         b: %Zx /= %Zx\n", b, gmpb);
				if (cmpXx || cmpXz)
					gmp_printf("         X: (%Zx:%Zx) /= (%Zx:%Zx)\n", X->x, X->z, gmpX->x, gmpX->z);
                if (cmpbt1 || cmpbt2)
                    gmp_printf("         bt(1,2): (%d, %d) /= (%d, %d)\n", hst_bt1[i], hst_bt2[i], bt1, bt2);
			}
			else {
				/* gmp_printf("correct: f = %Zx, mod = %Zx\n", f, m); */
			}

			mpz_clear(f);
			mpz_clear(b);
			mpz_clear(gmpf);
			mpz_clear(gmpb);
			mpz_clear(m);
			mpz_clear(X->x);
			mpz_clear(X->z);
			mpz_clear(gmpX->x);
			mpz_clear(gmpX->z);
		}

		double elapse_s1 = (double)(test_state_s1.stop.tv_sec - test_state_s1.start.tv_sec) +
						((double)(test_state_s1.stop.tv_nsec - test_state_s1.start.tv_nsec))/1000000000;
		double elapse_r = (double)(test_state_r.stop.tv_sec - test_state_r.start.tv_sec) +
						((double)(test_state_r.stop.tv_nsec - test_state_r.start.tv_nsec))/1000000000;
		double elapse_s2 = (double)(test_state_s2.stop.tv_sec - test_state_s2.start.tv_sec) +
						((double)(test_state_s2.stop.tv_nsec - test_state_s2.start.tv_nsec))/1000000000;
		printf("Test: %s\tWall time: %f\tOps/s: %e\n",
				ul_ecm_stage1,
				elapse_s1,
				global_work_size / elapse_s1);
		printf("      %s\tWall time: %f\tOps/s: %e\n",
				ul_ecm_reorder,
				elapse_r,
				global_work_size / elapse_r);
		printf("      %s\tWall time: %f\tOps/s: %e\n",
				ul_ecm_stage2,
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
	ocl_mem_cleanup(dev_b);
	ocl_mem_cleanup(_dev_b);
	ocl_mem_cleanup(dev_m);
	ocl_mem_cleanup(_dev_m);
	ocl_mem_cleanup(dev_plan);
	ocl_mem_cleanup(dev_plan_bc);
	ocl_mem_cleanup(dev_stage2_plan);
	ocl_mem_cleanup(dev_S1);

	heap_mem_cleanup(hst_perm);
	heap_mem_cleanup(hst_f);
	heap_mem_cleanup(hst_X);
	heap_mem_cleanup(hst_b);
	heap_mem_cleanup(hst_m);
	heap_mem_cleanup(hst_plan);
	heap_mem_cleanup(hst_stage2_plan);

	ul_test_clear(&test_state_s1);
	ul_test_clear(&test_state_r);
	ul_test_clear(&test_state_s2);
	return ret;
}

