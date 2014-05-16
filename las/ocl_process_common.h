/*
 * ocl_process_common.h
 *
 *  Created on: January 28, 2014
 *      Author: tcarstens
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cofact.h"
#include "ocl_cofact.h"

#if USE_OPENCL
#include "cofact_plan.h"


#define COMPARE_OCL_HOST 0
#define PRINTF_FACTOR_FOUND 0


#if COMPARE_OCL_HOST
#include "pm1.h"
#include "pp1.h"
#include "ecm.h"
#endif


void pm1_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch)
{
#if PRINTF_PROGRESS_REPORT
    printf("pm1_process, %d-bit, algo_idx=%d, n_batch=%lu\n", BITSIZE, algo->algo_idx, n_batch);
#endif
    if (n_batch == 0) {
#if PRINTF_PROGRESS_REPORT
        printf("n_batch = 0, skipping\n");
#endif
        return;
    }

    int ret = 0;

	pm1_plan_t *plan = (pm1_plan_t *) algo->plan;
	ocl_kern_state_t kern_state_s1 = { 0 };
	ocl_kern_state_t kern_state_s2 = { 0 };

    cl_int err = CL_SUCCESS;
    double tin, tout;

	tin = dbltime();

    /* Create our kernels */
    {
	    ret = ocl_kern_setup(&ocl_state, kern_pm1_stage1, &kern_state_s1, n_batch);
	    if (ret)
		    goto CLEANUP;

	    ret = ocl_kern_setup(&ocl_state, kern_pm1_stage2, &kern_state_s2, n_batch);
	    if (ret)
		    goto CLEANUP;

#if PRINTF_PROGRESS_REPORT
        printf("s1: local=%d, global=%d\n", kern_state_s1.local_work_size, kern_state_s1.global_work_size);
        printf("s2: local=%d, global=%d\n", kern_state_s2.local_work_size, kern_state_s2.global_work_size);
#endif
    }

    /* Create our buffers */
    cl_mem dev_f = 0;
    cl_mem dev_X = 0;
    cl_mem dev_n = 0;
    cl_mem dev_s1_plan = 0;
    cl_mem dev_s1_E = 0;
    cl_mem dev_s2_plan = 0;
    cl_mem dev_s2_S1 = 0;
    cl_mem dev_s2_pairs = 0;
    ul_ocl *hst_f = 0;
    ul_ocl *hst_n = 0;

    {
        dev_f = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_WRITE, sizeof(ul_ocl) * n_batch, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }

        dev_X = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_WRITE, sizeof(ul_ocl) * n_batch, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }

        dev_n = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_WRITE, sizeof(ul_ocl) * n_batch, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }

        dev_s1_plan = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(ocl_pm1_plan_t), NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }
        dev_s1_E = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(u_int64_t) * plan->E_n_words, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }

        dev_s2_plan = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(ocl_stage2_plan_t), NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }
        dev_s2_S1 = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(u_int32_t) * plan->stage2.n_S1, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }
        dev_s2_pairs = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(u_int8_t) * plan->stage2.n_pairs, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }

        hst_f = (ul_ocl *)malloc(sizeof(ul_ocl) * n_batch);
        if (!hst_f) { fprintf(stderr, "Could not malloc hst_f %s:%d\n", __FILE__, __LINE__); exit(-1); }
        hst_n = (ul_ocl *)malloc(sizeof(ul_ocl) * n_batch);
        if (!hst_n) { fprintf(stderr, "Could not malloc hst_n %s:%d\n", __FILE__, __LINE__); exit(-1); }
    }

    /* Copy the plan and moduli to the device */
    {
        ocl_pm1_plan_t hst_s1_plan = { 0 };

		hst_s1_plan.B1 = plan->B1;
		hst_s1_plan.B2 = plan->stage2.B2;
		hst_s1_plan.exp2 = plan->exp2;
		hst_s1_plan.E_mask = plan->E_mask;
		hst_s1_plan.E_n_words = plan->E_n_words;


        ocl_stage2_plan_t hst_s2_plan = { 0 };

		hst_s2_plan.B2 = plan->stage2.B2;
		hst_s2_plan.d = plan->stage2.d;
		hst_s2_plan.i0 = plan->stage2.i0;
		hst_s2_plan.i1 = plan->stage2.i1;
		hst_s2_plan.n_S1 = plan->stage2.n_S1;
        hst_s2_plan.n_pairs = plan->stage2.n_pairs;

        int i = 0;
        for (i = 0; i < n_batch; i++) {
            candidate_t *cand = batch[i];
            mpz_get_ul_ocl(&hst_n[i], cand->rem[cand->side]);
        }

		err = clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
				dev_s1_plan, CL_TRUE, 0, sizeof(ocl_pm1_plan_t),
				&hst_s1_plan, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
				dev_s1_E, CL_TRUE, 0, sizeof(u_int64_t) * plan->E_n_words,
				plan->E, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
				dev_s2_plan, CL_TRUE, 0, sizeof(ocl_stage2_plan_t),
				&hst_s2_plan, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
				dev_s2_S1, CL_TRUE, 0, sizeof(u_int32_t) * plan->stage2.n_S1,
				plan->stage2.S1, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
				dev_s2_pairs, CL_TRUE, 0, sizeof(u_int8_t) * plan->stage2.n_pairs,
				plan->stage2.pairs, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
                dev_n, CL_TRUE, 0, sizeof(ul_ocl) * n_batch,
                hst_n, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "clEnqueueWriteBuffer error %.8x\n", err); exit(-1); }
    }

	/* Run kernels, fetch results */
	{
		/* stage 1*/
		{
			int arg = 0;
			err  = clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_f);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_X);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_n);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_s1_plan);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(ocl_pm1_plan_t), NULL);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_s1_E);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(u_int64_t) * plan->E_n_words, NULL);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_int), (void*)&n_batch);
			if (CL_SUCCESS != err) { fprintf(stderr, "Error %.8x in clSetKernelArg at %s:%d\n", err, __FILE__, __LINE__); exit(-1); }

#if PRINTF_PROGRESS_REPORT
            printf("...running stage 1\n");
#endif
			ret = ocl_kern_run(&ocl_state, &kern_state_s1);
			if (ret) { goto CLEANUP; }
            ocl_kern_clear(&kern_state_s1);
		}

		/* stage 2 */
		{
			int arg = 0;
			err = clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_f);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_X);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_n);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_s2_plan);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(ocl_stage2_plan_t), NULL);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_s2_S1);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(u_int32_t) * plan->stage2.n_S1, NULL);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_s2_pairs);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(u_int8_t) * plan->stage2.n_pairs, NULL);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_int), (void*)&n_batch);
			if (CL_SUCCESS != err) { fprintf(stderr, "Error %.8x in clSetKernelArg at %s:%d\n", err, __FILE__, __LINE__); exit(-1); }

#if PRINTF_PROGRESS_REPORT
            printf("...running stage 2\n");
#endif
			ret = ocl_kern_run(&ocl_state, &kern_state_s2);
			if (ret) { goto CLEANUP; }
            ocl_kern_clear(&kern_state_s2);
		}

		/* fetch f */
		{
			err  = clEnqueueReadBuffer(ocl_state.cqCommandQueue, dev_f, CL_TRUE, 0, sizeof(ul_ocl) * n_batch, hst_f, 0, NULL, NULL);
			if (CL_SUCCESS != err) { fprintf(stderr, "Error %.8x in clEnqueueReadBuffer at %s:%d\n", err, __FILE__, __LINE__); exit(-1); }
		}
	}

    /* go through the f's and add viable candidates into the next bin */
#if COMPARE_OCL_HOST && ((BITSIZE == 64) || (BITSIZE == 128))
    int correct = 0;
#endif
    {
        int i = 0;
        int n_processed = 0;
        
        for (i = 0; i < n_batch; i++) {
            /* index into the batch bin */
            cl_int j = i; /* if you do reorder between stage 1 and stage 2, replace i with perm[i] or similar */

            n_processed++;

            /* Now we process the results */
            {
                candidate_t *cand = batch[j];

                mpz_t gf;
                mpz_init(gf);
                mpz_set_ul_ocl(gf, &hst_f[i]);

#if COMPARE_OCL_HOST && ((BITSIZE == 64) || (BITSIZE == 128))
                /* Compare results against host implementation */
                mpz_t gf_;
                {
                    mod n;
                    ul f, X;
                    
                    ul_init(f);
                    ul_init(X);
                    mod_init(n);
                    mpz_init(gf_);

                    mpz_get_ul(n->n, cand->rem[cand->side]);
                    mod_set(n, n->n);
                    ul_set_ui(X, 0);

                    pm1_stage1_ul(f, X, n, plan);
            		if(ul_cmp_ui(f, 1) != 0 && ul_cmp(f, n->n) != 0) {
                    }
                    else {
                        pm1_stage2_ul(f, X, n, &plan->stage2);
                    }
                    mpz_set_ul(gf_, f);
                    
                    ul_clear(f);
                    ul_clear(X);
                    mod_clear(n);
                }
                if (mpz_cmp(gf, gf_) != 0) {
                    gmp_printf("ERROR: pm1 i=%d disagree: ocl: %Zd, hst: %Zd, n=%Zd\n", i, gf, gf_, cand->rem[cand->side]);
                }
                else
                    correct++;
                
                mpz_clear(gf_);
#endif

		        if(mpz_cmp_ui(gf, 1) != 0 && mpz_cmp(gf, cand->rem[cand->side]) != 0)
		        {
			        // factor found in stage 1 or stage2
#if PRINTF_FACTOR_FOUND
                    gmp_printf("factor found: %Zd (mod %Zd)\n", gf, cand->rem[cand->side]);
#endif
			        candidate_add_factor_mpz(cand, cand->side, gf);
		        }
		        cofact_next_algo(cand, algo->algo_idx + 1);

                mpz_clear(gf);
            }
        }
    }

#if PRINTF_PROGRESS_REPORT
    {
        double elapse_s1 = (double)(kern_state_s1.stop.tv_sec - kern_state_s1.start.tv_sec) +
                            ((double)(kern_state_s1.stop.tv_nsec - kern_state_s1.start.tv_nsec))/1000000000;
        double elapse_s2 = (double)(kern_state_s2.stop.tv_sec - kern_state_s2.start.tv_sec) +
                            ((double)(kern_state_s2.stop.tv_nsec - kern_state_s2.start.tv_nsec))/1000000000;

        printf("pm1 process complete\n");
        printf("    stage1:  wall=%f, ops/s=%e\n", elapse_s1, kern_state_s1.global_work_size / elapse_s1);
        printf("    stage2:  wall=%f, ops/s=%e\n", elapse_s2, kern_state_s2.global_work_size / elapse_s2);
#if COMPARE_OCL_HOST && ((BITSIZE == 64) || (BITSIZE == 128))
        printf("  %d of %d verified against the host impl\n", correct, n_batch);
#endif
    }
#endif


CLEANUP:
    if (dev_f) { clReleaseMemObject(dev_f); dev_f = 0; }
    if (dev_X) { clReleaseMemObject(dev_X); dev_X = 0; }
    if (dev_n) { clReleaseMemObject(dev_n); dev_n = 0; }
    if (dev_s1_plan) { clReleaseMemObject(dev_s1_plan); dev_s1_plan = 0; }
    if (dev_s1_E) { clReleaseMemObject(dev_s1_E); dev_s1_E = 0; }
    if (dev_s2_plan) { clReleaseMemObject(dev_s2_plan); dev_s2_plan = 0; }
    if (dev_s2_S1) { clReleaseMemObject(dev_s2_S1); dev_s2_S1 = 0; }
    if (dev_s2_pairs) { clReleaseMemObject(dev_s2_pairs); dev_s2_pairs = 0; }

    if (hst_n) { free(hst_n); hst_n = 0; }
    if (hst_f) { free(hst_f); hst_f = 0; }

	tout = dbltime();
	pm1_total += (tout - tin);
}


void pp1_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch)
{
#if PRINTF_PROGRESS_REPORT
    printf("pp1_process, %d-bit, algo_idx=%d, n_batch=%lu\n", BITSIZE, algo->algo_idx, n_batch);
#endif
    if (n_batch == 0) {
#if PRINTF_PROGRESS_REPORT
        printf("n_batch = 0, skipping\n");
#endif
        return;
    }


    int ret = 0;

	pp1_plan_t *plan = (pp1_plan_t *) algo->plan;
	ocl_kern_state_t kern_state_s1 = { 0 };
	ocl_kern_state_t kern_state_s2 = { 0 };

    cl_int err = CL_SUCCESS;
    double tin, tout;

	tin = dbltime();

    /* Create our kernels */
    {
	    ret = ocl_kern_setup(&ocl_state, kern_pp1_stage1, &kern_state_s1, n_batch);
	    if (ret)
		    goto CLEANUP;

	    ret = ocl_kern_setup(&ocl_state, kern_pp1_stage2, &kern_state_s2, n_batch);
	    if (ret)
		    goto CLEANUP;

#if PRINTF_PROGRESS_REPORT
        printf("s1: local=%d, global=%d\n", kern_state_s1.local_work_size, kern_state_s1.global_work_size);
        printf("s2: local=%d, global=%d\n", kern_state_s2.local_work_size, kern_state_s2.global_work_size);
#endif
    }

    /* Create our buffers */
    cl_mem dev_f = 0;
    cl_mem dev_X = 0;
    cl_mem dev_n = 0;
    cl_mem dev_s1_plan = 0;
    cl_mem dev_s1_bc = 0;
    cl_mem dev_s2_plan = 0;
    cl_mem dev_s2_S1 = 0;
    cl_mem dev_s2_pairs = 0;
    ul_ocl *hst_f = 0;
    ul_ocl *hst_n = 0;

    {
        dev_f = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_WRITE, sizeof(ul_ocl) * n_batch, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }

        dev_X = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_WRITE, sizeof(ul_ocl) * n_batch, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }

        dev_n = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_WRITE, sizeof(ul_ocl) * n_batch, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }

        dev_s1_plan = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(ocl_pp1_plan_t), NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }
        dev_s1_bc = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(u_int8_t) * plan->bc_len, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }

        dev_s2_plan = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(ocl_stage2_plan_t), NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }
        dev_s2_S1 = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(u_int32_t) * plan->stage2.n_S1, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }
        dev_s2_pairs = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(u_int8_t) * plan->stage2.n_pairs, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }

        hst_f = (ul_ocl *)malloc(sizeof(ul_ocl) * n_batch);
        if (!hst_f) { fprintf(stderr, "Could not malloc hst_f %s:%d\n", __FILE__, __LINE__); exit(-1); }
        hst_n = (ul_ocl *)malloc(sizeof(ul_ocl) * n_batch);
        if (!hst_n) { fprintf(stderr, "Could not malloc hst_n %s:%d\n", __FILE__, __LINE__); exit(-1); }
    }

    /* Copy the plan and moduli to the device */
    {
        ocl_pp1_plan_t hst_s1_plan = { 0 };

		hst_s1_plan.B1 = plan->B1;
		hst_s1_plan.bc_len = plan->bc_len;
		hst_s1_plan.exp2 = plan->exp2;

        ocl_stage2_plan_t hst_s2_plan = { 0 };

		hst_s2_plan.B2 = plan->stage2.B2;
		hst_s2_plan.d = plan->stage2.d;
		hst_s2_plan.i0 = plan->stage2.i0;
		hst_s2_plan.i1 = plan->stage2.i1;
		hst_s2_plan.n_S1 = plan->stage2.n_S1;
        hst_s2_plan.n_pairs = plan->stage2.n_pairs;

        int i = 0;
        for (i = 0; i < n_batch; i++) {
            candidate_t *cand = batch[i];
            mpz_get_ul_ocl(&hst_n[i], cand->rem[cand->side]);
        }

		err = clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
				dev_s1_plan, CL_TRUE, 0, sizeof(ocl_pp1_plan_t),
				&hst_s1_plan, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
				dev_s1_bc, CL_TRUE, 0, sizeof(u_int8_t) * plan->bc_len,
				plan->bc, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
				dev_s2_plan, CL_TRUE, 0, sizeof(ocl_stage2_plan_t),
				&hst_s2_plan, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
				dev_s2_S1, CL_TRUE, 0, sizeof(u_int32_t) * plan->stage2.n_S1,
				plan->stage2.S1, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
				dev_s2_pairs, CL_TRUE, 0, sizeof(u_int8_t) * plan->stage2.n_pairs,
				plan->stage2.pairs, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
                dev_n, CL_TRUE, 0, sizeof(ul_ocl) * n_batch,
                hst_n, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "clEnqueueWriteBuffer error %.8x\n", err); exit(-1); }
    }

	/* Run kernels, fetch results */
	{
		/* stage 1*/
		{
			int arg = 0;
			err  = clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_f);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_X);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_n);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_s1_plan);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(ocl_pp1_plan_t), NULL);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_s1_bc);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(u_int8_t) * plan->bc_len, NULL);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_int), (void*)&n_batch);
			if (CL_SUCCESS != err) { fprintf(stderr, "Error %.8x in clSetKernelArg at %s:%d\n", err, __FILE__, __LINE__); exit(-1); }

#if PRINTF_PROGRESS_REPORT
            printf("...running stage 1\n");
#endif
			ret = ocl_kern_run(&ocl_state, &kern_state_s1);
			if (ret) { goto CLEANUP; }
            ocl_kern_clear(&kern_state_s1);
		}

		/* stage 2 */
		{
			int arg = 0;
			err =  clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_f);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_X);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_n);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_s2_plan);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(ocl_stage2_plan_t), NULL);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_s2_S1);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(u_int32_t) * plan->stage2.n_S1, NULL);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_s2_pairs);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(u_int8_t) * plan->stage2.n_pairs, NULL);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_int), (void*)&n_batch);
			if (CL_SUCCESS != err) { fprintf(stderr, "Error %.8x in clSetKernelArg at %s:%d\n", err, __FILE__, __LINE__); exit(-1); }

#if PRINTF_PROGRESS_REPORT
            printf("...running stage 2\n");
#endif
			ret = ocl_kern_run(&ocl_state, &kern_state_s2);
			if (ret) { goto CLEANUP; }
            ocl_kern_clear(&kern_state_s2);
		}

		/* fetch f */
		{
			err  = clEnqueueReadBuffer(ocl_state.cqCommandQueue, dev_f, CL_TRUE, 0, sizeof(ul_ocl) * n_batch, hst_f, 0, NULL, NULL);
			if (CL_SUCCESS != err) { fprintf(stderr, "Error %.8x in clEnqueueReadBuffer at %s:%d\n", err, __FILE__, __LINE__); exit(-1); }
		}
	}

    /* go through the f's and add viable candidates into the next bin */
#if COMPARE_OCL_HOST && ((BITSIZE == 64) || (BITSIZE == 128))
    int correct = 0;
#endif
    {
        int i = 0;
        int n_processed = 0;

        for (i = 0; i < n_batch; i++) {
            /* index into the batch bin */
            cl_int j = i; /* if you reorder between stage 1 and stage 2, adjust here (maybe s/i/perm[i]/ */

            n_processed++;

            /* Now we process the results */
            {
                candidate_t *cand = batch[j];

                mpz_t gf;
                mpz_init(gf);
                mpz_set_ul_ocl(gf, &hst_f[i]);

#if COMPARE_OCL_HOST && ((BITSIZE == 64) || (BITSIZE == 128))
                /* Compare results against host implementation */
                mpz_t gf_;
                {
                    mod n;
                    ul f, X;
                    
                    ul_init(f);
                    ul_init(X);
                    mod_init(n);
                    mpz_init(gf_);

                    mpz_get_ul(n->n, cand->rem[cand->side]);
                    mod_set(n, n->n);
                    ul_set_ui(X, 0);

                    pp1_stage1_ul(f, X, n, plan);
            		if(ul_cmp_ui(f, 1) != 0 && ul_cmp(f, n->n) != 0) {
                    }
                    else {
                        pp1_stage2_ul(f, X, n, &plan->stage2);
                    }
                    mpz_set_ul(gf_, f);
                    
                    ul_clear(f);
                    ul_clear(X);
                    mod_clear(n);
                }
                if (mpz_cmp(gf, gf_) != 0) {
                    gmp_printf("ERROR: pp1 i=%d disagree: ocl: %Zd, hst: %Zd, n=%Zd\n", i, gf, gf_, cand->rem[cand->side]);
                }
                else
                    correct++;
                
                mpz_clear(gf_);
#endif

		        if(mpz_cmp_ui(gf, 1) != 0 && mpz_cmp(gf, cand->rem[cand->side]) != 0)
		        {
			        // factor found in stage 1 or stage2
#if PRINTF_FACTOR_FOUND
                    gmp_printf("factor found: %Zd (mod %Zd)\n", gf, cand->rem[cand->side]);
#endif
			        candidate_add_factor_mpz(cand, cand->side, gf);
		        }
		        cofact_next_algo(cand, algo->algo_idx + 1);

                mpz_clear(gf);
            }
        }
    }

#if PRINTF_PROGRESS_REPORT
    {
        double elapse_s1 = (double)(kern_state_s1.stop.tv_sec - kern_state_s1.start.tv_sec) +
                            ((double)(kern_state_s1.stop.tv_nsec - kern_state_s1.start.tv_nsec))/1000000000;
        double elapse_s2 = (double)(kern_state_s2.stop.tv_sec - kern_state_s2.start.tv_sec) +
                            ((double)(kern_state_s2.stop.tv_nsec - kern_state_s2.start.tv_nsec))/1000000000;

        printf("pp1 process complete\n");
        printf("    stage1:  wall=%f, ops/s=%e\n", elapse_s1, kern_state_s1.global_work_size / elapse_s1);
        printf("    stage2:  wall=%f, ops/s=%e\n", elapse_s2, kern_state_s2.global_work_size / elapse_s2);
#if COMPARE_OCL_HOST && ((BITSIZE == 64) || (BITSIZE == 128))
        printf("  %d of %d verified against the host impl\n", correct, n_batch);
#endif
    }
#endif


CLEANUP:
    if (dev_f) { clReleaseMemObject(dev_f); dev_f = 0; }
    if (dev_X) { clReleaseMemObject(dev_X); dev_X = 0; }
    if (dev_n) { clReleaseMemObject(dev_n); dev_n = 0; }
    if (dev_s1_plan) { clReleaseMemObject(dev_s1_plan); dev_s1_plan = 0; }
    if (dev_s1_bc) { clReleaseMemObject(dev_s1_bc); dev_s1_bc = 0; }
    if (dev_s2_plan) { clReleaseMemObject(dev_s2_plan); dev_s2_plan = 0; }
    if (dev_s2_S1) { clReleaseMemObject(dev_s2_S1); dev_s2_S1 = 0; }
    if (dev_s2_pairs) { clReleaseMemObject(dev_s2_pairs); dev_s2_pairs = 0; }

    if (hst_n) { free(hst_n); hst_n = 0; }
    if (hst_f) { free(hst_f); hst_f = 0; }

	tout = dbltime();
	pp1_total += (tout - tin);
}



typedef struct
{
	ul_ocl x;
	ul_ocl z;
} ellM_point_ul_ocl;


void ecm_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch)
{
#if PRINTF_PROGRESS_REPORT
    printf("ecm_process, %d-bit, algo_idx=%d, n_batch=%lu\n", BITSIZE, algo->algo_idx, n_batch);
#endif
    if (n_batch == 0) {
#if PRINTF_PROGRESS_REPORT
        printf("n_batch = 0, skipping\n");
#endif
        return;
    }
    
    int ret = 0;

	ecm_plan_t *plan = (ecm_plan_t *) algo->plan;
	ocl_kern_state_t kern_state_s1 = { 0 };
	ocl_kern_state_t kern_state_s2 = { 0 };

    cl_int err = CL_SUCCESS;
    double tin, tout;

	tin = dbltime();

    /* Create our kernels */
    {
	    ret = ocl_kern_setup(&ocl_state, kern_ecm_stage1, &kern_state_s1, n_batch);
	    if (ret)
		    goto CLEANUP;

	    ret = ocl_kern_setup(&ocl_state, kern_ecm_stage2, &kern_state_s2, n_batch);
	    if (ret)
		    goto CLEANUP;

#if PRINTF_PROGRESS_REPORT
        printf("s1: local=%d, global=%d\n", kern_state_s1.local_work_size, kern_state_s1.global_work_size);
        printf("s2: local=%d, global=%d\n", kern_state_s2.local_work_size, kern_state_s2.global_work_size);
#endif
    }

    /* Create our buffers */
    cl_mem dev_f = 0;
    cl_mem dev_X = 0;
    cl_mem dev_b = 0;
    cl_mem dev_n = 0;
    cl_mem dev_s1_plan = 0;
    cl_mem dev_s1_bc = 0;
    cl_mem dev_s2_plan = 0;
    cl_mem dev_s2_S1 = 0;
    cl_mem dev_s2_pairs = 0;
    ul_ocl *hst_f = 0;
    ul_ocl *hst_n = 0;

    {
        dev_f = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_WRITE, sizeof(ul_ocl) * n_batch, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }

        dev_X = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_WRITE, sizeof(ellM_point_ul_ocl) * n_batch, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }

        dev_b = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_WRITE, sizeof(ul_ocl) * n_batch, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }

        dev_n = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_WRITE, sizeof(ul_ocl) * n_batch, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }

        dev_s1_plan = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(ocl_ecm_plan_t), NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }
        dev_s1_bc = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(u_int8_t) * plan->bc_len, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }

        dev_s2_plan = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(ocl_stage2_plan_t), NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }
        dev_s2_S1 = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(u_int32_t) * plan->stage2.n_S1, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }
        dev_s2_pairs = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(u_int8_t) * plan->stage2.n_pairs, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x\n", err); exit(-1); }

        hst_f = (ul_ocl *)malloc(sizeof(ul_ocl) * n_batch);
        if (!hst_f) { fprintf(stderr, "Could not malloc hst_f %s:%d\n", __FILE__, __LINE__); exit(-1); }
        hst_n = (ul_ocl *)malloc(sizeof(ul_ocl) * n_batch);
        if (!hst_n) { fprintf(stderr, "Could not malloc hst_n %s:%d\n", __FILE__, __LINE__); exit(-1); }
    }

    /* Copy the plan and moduli to the device */
    {
        ocl_ecm_plan_t hst_s1_plan = { 0 };

        hst_s1_plan.B1 = plan->B1;
        hst_s1_plan.bc_len = plan->bc_len;
        hst_s1_plan.exp2 = plan->exp2;
        hst_s1_plan.sigma = plan->sigma;
        hst_s1_plan.parameterization = plan->parameterization;


        ocl_stage2_plan_t hst_s2_plan = { 0 };

		hst_s2_plan.B2 = plan->stage2.B2;
		hst_s2_plan.d = plan->stage2.d;
		hst_s2_plan.i0 = plan->stage2.i0;
		hst_s2_plan.i1 = plan->stage2.i1;
		hst_s2_plan.n_S1 = plan->stage2.n_S1;
        hst_s2_plan.n_pairs = plan->stage2.n_pairs;

        int i = 0;
        for (i = 0; i < n_batch; i++) {
            candidate_t *cand = batch[i];
            mpz_get_ul_ocl(&hst_n[i], cand->rem[cand->side]);
        }

		err = clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
				dev_s1_plan, CL_TRUE, 0, sizeof(ocl_ecm_plan_t),
				&hst_s1_plan, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
				dev_s1_bc, CL_TRUE, 0, sizeof(u_int8_t) * plan->bc_len,
				plan->bc, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
				dev_s2_plan, CL_TRUE, 0, sizeof(ocl_stage2_plan_t),
				&hst_s2_plan, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
				dev_s2_S1, CL_TRUE, 0, sizeof(u_int32_t) * plan->stage2.n_S1,
				plan->stage2.S1, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
				dev_s2_pairs, CL_TRUE, 0, sizeof(u_int8_t) * plan->stage2.n_pairs,
				plan->stage2.pairs, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
                dev_n, CL_TRUE, 0, sizeof(ul_ocl) * n_batch,
                hst_n, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "clEnqueueWriteBuffer error %.8x\n", err); exit(-1); }
    }

	/* Run kernels, fetch results */
	{
		/* stage 1*/
		{
			int arg = 0;
			err  = clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_f);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_X);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_b);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_n);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_s1_plan);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(ocl_ecm_plan_t), NULL);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_s1_bc);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(u_int8_t) * plan->bc_len, NULL);
			err |= clSetKernelArg(kern_state_s1.ckKernel, arg++, sizeof(cl_int), (void*)&n_batch);
			if (CL_SUCCESS != err) { fprintf(stderr, "Error %.8x in clSetKernelArg at %s:%d\n", err, __FILE__, __LINE__); exit(-1); }

#if PRINTF_PROGRESS_REPORT
            printf("...running stage 1\n");
#endif
			ret = ocl_kern_run(&ocl_state, &kern_state_s1);
			if (ret) { goto CLEANUP; }
            ocl_kern_clear(&kern_state_s1);
		}

		/* stage 2 */
		{
			int arg = 0;
			err =  clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_f);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_X);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_b);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_n);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_s2_plan);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(ocl_stage2_plan_t), NULL);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_s2_S1);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(u_int32_t) * plan->stage2.n_S1, NULL);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_s2_pairs);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(u_int8_t) * plan->stage2.n_pairs, NULL);
			err |= clSetKernelArg(kern_state_s2.ckKernel, arg++, sizeof(cl_int), (void*)&n_batch);
			if (CL_SUCCESS != err) { fprintf(stderr, "Error %.8x in clSetKernelArg at %s:%d\n", err, __FILE__, __LINE__); exit(-1); }

#if PRINTF_PROGRESS_REPORT
            printf("...running stage 2\n");
#endif
			ret = ocl_kern_run(&ocl_state, &kern_state_s2);
			if (ret) { goto CLEANUP; }
            ocl_kern_clear(&kern_state_s2);
		}

		/* fetch f */
		{
			err  = clEnqueueReadBuffer(ocl_state.cqCommandQueue, dev_f, CL_TRUE, 0, sizeof(ul_ocl) * n_batch, hst_f, 0, NULL, NULL);
			if (CL_SUCCESS != err) { fprintf(stderr, "Error %.8x in clEnqueueReadBuffer at %s:%d\n", err, __FILE__, __LINE__); exit(-1); }
		}
	}

    /* go through the f's and add viable candidates into the next bin */
#if COMPARE_OCL_HOST && ((BITSIZE == 64) || (BITSIZE == 128))
    int correct = 0;
#endif
    {
        int i = 0;
        int n_processed = 0;
        
        for (i = 0; i < n_batch; i++) {
            /* index into the batch bin */
            cl_int j = i; /* if you reorder between stage 1 and stage 2, adjust here by replacing i with something like perm[i] */

            n_processed++;

            /* Now we process the results */
            {
                candidate_t *cand = batch[j];

                mpz_t gf;
                mpz_init(gf);
                mpz_set_ul_ocl(gf, &hst_f[i]);

#if COMPARE_OCL_HOST && ((BITSIZE == 64) || (BITSIZE == 128))
                /* Compare results against host implementation */
                mpz_t gf_;
                {
                    mod n;
                    ul f, b;
                	ellM_point_t X;
                    
                    ul_init(f);
                    ul_init(b);
                    ul_init(X->x);
                    ul_init(X->z);
                    mod_init(n);
                    mpz_init(gf_);

                    mpz_get_ul(n->n, cand->rem[cand->side]);
                    mod_set(n, n->n);
                    ecm_stage1_ul(f, X, b, n, plan);
            		if(ul_cmp_ui(f, 1) != 0 && ul_cmp(f, n->n) != 0) {
                    }
                    else {
                        ecm_stage2_ul(f, X, b, n, &plan->stage2);
                    }
                    mpz_set_ul(gf_, f);
                    
                    ul_clear(f);
                    ul_clear(b);
                    ul_clear(X->x);
                    ul_clear(X->z);
                    mod_clear(n);
                }
                if (mpz_cmp(gf, gf_) != 0) {
                    gmp_printf("ERROR: ecm i=%d disagree: ocl: %Zd, hst: %Zd, n=%Zd\n", i, gf, gf_, cand->rem[cand->side]);
                }
                else
                    correct++;
                mpz_clear(gf_);
#endif

		        if(mpz_cmp_ui(gf, 1) != 0 && mpz_cmp(gf, cand->rem[cand->side]) != 0)
		        {
			        // factor found in stage 1 or stage2
#if PRINTF_FACTOR_FOUND
                    gmp_printf("factor found: %Zd (mod %Zd)\n", gf, cand->rem[cand->side]);
#endif
			        candidate_add_factor_mpz(cand, cand->side, gf);
		        }
		        cofact_next_algo(cand, algo->algo_idx + 1);

                mpz_clear(gf);
            }
        }
    }

#if PRINTF_PROGRESS_REPORT
    {
        double elapse_s1 = (double)(kern_state_s1.stop.tv_sec - kern_state_s1.start.tv_sec) +
                            ((double)(kern_state_s1.stop.tv_nsec - kern_state_s1.start.tv_nsec))/1000000000;
        double elapse_s2 = (double)(kern_state_s2.stop.tv_sec - kern_state_s2.start.tv_sec) +
                            ((double)(kern_state_s2.stop.tv_nsec - kern_state_s2.start.tv_nsec))/1000000000;

        printf("ecm process complete\n");
        printf("    stage1:  wall=%f, ops/s=%e\n", elapse_s1, kern_state_s1.global_work_size / elapse_s1);
        printf("    stage2:  wall=%f, ops/s=%e\n", elapse_s2, kern_state_s2.global_work_size / elapse_s2);
#if COMPARE_OCL_HOST && ((BITSIZE == 64) || (BITSIZE == 128))
        printf("  %d of %d verified against the host impl\n", correct, n_batch);
#endif
    }
#endif


CLEANUP:
    if (dev_f) { clReleaseMemObject(dev_f); dev_f = 0; }
    if (dev_b) { clReleaseMemObject(dev_b); dev_b = 0; }
    if (dev_X) { clReleaseMemObject(dev_X); dev_X = 0; }
    if (dev_n) { clReleaseMemObject(dev_n); dev_n = 0; }
    if (dev_s1_plan) { clReleaseMemObject(dev_s1_plan); dev_s1_plan = 0; }
    if (dev_s1_bc) { clReleaseMemObject(dev_s1_bc); dev_s1_bc = 0; }
    if (dev_s2_plan) { clReleaseMemObject(dev_s2_plan); dev_s2_plan = 0; }
    if (dev_s2_S1) { clReleaseMemObject(dev_s2_S1); dev_s2_S1 = 0; }
    if (dev_s2_pairs) { clReleaseMemObject(dev_s2_pairs); dev_s2_pairs = 0; }

    if (hst_n) { free(hst_n); hst_n = 0; }
    if (hst_f) { free(hst_f); hst_f = 0; }

	tout = dbltime();
	ecm_total += (tout - tin);
}
#endif /* USE_OPENCL */
