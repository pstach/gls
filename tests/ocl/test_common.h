/*
 * test_ulcommon.h
 *
 *  Created on: Nov 12, 2013
 *      Author: tcarstens
 */


#ifndef __TEST_COMMON__
#define __TEST_COMMON__

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <CL/opencl.h>
#include <gmp.h>


#include "ocl_error.h"
#include "state.h"
#include "tests.h"


/* Defines a new OCL mem object, tries to allocate it */
#define ocl_mem(mem, t, cxt, perms, global_size) cl_mem mem = 0; \
mem = clCreateBuffer(cxt, perms, sizeof(t) * global_size, NULL, &err); \
if (CL_SUCCESS != err) \
	FAIL_CL("clCreateBuffer", err);

/* Releases a mem object (if it is nonzero) */
#define ocl_mem_cleanup(mem) if (mem) { clReleaseMemObject(mem); mem = 0; }

/* Defines a hew host mem buffer, tries to allocate it */
#define heap_mem(mem, t, size) static t *mem = NULL; \
mem = (t *)malloc(sizeof(t) * size); \
if (!mem) { ret = -1; fprintf(stderr, "Error: failure to malloc %s\n", #mem); goto CLEANUP; }

/* Frees a host mem buffer (if it is nonnul) */
#define heap_mem_cleanup(mem) if (mem) { free(mem); mem = NULL; }


struct ul_test_state_t {
	cl_kernel ckKernel;						/* Our kernel */
	size_t kernel_work_group_size;			/* CL_KERNEL_WORK_GROUP_SIZE */
	struct timespec start;					/* Start time */
	struct timespec stop;					/* Stop time */

	size_t local_work_size;
	size_t global_work_size;
};

int ul_test_setup(struct state_t *state, const char *szKernName, struct ul_test_state_t *test_state, size_t desired_global_work_size) {
	int ret = 0;

	bzero(test_state, sizeof(*test_state));

	/* Create the kernel, obtain kernel_work_group_size */
	{
		cl_int err;

		test_state->ckKernel = clCreateKernel(state->cpProgram, szKernName, &err);
		if (CL_SUCCESS != err) {
			fprintf(stderr, "Error while creating kernel %s\n", szKernName);
			FAIL_CL("clCreateKernel", err);
		}

		err = clGetKernelWorkGroupInfo(test_state->ckKernel,
				state->cdDevice,
				CL_KERNEL_WORK_GROUP_SIZE,
				sizeof(test_state->kernel_work_group_size),
				&test_state->kernel_work_group_size,
				NULL);
		if (CL_SUCCESS != err)
			FAIL_CL("clGetKernelWorkGroupInfo", err);
	}

	/* Determine local work size */
	{
		/* The actual local work group size is also bounded by max_work_group_size and max_work_item_sizes */
		test_state->local_work_size = test_state->kernel_work_group_size;
        while (test_state->local_work_size > desired_global_work_size)
            test_state->local_work_size >>= 1;

		if (state->max_work_group_size < test_state->local_work_size)
			test_state->local_work_size = state->max_work_group_size;

		if (state->max_work_item_sizes[0] < test_state->local_work_size)
			test_state->local_work_size = state->max_work_item_sizes[0];
	}

	/* Determine global work size */
	{
		/* The global work group size must be the lowest multiple of local_work_size which is greater-than the desired global work size */
		test_state->global_work_size = desired_global_work_size;

		size_t rem = test_state->global_work_size % test_state->local_work_size;
		if (rem)
			test_state->global_work_size += test_state->local_work_size - rem;
	}

	CLEANUP:
	return ret;
}

void ul_test_clear(struct ul_test_state_t *test_state) {
	if (test_state->ckKernel) {
		clReleaseKernel(test_state->ckKernel);
		test_state->ckKernel = 0;
	}
}

int ul_test_run_kernel(struct state_t *state, struct ul_test_state_t *test_state) {
	int ret = 0;

	if (clock_gettime(CLOCK_REALTIME, &test_state->start) < 0) {
		perror("clock_gettime");
		ret = -1;
		goto CLEANUP;
	}

	cl_int err = 0;
	cl_event eKernel = 0;
	err = clEnqueueNDRangeKernel(state->cqCommandQueue,
			test_state->ckKernel,
			1,
			NULL,
			&test_state->global_work_size,
			&test_state->local_work_size,
			0,
			NULL,
			&eKernel);
	if (CL_SUCCESS != err)
		FAIL_CL("clEnqueueNDRangeKernel", err);
	err = clWaitForEvents(1, &eKernel);
	if (CL_SUCCESS != err)
		FAIL_CL("clWaitForEvents", err);
	clReleaseEvent(eKernel);

	if (clock_gettime(CLOCK_REALTIME, &test_state->stop) < 0) {
		perror("clock_gettime");
		ret = -1;
		goto CLEANUP;
	}

	CLEANUP:
	return ret;
}



#endif /* __TEST_COMMON__ */
