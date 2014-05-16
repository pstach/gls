/*
 * testmp.c
 *
 *  Created on: Nov 11, 2013
 *      Author: tcarstens
 */


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <CL/opencl.h>

#include "state.h"
#include "ocl_error.h"
#include "cofact_plan.h"

#include "tests.h"

#define max(a, b) ((a > b) ? a : b)

void ocl_handle_error_in_context(const char *errinfo, const void *private_info, size_t cb, void *user_data);

int main_test(const char *build_opts);

int main() {
	int ret = 0;

	/*
	 * This is where we set the test agenda. You can
	 * experiment with different flags (ie, to compare
	 * performance). Between tests, you can also make
	 * changes to the global variable g_config to
	 * change which platform or device you want to use
	 * in the next test.
	 */

	g_config.config_platform = 1;
	g_config.config_device = 0;
	main_test("-Werror -cl-mad-enable");
	// main_test("-Werror -cl-mad-enable -D UL_NVIDIA=1");

	return ret;
}

int main_test(const char *build_opts_in) {
	/* return code */
	int ret = 0;

	/* formulate cofactorization plans */
	{
		unsigned int seed = time(NULL);
		srandom(seed);

		pm1_plan_init(&g_config.pm1_stage1_plan, 315, 2205);
		pp1_plan_init(&g_config.pp1_stage1_plan, 525, 3255);
		ecm_plan_init(&g_config.ecm_stage1_plan, 105, 3255, MONTY12, 2);
	}

	char *build_opts = NULL;
	{
		int PP1_STAGE2_XJ_LEN = max(g_config.pm1_stage1_plan.stage2.n_S1, g_config.pp1_stage1_plan.stage2.n_S1);
		int ECM_COMMONZ_T_LEN = (g_config.ecm_stage1_plan.stage2.n_S1) + (g_config.ecm_stage1_plan.stage2.i1 - g_config.ecm_stage1_plan.stage2.i0 - ((g_config.ecm_stage1_plan.stage2.i0 == 0) ? 1 : 0));
		int ECM_STAGE2_PID_LEN = g_config.ecm_stage1_plan.stage2.i1 - g_config.ecm_stage1_plan.stage2.i0;
		int ECM_STAGE2_PJ_LEN = g_config.ecm_stage1_plan.stage2.n_S1;

		int build_opts_len =
				snprintf(NULL, 0, "%s -D TEST_SET=0x%x -D TESTOP_SET=0x%x -D PP1_STAGE2_XJ_LEN=%d -D ECM_COMMONZ_T_LEN=%d -D ECM_STAGE2_PID_LEN=%d -D ECM_STAGE2_PJ_LEN=%d",
						 build_opts_in,
						 TEST_SET,
						 TESTOP_SET,
						 PP1_STAGE2_XJ_LEN,
						 ECM_COMMONZ_T_LEN,
						 ECM_STAGE2_PID_LEN,
						 ECM_STAGE2_PJ_LEN);
		if (build_opts_len <= 0) {
			ret = -1;
			goto CLEANUP;
		}
		build_opts_len++; /* snprintf does not include null byte in ret value */
		build_opts = (char *)malloc(build_opts_len);
		if (!build_opts) {
			FAIL_MALLOC(build_opts)
			ret = -1;
			goto CLEANUP;
		}
		snprintf(build_opts, build_opts_len, "%s -D TEST_SET=0x%x -D TESTOP_SET=0x%x -D PP1_STAGE2_XJ_LEN=%d -D ECM_COMMONZ_T_LEN=%d -D ECM_STAGE2_PID_LEN=%d -D ECM_STAGE2_PJ_LEN=%d",
				 build_opts_in,
				 TEST_SET,
				 TESTOP_SET,
				 PP1_STAGE2_XJ_LEN,
				 ECM_COMMONZ_T_LEN,
				 ECM_STAGE2_PID_LEN,
				 ECM_STAGE2_PJ_LEN);
		printf("Testing with build_opts=%s\n", build_opts);
	}

	/* Test suite state */
	struct state_t state = { 0 };

	/* CUDA compiler cache makes it difficult to iterate on versions of the OCL kernels */
	setenv("CUDA_CACHE_DISABLE", "1", 1);

	/* Configure the platform */
	{
		cl_int err;

		/* Figure out how many platforms there are */
		err = clGetPlatformIDs(0, NULL, &state.num_platforms);
		if (CL_SUCCESS != err)
			FAIL_CL("clGetPlatformIDs", err);

		/* Verify that the desired platform falls within range of what the system has */
		if (!(g_config.config_platform < state.num_platforms)) {
			fprintf(stderr, "Error: config_platform = %u, which is outside of bounds (num_platforms = %u)\n", g_config.config_platform, state.num_platforms);
			ret = -1;
			goto CLEANUP;
		}

		/* Allocate a buffer to receive their platform IDs */
		state.cpPlatforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * state.num_platforms);
		if (!state.cpPlatforms)
			FAIL_MALLOC("state.cpPlatforms");

		/* Fetch the platform IDs */
		err = clGetPlatformIDs(state.num_platforms, state.cpPlatforms, NULL);
		if (CL_SUCCESS != err)
			FAIL_CL("clGetPlatformIDs", err);

		/* Fetch the desired platform ID */
		state.cpPlatform = state.cpPlatforms[g_config.config_platform];

		/* Get some information about the chosen platform */
		{
			/* Get the length of the version string */
			err = clGetPlatformInfo(state.cpPlatform, CL_PLATFORM_VERSION, 0, NULL, &state.len_platform_version);
			if (CL_SUCCESS != err)
				FAIL_CL("clGetPlatformInfo", err);

			/* Allocate a buffer for the version string */
			state.szPlatformVersion = (char *)malloc(state.len_platform_version);
			if (!state.szPlatformVersion)
				FAIL_MALLOC("state.szPlatformVersion");

			/* Fetch the version string */
			err = clGetPlatformInfo(state.cpPlatform, CL_PLATFORM_VERSION, state.len_platform_version, state.szPlatformVersion, NULL);
			if (CL_SUCCESS != err)
				FAIL_CL("clGetPlatformInfo", err);
		}
	}

	/* Configure the device */
	{
		cl_int err;

		/* Figure out how many devices there are */
		err = clGetDeviceIDs(state.cpPlatform, CL_DEVICE_TYPE_ALL, 0, NULL, &state.num_devices);
		if (CL_SUCCESS != err)
			FAIL_CL("clGetDeviceIDs", err);

		/* Verify that the desired device falls within range of what the system has */
		if (!(g_config.config_device < state.num_devices)) {
			fprintf(stderr, "Error: config_device = %u, which is outside of bounds (num_devices = %u)\n", g_config.config_device, state.num_devices);
			ret = -1;
			goto CLEANUP;
		}

		/* Allocate a buffer to receive their platform IDs */
		state.cdDevices = (cl_device_id*)malloc(sizeof(cl_device_id) * state.num_devices);
		if (!state.cdDevices)
			FAIL_MALLOC("cpDevices");

		/* Fetch the platform IDs */
		err = clGetDeviceIDs(state.cpPlatform, CL_DEVICE_TYPE_ALL, state.num_devices, state.cdDevices, NULL);
		if (CL_SUCCESS != err)
			FAIL_CL("clGetDeviceIDs", err);

		/* Fetch the desired platform ID */
		state.cdDevice = state.cdDevices[g_config.config_device];

		/* Fetch some information about the chosen device */
		{
			cl_int err;

			/* CL_DEVICE_NAME */
			{
				err = clGetDeviceInfo(state.cdDevice, CL_DEVICE_NAME, 0, NULL, &state.len_device_name);
				if (CL_SUCCESS != err)
					FAIL_CL("clGetDeviceInfo", err);
				state.szDeviceName = (char *)malloc(state.len_device_name);
				if (!state.szDeviceName)
					FAIL_MALLOC("state.szDeviceName");
				err = clGetDeviceInfo(state.cdDevice, CL_DEVICE_NAME, state.len_device_name, state.szDeviceName, NULL);
				if (CL_SUCCESS != err)
					FAIL_CL("clGetDeviceInfo", err);
			}
			/* CL_DEVICE_MAX_WORK_GROUP_SIZE */
			{
				err = clGetDeviceInfo(state.cdDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(state.max_work_group_size), &state.max_work_group_size, NULL);
				if (CL_SUCCESS != err)
					FAIL_CL("clGetDeviceInfo", err);
			}
			/* CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS */
			{
				err = clGetDeviceInfo(state.cdDevice, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(state.max_work_item_dimensions), &state.max_work_item_dimensions, NULL);
				if (CL_SUCCESS != err)
					FAIL_CL("clGetDeviceInfo", err);
			}
			/* CL_DEVICE_MAX_WORK_ITEM_SIZES */
			{
				size_t bufsize = sizeof(size_t) * state.max_work_item_dimensions;
				state.max_work_item_sizes = (size_t *)malloc(bufsize);
				if (!state.max_work_item_sizes)
					FAIL_MALLOC("max_work_item_sizes");
				err = clGetDeviceInfo(state.cdDevice, CL_DEVICE_MAX_WORK_ITEM_SIZES, bufsize, state.max_work_item_sizes, NULL);
				if (CL_SUCCESS != err)
					FAIL_CL("clGetDeviceInfo", err);
			}
		}
	}

	/* Print info about the platform & device */
	{
		printf("Testing on %s - %s\n", state.szPlatformVersion, state.szDeviceName);
	}

	/* Obtain a device context and a command queue */
	{
		cl_int err;

		state.cxGPUContext = clCreateContext(NULL, 1, &state.cdDevice, &ocl_handle_error_in_context, NULL, &err);
		if (CL_SUCCESS != err)
			FAIL_CL("clCreateContext", err);

		state.cqCommandQueue = clCreateCommandQueue(state.cxGPUContext, state.cdDevice, 0, &err);
		if (CL_SUCCESS != err)
			FAIL_CL("clCreateCommandQueue", err);
	}

	/* Load and compile our program */
	{
		struct timespec build_start;
		struct timespec build_stop;

		printf("Building...\n");

		if (clock_gettime(CLOCK_REALTIME, &build_start) < 0) {
			perror("clock_gettime");
			ret = -1;
			goto CLEANUP;
		}

		/* Read the source */
		{
			state.source.fSource = fopen(g_config.config_mp_source, "rb");
			if (!state.source.fSource) {
				perror("fopen");
				ret = -1;
				goto CLEANUP;
			}

			/* Determine the size of the source code */
			if (fseek(state.source.fSource, 0, SEEK_END) < 0) {
				perror("fseek");
				ret = -1;
				goto CLEANUP;
			}
			long lSize = ftell(state.source.fSource);
			if (lSize < 0) {
				perror("ftell");
				ret = -1;
				goto CLEANUP;
			}
			state.source.len_source = lSize; /* lSize >= 0 */
			if (fseek(state.source.fSource, 0, SEEK_SET) < 0) {
				perror("fseek");
				ret = -1;
				goto CLEANUP;
			}

			/* Allocate a buffer for the source code */
			state.source.szSource = (char *)malloc(state.source.len_source + 1); /* +1 for '\0' */
			if (!state.source.szSource)
				FAIL_MALLOC("szSource");

			/* Read the source, then '\0'-terminate */
			if (1 != fread(state.source.szSource, state.source.len_source, 1, state.source.fSource)) {
				perror("fread");
				ret = -1;
				goto CLEANUP;
			}
			state.source.szSource[state.source.len_source] = '\0';

			/* Close the file */
			fclose(state.source.fSource);
			state.source.fSource = NULL;
		}

		/* Create the program */
		{
			cl_int err;

			state.cpProgram = clCreateProgramWithSource(state.cxGPUContext, 1, (const char **)&state.source.szSource, &state.source.len_source, &err);
			if (CL_SUCCESS != err)
				FAIL_CL("clCreateProgramWithSource", err);

			err = clBuildProgram(state.cpProgram, 0, NULL, build_opts, NULL, NULL);
			if ((CL_SUCCESS == err) || (CL_BUILD_PROGRAM_FAILURE == err)) {
				cl_uint err2;

				/* Determine the size of the build log */
				err2 = clGetProgramBuildInfo(state.cpProgram, state.cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &state.source.len_build_log);
				if (CL_SUCCESS != err2)
					FAIL_CL("clGetProgramBuildInfo", err2);

				/* Allocate memory for the build log */
				state.source.szBuildLog = (char *)malloc(state.source.len_build_log);
				if (!state.source.szBuildLog)
					FAIL_MALLOC("state.source.szBuildLog");

				/* Fetch the build log */
				err2 = clGetProgramBuildInfo(state.cpProgram, state.cdDevice, CL_PROGRAM_BUILD_LOG, state.source.len_build_log, state.source.szBuildLog, NULL);
				if (CL_SUCCESS != err2)
					FAIL_CL("clGetProgramBuildInfo", err2);

				/* Print the build log */
				if (strlen(state.source.szBuildLog) > 2)
					fprintf((CL_SUCCESS != err) ? stderr : stdout, "%s\n\n", state.source.szBuildLog);

				/* Cleanup the build log */
				free(state.source.szBuildLog);
				state.source.szBuildLog = NULL;
			}

			if (CL_SUCCESS != err)
				FAIL_CL("clBuildProgram", err);

			/* Cleanup the source contents */
			free(state.source.szSource);
			state.source.szSource = NULL;
		}

#ifdef NVIDIA_OUTPUT_PTX
		/* Write program binary to disk */
		{
			cl_int err;

			err = clGetProgramInfo(state.cpProgram, CL_PROGRAM_BINARY_SIZES, sizeof(state.binary.len_binary), &state.binary.len_binary, NULL);
			if (CL_SUCCESS != err)
				FAIL_CL("clGetProgramInfo", err);

			if (0 == state.binary.len_binary) {
				fprintf(stderr, "len_binary = 0!\n");
				goto CLEANUP;
			}

			state.binary.binary = (unsigned char *)malloc(state.binary.len_binary);
			if (!state.binary.binary)
				FAIL_MALLOC("state.binary.binary");

			err = clGetProgramInfo(state.cpProgram, CL_PROGRAM_BINARIES, state.binary.len_binary, &state.binary.binary, NULL);
			if (CL_SUCCESS != err)
				FAIL_CL("clGetProgramInfo", err);

			state.binary.fBinary = fopen(g_config.config_mp_binary, "wb");
			if (!state.binary.fBinary) {
				perror("fopen (binary)");
				ret = -1;
				goto CLEANUP;
			}

			if (1 != fwrite(state.binary.binary, state.binary.len_binary, 1, state.binary.fBinary)) {
				perror("fwrite");
				ret = -1;
				goto CLEANUP;
			}

			fclose(state.binary.fBinary);
			state.binary.fBinary = NULL;
		}
#endif /* NVIDIA_OUTPUT_PTX */

		if (clock_gettime(CLOCK_REALTIME, &build_stop) < 0) {
			perror("clock_gettime");
			ret = -1;
			goto CLEANUP;
		}

		double elapse = (double)(build_stop.tv_sec - build_start.tv_sec) +
						((double)(build_stop.tv_nsec - build_start.tv_nsec))/1000000000;
		printf("Build complete, wall time %f s\n", elapse);
	}

	/* Now we can finally run the tests */
	{
		if (TEST_SET & TEST_32) {
			ret = ul32_test_all(&state);
			if (ret)
				goto CLEANUP;
		}

		if (TEST_SET & TEST_64) {
			ret = ul64_test_all(&state);
			if (ret)
				goto CLEANUP;
		}

		if (TEST_SET & TEST_96) {
			ret = ul96_test_all(&state);
			if (ret)
				goto CLEANUP;
		}

		if (TEST_SET & TEST_128) {
			ret = ul128_test_all(&state);
			if (ret)
				goto CLEANUP;
		}

		if (TEST_SET & TEST_160) {
			ret = ul160_test_all(&state);
			if (ret)
				goto CLEANUP;
		}

		if (TEST_SET & TEST_192) {
			ret = ul192_test_all(&state);
			if (ret)
				goto CLEANUP;
		}

		if (TEST_SET & TEST_224) {
			ret = ul224_test_all(&state);
			if (ret)
				goto CLEANUP;
		}

		if (TEST_SET & TEST_256) {
			ret = ul256_test_all(&state);
			if (ret)
				goto CLEANUP;
		}
	}

	/* Cleanup */
	CLEANUP:
	printf("Done.\n\n");


	pm1_plan_clear(&g_config.pm1_stage1_plan);
	pp1_plan_clear(&g_config.pp1_stage1_plan);
	ecm_plan_clear(&g_config.ecm_stage1_plan);


	if (state.cpPlatforms) {
		free(state.cpPlatforms);
		state.cpPlatforms = NULL;
	}
	if (state.cdDevices) {
		free(state.cdDevices);
		state.cdDevices = NULL;
	}
	if (state.szPlatformVersion) {
		free(state.szPlatformVersion);
		state.szPlatformVersion = NULL;
	}
	if (state.szDeviceName) {
		free(state.szDeviceName);
		state.szDeviceName = NULL;
	}
	if (state.max_work_item_sizes) {
		free(state.max_work_item_sizes);
		state.max_work_item_sizes = NULL;
	}

	if (state.source.fSource) {
		fclose(state.source.fSource);
		state.source.fSource = NULL;
	}
	if (state.source.szSource) {
		free(state.source.szSource);
		state.source.szSource = NULL;
	}
	if (state.source.szBuildLog) {
		free(state.source.szBuildLog);
		state.source.szBuildLog = NULL;
	}

#ifdef NVIDIA_OUTPUT_PTX
	if (state.binary.fBinary) {
		fclose(state.binary.fBinary);
		state.binary.fBinary = NULL;
	}
	if (state.binary.binary) {
		free(state.binary.binary);
		state.binary.binary = NULL;
	}
#endif /* NVIDIA_OUTPUT_PTX */

	if (state.cpProgram)
		clReleaseProgram(state.cpProgram);
	if (state.cqCommandQueue)
		clReleaseCommandQueue(state.cqCommandQueue);
	if (state.cxGPUContext)
		clReleaseContext(state.cxGPUContext);
	return ret;
}

void ocl_handle_error_in_context(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
	fprintf(stderr, "Error in compute context: %s\n", errinfo);
}
