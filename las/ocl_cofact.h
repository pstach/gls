/*
 * ocl_cofact.h
 *
 *  Created on: January 28, 2014
 *      Author: tcarstens
 */

#define USE_OPENCL 1
#define PRINTF_PROGRESS_REPORT 0

#if USE_OPENCL
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <CL/opencl.h>

#include "las.h"
#include "cofact.h"
#include "ocl_types.h"

typedef struct ocl_state_s {
	cl_uint num_platforms;					/* Number of platforms in cpPlatforms */
	cl_platform_id *cpPlatforms;			/* Array of all platform IDs */
	cl_platform_id cpPlatform;				/* The platform ID we're using */
	size_t len_platform_version;			/* Length of the platform version str */
	char *szPlatformVersion;				/* Version string for the platform
											 * we're using */

	cl_uint num_devices;					/* Number of devices in cdDevices */
	cl_device_id *cdDevices;				/* Array of all device IDs (for the
											 * platform we're using) */
	cl_device_id cdDevice;					/* The device ID we're using */
	size_t len_device_name;					/* Length of the device name str */
	char *szDeviceName;						/* Name string for the device we're
											 * using */
	size_t max_work_group_size;				/* CL_DEVICE_MAX_WORK_GROUP_SIZE */
	cl_uint max_work_item_dimensions;		/* CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS */
	size_t *max_work_item_sizes;			/* CL_DEVICE_MAX_WORK_ITEM_SIZES */

	cl_context cxGPUContext;
	cl_command_queue cqCommandQueue;
	const char *buildopts;
	cl_program cpProgram;
} ocl_state_t;

extern ocl_state_t ocl_state;

#define OCL_BUILDOPT_NVIDIA ((unsigned int)(1 << 0))

void ocl_init(ocl_state_t *ocl_state, const cl_uint config_platform, const cl_uint config_device, const unsigned int config_buildopts);
void ocl_close(ocl_state_t *ocl_state);
void ocl_build(ocl_state_t *ocl_state, const char *config_mp_source, const char *config_build_opts);


typedef struct ocl_kern_state_s {
	cl_kernel ckKernel;						/* Our kernel */
	size_t kernel_work_group_size;			/* CL_KERNEL_WORK_GROUP_SIZE */
	struct timespec start;					/* Start time */
	struct timespec stop;					/* Stop time */

	size_t local_work_size;
	size_t global_work_size;
} ocl_kern_state_t;

int ocl_kern_setup(ocl_state_t *ocl_state, const char *szKernName, ocl_kern_state_t *kern_state, size_t desired_global_work_size);
void ocl_kern_clear(ocl_kern_state_t *kern_state);
int ocl_kern_run(ocl_state_t *ocl_state, ocl_kern_state_t *kern_state);


void pm1_ul32_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void pm1_ul64_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void pm1_ul96_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void pm1_ul128_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void pm1_ul160_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void pm1_ul192_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void pm1_ul224_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void pm1_ul256_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);

void pp1_ul32_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void pp1_ul64_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void pp1_ul96_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void pp1_ul128_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void pp1_ul160_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void pp1_ul192_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void pp1_ul224_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void pp1_ul256_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);

void ecm_ul32_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void ecm_ul64_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void ecm_ul96_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void ecm_ul128_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void ecm_ul160_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void ecm_ul192_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void ecm_ul224_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
void ecm_ul256_process_ocl(cofact_algo_t *algo, candidate_t **batch, int n_batch);
#endif /* USE_OPENCL */
