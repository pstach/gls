/*
 * state.h
 *
 *  Created on: Nov 12, 2013
 *      Author: tcarstens
 */

#ifndef STATE_H_
#define STATE_H_

#include <stdio.h>
#include <time.h>
#include <CL/opencl.h>

#include "types.h"
#include "cofact_plan.h"


/* If defined and running on nVidia, will output the PTX
 * generated from our input source.
 */
// #define NVIDIA_OUTPUT_PTX


struct config_t {
	cl_uint config_platform;			    /* Index into the platforms array */
	cl_uint config_device;			        /* Index into the devices array */
	const char *config_mp_source;			/* File name of kernel source */
#ifdef NVIDIA_OUTPUT_PTX
	const char *config_mp_binary;			/* Destination file name for built program */
#endif /* NVIDIA_OUTPUT_PTX */
	size_t config_global_work_size;	        /* Desired global work group size.
	 	 	 	 	 	 	 	 	 	 	 * Will be rounded-up to a multiple
	 	 	 	 	 	 	 	 	 	 	 * of the local work group size. */
	pm1_plan_t pm1_stage1_plan;
	pp1_plan_t pp1_stage1_plan;
	ecm_plan_t ecm_stage1_plan;
};

/* Defined in config.c */
extern struct config_t g_config;

struct source_state_t {
	FILE *fSource;							/* Handle to source file */
	size_t len_source;						/* Length of source file */
	char *szSource;							/* Contents of source file */
	size_t len_build_log;					/* Length of build log */
	char *szBuildLog;						/* Contents of build log */
};

#ifdef NVIDIA_OUTPUT_PTX
struct binary_state_t {
	FILE *fBinary;							/* Handle to the binary-out file */
	size_t len_binary;						/* Length of the binary */
	unsigned char *binary;					/* The binary */
};
#endif /* NVIDIA_OUTPUT_PTX */

struct state_t {
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

	struct source_state_t source;			/* OCL Program source */
#ifdef NVIDIA_OUTPUT_PTX
	struct binary_state_t binary;			/* OCL Program binary */
#endif /* NVIDIA_OUTPUT_PTX */

	size_t max_work_group_size;				/* CL_DEVICE_MAX_WORK_GROUP_SIZE */
	cl_uint max_work_item_dimensions;		/* CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS */
	size_t *max_work_item_sizes;			/* CL_DEVICE_MAX_WORK_ITEM_SIZES */

	cl_context cxGPUContext;
	cl_command_queue cqCommandQueue;
	cl_program cpProgram;
};


#endif /* STATE_H_ */
