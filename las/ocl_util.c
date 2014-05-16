/*
 * ocl_util.c
 *
 *  Created on: January 28, 2014
 *      Author: tcarstens
 */
#include "ocl_cofact.h"


#if USE_OPENCL

ocl_state_t ocl_state = { 0 };


void ocl_init(ocl_state_t *ocl_state, const cl_uint config_platform, const cl_uint config_device, const unsigned int config_buildopts) {
    cl_int err = CL_SUCCESS;
    
    if (config_buildopts & OCL_BUILDOPT_NVIDIA) {
        ocl_state->buildopts = "-Werror -cl-mad-enable -D UL_NVIDIA=1";
    }
    else {
        ocl_state->buildopts = "-Werror -cl-mad-enable";
    }

    /* Configure the platform */
    {
        err = clGetPlatformIDs(0, NULL, &ocl_state->num_platforms);
        if (config_platform >= ocl_state->num_platforms) {
            fprintf(stderr, "config_platform = %d >= %d = num_platforms\n", config_platform, ocl_state->num_platforms);
            exit(-1);
        }

        ocl_state->cpPlatforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * ocl_state->num_platforms);
        err = clGetPlatformIDs(ocl_state->num_platforms, ocl_state->cpPlatforms, NULL);
	    ocl_state->cpPlatform = ocl_state->cpPlatforms[config_platform];
        
        err = clGetPlatformInfo(ocl_state->cpPlatform, CL_PLATFORM_VERSION, 0, NULL, &ocl_state->len_platform_version);
        ocl_state->szPlatformVersion = (char *)malloc(ocl_state->len_platform_version);
        err = clGetPlatformInfo(ocl_state->cpPlatform, CL_PLATFORM_VERSION, ocl_state->len_platform_version, ocl_state->szPlatformVersion, NULL);
    }

    /* Configure the device */
    {
		err = clGetDeviceIDs(ocl_state->cpPlatform, CL_DEVICE_TYPE_ALL, 0, NULL, &ocl_state->num_devices);

		if (config_device >= ocl_state->num_devices) {
			fprintf(stderr, "config_device = %d >= %d = num_devices\n", config_device, ocl_state->num_devices);
			exit(-1);
		}

		ocl_state->cdDevices = (cl_device_id *)malloc(sizeof(cl_device_id) * ocl_state->num_devices);
		err = clGetDeviceIDs(ocl_state->cpPlatform, CL_DEVICE_TYPE_ALL, ocl_state->num_devices, ocl_state->cdDevices, NULL);
		ocl_state->cdDevice = ocl_state->cdDevices[config_device];

		/* Fetch some information about the chosen device */
		{
			/* CL_DEVICE_NAME */
			{
				err = clGetDeviceInfo(ocl_state->cdDevice, CL_DEVICE_NAME, 0, NULL, &ocl_state->len_device_name);
				ocl_state->szDeviceName = (char *)malloc(ocl_state->len_device_name);
				err = clGetDeviceInfo(ocl_state->cdDevice, CL_DEVICE_NAME, ocl_state->len_device_name, ocl_state->szDeviceName, NULL);
			}

			/* CL_DEVICE_MAX_WORK_GROUP_SIZE */
			err = clGetDeviceInfo(ocl_state->cdDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(ocl_state->max_work_group_size), &ocl_state->max_work_group_size, NULL);

			/* CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS */
			err = clGetDeviceInfo(ocl_state->cdDevice, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(ocl_state->max_work_item_dimensions), &ocl_state->max_work_item_dimensions, NULL);

			/* CL_DEVICE_MAX_WORK_ITEM_SIZES */
			{
				size_t bufsize = sizeof(size_t) * ocl_state->max_work_item_dimensions;
				ocl_state->max_work_item_sizes = (size_t *)malloc(bufsize);
				err = clGetDeviceInfo(ocl_state->cdDevice, CL_DEVICE_MAX_WORK_ITEM_SIZES, bufsize, ocl_state->max_work_item_sizes, NULL);
			}
		}

		printf("Configuring OpenCL cofactorization for \"%s\" - \"%s\"\n", ocl_state->szPlatformVersion, ocl_state->szDeviceName);
    }

	/* Obtain a device context and a command queue */
	{
		ocl_state->cxGPUContext = clCreateContext(NULL, 1, &ocl_state->cdDevice, NULL, NULL, &err);
		ocl_state->cqCommandQueue = clCreateCommandQueue(ocl_state->cxGPUContext, ocl_state->cdDevice, 0, &err);
	}
}

void ocl_close(ocl_state_t *ocl_state) {
    /* Cleanup ocl_state */
	if (ocl_state->cpPlatforms) {
		free(ocl_state->cpPlatforms);
		ocl_state->cpPlatforms = NULL;
	}
	if (ocl_state->cdDevices) {
		free(ocl_state->cdDevices);
		ocl_state->cdDevices = NULL;
	}
	if (ocl_state->szPlatformVersion) {
		free(ocl_state->szPlatformVersion);
		ocl_state->szPlatformVersion = NULL;
	}
	if (ocl_state->szDeviceName) {
		free(ocl_state->szDeviceName);
		ocl_state->szDeviceName = NULL;
	}
	if (ocl_state->max_work_item_sizes) {
		free(ocl_state->max_work_item_sizes);
		ocl_state->max_work_item_sizes = NULL;
	}
	if (ocl_state->cpProgram)
		clReleaseProgram(ocl_state->cpProgram);
	if (ocl_state->cqCommandQueue)
		clReleaseCommandQueue(ocl_state->cqCommandQueue);
	if (ocl_state->cxGPUContext)
		clReleaseContext(ocl_state->cxGPUContext);
    return;
}


void ocl_build(ocl_state_t *ocl_state, const char *config_mp_source, const char *config_build_opts) {
	/* Load and compile our program */
	{
        FILE *fSource;							/* Handle to source file */
        size_t len_source;						/* Length of source file */
        char *szSource;							/* Contents of source file */
        size_t len_build_log;					/* Length of build log */
        char *szBuildLog;						/* Contents of build log */

		struct timespec build_start;
		struct timespec build_stop;

		printf("Building OpenCL kernels...\n");
		
		// ocl_state->buildopts
		{
		}

		if (clock_gettime(CLOCK_REALTIME, &build_start) < 0) {
			perror("clock_gettime");
			exit(-1);
		}

		/* Read the source */
		{
			fSource = fopen(config_mp_source, "rb");
			if (!fSource) {
				perror("fopen");
				exit(-1);
			}

			/* Determine the size of the source code */
			if (fseek(fSource, 0, SEEK_END) < 0) {
				perror("fseek");
				exit(-1);
			}
			long lSize = ftell(fSource);
			if (lSize < 0) {
				perror("ftell");
				exit(-1);
			}
			len_source = lSize; /* lSize >= 0 */
			if (fseek(fSource, 0, SEEK_SET) < 0) {
				perror("fseek");
				exit(-1);
			}

			/* Allocate a buffer for the source code */
			szSource = (char *)malloc(len_source + 1); /* +1 for '\0' */

			/* Read the source, then '\0'-terminate */
			if (1 != fread(szSource, len_source, 1, fSource)) {
				perror("fread");
				exit(-1);
			}
			szSource[len_source] = '\0';

			/* Close the file */
			fclose(fSource);
			fSource = NULL;
		}

		/* Create the program */
		{
			cl_int err;

			ocl_state->cpProgram = clCreateProgramWithSource(ocl_state->cxGPUContext, 1, (const char **)&szSource, &len_source, &err);
			err = clBuildProgram(ocl_state->cpProgram, 0, NULL, config_build_opts, NULL, NULL);
			if ((CL_SUCCESS == err) || (CL_BUILD_PROGRAM_FAILURE == err)) {
				cl_uint err2;

				/* Determine the size of the build log */
				err2 = clGetProgramBuildInfo(ocl_state->cpProgram, ocl_state->cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &len_build_log);

				/* Allocate memory for the build log */
				szBuildLog = (char *)malloc(len_build_log);

				/* Fetch the build log */
				err2 = clGetProgramBuildInfo(ocl_state->cpProgram, ocl_state->cdDevice, CL_PROGRAM_BUILD_LOG, len_build_log, szBuildLog, NULL);

				/* Print the build log */
				if (strlen(szBuildLog) > 2)
					fprintf((CL_SUCCESS != err) ? stderr : stdout, "%s\n\n", szBuildLog);

				/* Cleanup the build log */
				free(szBuildLog);
				szBuildLog = NULL;
			}

			/* Cleanup the source contents */
			free(szSource);
			szSource = NULL;
		}

		if (clock_gettime(CLOCK_REALTIME, &build_stop) < 0) {
			perror("clock_gettime");
			exit(-1);
		}

		double elapse = (double)(build_stop.tv_sec - build_start.tv_sec) +
						((double)(build_stop.tv_nsec - build_start.tv_nsec))/1000000000;
		printf("Build complete, wall time %f s\n", elapse);
	}
}


int ocl_kern_setup(ocl_state_t *ocl_state, const char *szKernName, ocl_kern_state_t *kern_state, size_t desired_global_work_size) {
	int ret = 0;

    /* Zero-initialize the kernel */
	bzero(kern_state, sizeof(*kern_state));

	/* Create the kernel, obtain kernel_work_group_size */
	{
		cl_int err;

		kern_state->ckKernel = clCreateKernel(ocl_state->cpProgram, szKernName, &err);
		if (CL_SUCCESS != err) {
			fprintf(stderr, "Error %.8x (clCreateKernel) while creating kernel %s\n", err, szKernName);
            exit(-1);
		}

		err = clGetKernelWorkGroupInfo(kern_state->ckKernel,
				ocl_state->cdDevice,
				CL_KERNEL_WORK_GROUP_SIZE,
				sizeof(kern_state->kernel_work_group_size),
				&kern_state->kernel_work_group_size,
				NULL);
		if (CL_SUCCESS != err) {
			fprintf(stderr, "Error %.8x (clGetKernelWorkGroupInfo)\n", err);
            exit(-1);
		}
    }

	/* Determine local work size */
	{
		/* The actual local work group size is also bounded by max_work_group_size and max_work_item_sizes */
		kern_state->local_work_size = kern_state->kernel_work_group_size;
        while (kern_state->local_work_size > desired_global_work_size)
            kern_state->local_work_size >>= 1;

		if (ocl_state->max_work_group_size < kern_state->local_work_size)
			kern_state->local_work_size = ocl_state->max_work_group_size;

		if (ocl_state->max_work_item_sizes[0] < kern_state->local_work_size)
			kern_state->local_work_size = ocl_state->max_work_item_sizes[0];
	}

	/* Determine global work size */
	{
		/* The global work group size must be the lowest multiple of local_work_size which is greater-than the desired global work size */
		kern_state->global_work_size = desired_global_work_size;

		size_t rem = kern_state->global_work_size % kern_state->local_work_size;
		if (rem)
			kern_state->global_work_size += kern_state->local_work_size - rem;
	}

	CLEANUP:
	return ret;
}


void ocl_kern_clear(ocl_kern_state_t *kern_state) {
	if (kern_state->ckKernel) {
		clReleaseKernel(kern_state->ckKernel);
		kern_state->ckKernel = 0;
	}
}


int ocl_kern_run(ocl_state_t *ocl_state, ocl_kern_state_t *kern_state) {
	int ret = 0;

	if (clock_gettime(CLOCK_REALTIME, &kern_state->start) < 0) {
		perror("clock_gettime in ocl_kern_run");
		exit(-1);
	}

	cl_int err = 0;
	cl_event eKernel = 0;
	err = clEnqueueNDRangeKernel(ocl_state->cqCommandQueue,
			kern_state->ckKernel,
			1,
			NULL,
			&kern_state->global_work_size,
			&kern_state->local_work_size,
			0,
			NULL,
			&eKernel);
	if (CL_SUCCESS != err) {
		fprintf(stderr, "Error %.8x (clEnqueueNDRangeKernel)\n", err);
        exit(-1);
    }
	err = clWaitForEvents(1, &eKernel);
	if (CL_SUCCESS != err) {
		fprintf(stderr, "Error %d (clWaitForEvents)\n", err);
        exit(-1);
    }
	clReleaseEvent(eKernel);

	if (clock_gettime(CLOCK_REALTIME, &kern_state->stop) < 0) {
		perror("clock_gettime in ocl_kern_run");
		exit(-1);
	}

	CLEANUP:
	return ret;
}


#endif /* USE_OPENCL */
