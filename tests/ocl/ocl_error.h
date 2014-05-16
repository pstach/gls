/*
 * ocl_error.h
 *
 *  Created on: Nov 11, 2013
 *      Author: tcarstens
 */

#ifndef OCL_ERROR_H_
#define OCL_ERROR_H_


#include <CL/opencl.h>

const char *ocl_error_str(cl_int err);

#define FAIL_CL(fn, err) { fprintf(stderr, "Error: %s() returned %s at %s:%d\n", fn, ocl_error_str(err), __FILE__, __LINE__); ret = -1; goto CLEANUP; }
#define FAIL_MALLOC(var) { fprintf(stderr, "Error: malloc() returned NULL for %s at %s:%d\n", var, __FILE__, __LINE__); ret = -1; goto CLEANUP; }


#endif /* OCL_ERROR_H_ */
