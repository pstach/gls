/*
 * types.h
 *
 *  Created on: Nov 12, 2013
 *      Author: tcarstens
 */

#ifndef TYPES_H_
#define TYPES_H_

#include <stdio.h>
#include <gmp.h>
#include <CL/opencl.h>

typedef cl_uchar u_int8_t;
typedef cl_uint u_int32_t;
typedef cl_ulong u_int64_t;


typedef struct ocl_ul32_t {
	u_int32_t v0;
} ocl_ul32;

static inline void mpz_set_ocl_ul32(mpz_t dst, ocl_ul32 *src) {
	unsigned long int v0 = src->v0;

	mpz_set_ui(dst, v0);
}


typedef struct ocl_ul64_t {
	u_int32_t v0;
	u_int32_t v1;
} ocl_ul64;

static inline void mpz_set_ocl_ul64(mpz_t dst, ocl_ul64 *src) {
	unsigned long int v0 = ((unsigned long int)(src->v1) << 32) | src->v0;

	mpz_set_ui(dst, v0);
}

typedef struct ocl_ul96_t {
	u_int32_t v0;
	u_int32_t v1;
	u_int32_t v2;
} ocl_ul96;

static inline void mpz_set_ocl_ul96(mpz_t dst, ocl_ul96 *src) {
	unsigned long int v1 = src->v2;
	unsigned long int v0 = ((unsigned long int)(src->v1) << 32) | src->v0;

	mpz_set_ui(dst, v1);
	mpz_mul_2exp(dst, dst, 64);
	mpz_add_ui(dst, dst, v0);
}


typedef struct ocl_ul128_t {
	u_int32_t v0;
	u_int32_t v1;
	u_int32_t v2;
	u_int32_t v3;
} ocl_ul128;

static inline void mpz_set_ocl_ul128(mpz_t dst, ocl_ul128 *src) {
	unsigned long int v1 = ((unsigned long int)(src->v3) << 32) | src->v2;
	unsigned long int v0 = ((unsigned long int)(src->v1) << 32) | src->v0;

	mpz_set_ui(dst, v1);
	mpz_mul_2exp(dst, dst, 64);
	mpz_add_ui(dst, dst, v0);
}


typedef struct ocl_ul160_t {
	u_int32_t v0;
	u_int32_t v1;
	u_int32_t v2;
	u_int32_t v3;
	u_int32_t v4;
} ocl_ul160;

static inline void mpz_set_ocl_ul160(mpz_t dst, ocl_ul160 *src) {
	unsigned long int v2 = src->v4;
	unsigned long int v1 = ((unsigned long int)(src->v3) << 32) | src->v2;
	unsigned long int v0 = ((unsigned long int)(src->v1) << 32) | src->v0;

	mpz_set_ui(dst, v2);
	mpz_mul_2exp(dst, dst, 64);
	mpz_add_ui(dst, dst, v1);
	mpz_mul_2exp(dst, dst, 64);
	mpz_add_ui(dst, dst, v0);
}


typedef struct ocl_ul192_t {
	u_int32_t v0;
	u_int32_t v1;
	u_int32_t v2;
	u_int32_t v3;
	u_int32_t v4;
	u_int32_t v5;
} ocl_ul192;

static inline void mpz_set_ocl_ul192(mpz_t dst, ocl_ul192 *src) {
	unsigned long int v2 = ((unsigned long int)(src->v5) << 32) | src->v4;
	unsigned long int v1 = ((unsigned long int)(src->v3) << 32) | src->v2;
	unsigned long int v0 = ((unsigned long int)(src->v1) << 32) | src->v0;

	mpz_set_ui(dst, v2);
	mpz_mul_2exp(dst, dst, 64);
	mpz_add_ui(dst, dst, v1);
	mpz_mul_2exp(dst, dst, 64);
	mpz_add_ui(dst, dst, v0);
}


typedef struct ocl_ul224_t {
	u_int32_t v0;
	u_int32_t v1;
	u_int32_t v2;
	u_int32_t v3;
	u_int32_t v4;
	u_int32_t v5;
	u_int32_t v6;
} ocl_ul224;

static inline void mpz_set_ocl_ul224(mpz_t dst, ocl_ul224 *src) {
	unsigned long int v3 = src->v6;
	unsigned long int v2 = ((unsigned long int)(src->v5) << 32) | src->v4;
	unsigned long int v1 = ((unsigned long int)(src->v3) << 32) | src->v2;
	unsigned long int v0 = ((unsigned long int)(src->v1) << 32) | src->v0;

	mpz_set_ui(dst, v3);
	mpz_mul_2exp(dst, dst, 64);
	mpz_add_ui(dst, dst, v2);
	mpz_mul_2exp(dst, dst, 64);
	mpz_add_ui(dst, dst, v1);
	mpz_mul_2exp(dst, dst, 64);
	mpz_add_ui(dst, dst, v0);
}


typedef struct ocl_ul256_t {
	u_int32_t v0;
	u_int32_t v1;
	u_int32_t v2;
	u_int32_t v3;
	u_int32_t v4;
	u_int32_t v5;
	u_int32_t v6;
	u_int32_t v7;
} ocl_ul256;

static inline void mpz_set_ocl_ul256(mpz_t dst, ocl_ul256 *src) {
	unsigned long int v3 = ((unsigned long int)(src->v7) << 32) | src->v6;
	unsigned long int v2 = ((unsigned long int)(src->v5) << 32) | src->v4;
	unsigned long int v1 = ((unsigned long int)(src->v3) << 32) | src->v2;
	unsigned long int v0 = ((unsigned long int)(src->v1) << 32) | src->v0;

	mpz_set_ui(dst, v3);
	mpz_mul_2exp(dst, dst, 64);
	mpz_add_ui(dst, dst, v2);
	mpz_mul_2exp(dst, dst, 64);
	mpz_add_ui(dst, dst, v1);
	mpz_mul_2exp(dst, dst, 64);
	mpz_add_ui(dst, dst, v0);
}

#endif /* TYPES_H_ */
