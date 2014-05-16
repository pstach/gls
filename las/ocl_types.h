/*
 * ocl_types.h
 *
 *  Created on: January 28, 2014
 *      Author: tcarstens
 */


#ifndef OCL_TYPES_H_
#define OCL_TYPES_H_

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

static inline void mpz_get_ocl_ul32(ocl_ul32 *dst, mpz_t src) {
    dst->v0 = 0;
    switch(src->_mp_size) {
    default:
        dst->v0 = src->_mp_d[0];
    case 0:
        break;
    }
}


typedef struct ocl_ul64_t {
	u_int32_t v0;
	u_int32_t v1;
} ocl_ul64;

static inline void mpz_set_ocl_ul64(mpz_t dst, ocl_ul64 *src) {
	unsigned long int v0 = ((unsigned long int)(src->v1) << 32) | src->v0;

	mpz_set_ui(dst, v0);
}

static inline void mpz_get_ocl_ul64(ocl_ul64 *dst, mpz_t src) {
    dst->v0 = 0;
    dst->v1 = 0;
    switch(src->_mp_size) {
    default:
        dst->v1 = src->_mp_d[0] >> 32;
        dst->v0 = src->_mp_d[0];
    case 0:
        break;
    }
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

static inline void mpz_get_ocl_ul96(ocl_ul96 *dst, mpz_t src) {
    dst->v0 = 0;
    dst->v1 = 0;
    dst->v2 = 0;
    switch(src->_mp_size) {
    default:
        dst->v2 = src->_mp_d[1];
    case 1:
        dst->v1 = src->_mp_d[0] >> 32;
        dst->v0 = src->_mp_d[0];
    case 0:
        break;
    }
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

static inline void mpz_get_ocl_ul128(ocl_ul128 *dst, mpz_t src) {
    dst->v0 = 0;
    dst->v1 = 0;
    dst->v2 = 0;
    dst->v3 = 0;
    switch(src->_mp_size) {
    default:
        dst->v3 = src->_mp_d[1] >> 32;
        dst->v2 = src->_mp_d[1];
    case 1:
        dst->v1 = src->_mp_d[0] >> 32;
        dst->v0 = src->_mp_d[0];
    case 0:
        break;
    }
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

static inline void mpz_get_ocl_ul160(ocl_ul160 *dst, mpz_t src) {
    dst->v0 = 0;
    dst->v1 = 0;
    dst->v2 = 0;
    dst->v3 = 0;
    dst->v4 = 0;
    switch(src->_mp_size) {
    default:
        dst->v4 = src->_mp_d[2];
    case 2:
        dst->v3 = src->_mp_d[1] >> 32;
        dst->v2 = src->_mp_d[1];
    case 1:
        dst->v1 = src->_mp_d[0] >> 32;
        dst->v0 = src->_mp_d[0];
    case 0:
        break;
    }
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

static inline void mpz_get_ocl_ul192(ocl_ul192 *dst, mpz_t src) {
    dst->v0 = 0;
    dst->v1 = 0;
    dst->v2 = 0;
    dst->v3 = 0;
    dst->v4 = 0;
    dst->v5 = 0;
    switch(src->_mp_size) {
    default:
        dst->v5 = src->_mp_d[2] >> 32;
        dst->v4 = src->_mp_d[2];
    case 2:
        dst->v3 = src->_mp_d[1] >> 32;
        dst->v2 = src->_mp_d[1];
    case 1:
        dst->v1 = src->_mp_d[0] >> 32;
        dst->v0 = src->_mp_d[0];
    case 0:
        break;
    }
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

static inline void mpz_get_ocl_ul224(ocl_ul224 *dst, mpz_t src) {
    dst->v0 = 0;
    dst->v1 = 0;
    dst->v2 = 0;
    dst->v3 = 0;
    dst->v4 = 0;
    dst->v5 = 0;
    dst->v6 = 0;
    switch(src->_mp_size) {
    default:
        dst->v6 = src->_mp_d[3];
    case 3:
        dst->v5 = src->_mp_d[2] >> 32;
        dst->v4 = src->_mp_d[2];
    case 2:
        dst->v3 = src->_mp_d[1] >> 32;
        dst->v2 = src->_mp_d[1];
    case 1:
        dst->v1 = src->_mp_d[0] >> 32;
        dst->v0 = src->_mp_d[0];
    case 0:
        break;
    }
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

static inline void mpz_get_ocl_ul256(ocl_ul256 *dst, mpz_t src) {
    dst->v0 = 0;
    dst->v1 = 0;
    dst->v2 = 0;
    dst->v3 = 0;
    dst->v4 = 0;
    dst->v5 = 0;
    dst->v6 = 0;
    dst->v7 = 0;
    switch(src->_mp_size) {
    default:
        dst->v7 = src->_mp_d[3] >> 32;
        dst->v6 = src->_mp_d[3];
    case 3:
        dst->v5 = src->_mp_d[2] >> 32;
        dst->v4 = src->_mp_d[2];
    case 2:
        dst->v3 = src->_mp_d[1] >> 32;
        dst->v2 = src->_mp_d[1];
    case 1:
        dst->v1 = src->_mp_d[0] >> 32;
        dst->v0 = src->_mp_d[0];
    case 0:
        break;
    }
}

#endif /* OCL_TYPES_H_ */
