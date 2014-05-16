/*
 * ecm.h
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */

#ifndef ECM_H_
#define ECM_H_
#include "ul64.h"
#include "ul128.h"
#include "ulmpz.h"
#include "cofact_plan.h"

typedef struct
{
	ul64 x;
	ul64 z;
} ellM64_point_t[1];

typedef struct
{
	ul64 x;
	ul64 y;
} ellW64_point_t[1];

extern int ecm_stage1_ul64(ul64 f, ellM64_point_t X, ul64 b, mod64 m, ecm_plan_t *plan);
extern void ecm_stage2_ul64(ul64 r, ellM64_point_t P, ul64 b, mod64 m, stage2_plan_t *plan);

typedef struct
{
	ul128 x;
	ul128 z;
} ellM128_point_t[1];

typedef struct
{
	ul128 x;
	ul128 y;
} ellW128_point_t[1];

extern int ecm_stage1_ul128(ul128 f, ellM128_point_t X, ul128 b, mod128 m, ecm_plan_t *plan);
extern void ecm_stage2_ul128(ul128 f, ellM128_point_t X, ul128 b, mod128 m, stage2_plan_t *plan);

typedef struct
{
	mpz_t x;
	mpz_t z;
} ellMmpz_point_t[1];

typedef struct
{
	mpz_t x;
	mpz_t y;
} ellWmpz_point_t[1];

extern int ecm_stage1_ulmpz(mpz_t f, ellMmpz_point_t X, mpz_t b, modmpz_t m, ecm_plan_t *plan);
extern void ecm_stage2_ulmpz(mpz_t f, ellMmpz_point_t X, mpz_t b, modmpz_t m, stage2_plan_t *plan);

#endif /* ECM_H_ */
