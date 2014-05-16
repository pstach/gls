/*
 * cofact_plan.h
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */

#ifndef COFACT_PLAN_H_
#define COFACT_PLAN_H_

#define NEXT_D 254
#define NEXT_PASS 255

typedef struct stage2_plan_s
{
	unsigned int B2;
	unsigned int d;
	unsigned int i0, i1;
	unsigned int n_S1;
	u_int32_t *S1;
	unsigned char *pairs;
	unsigned int n_pairs;
} stage2_plan_t;

extern void stage2_plan_init(stage2_plan_t *plan, u_int32_t B2_min, u_int32_t B2);
extern void stage2_plan_clear(stage2_plan_t *plan);

typedef struct pm1_plan_s
{
	u_int64_t *E;
	u_int64_t E_mask;
	u_int32_t E_n_words;
	u_int32_t exp2;
	u_int32_t B1;
	stage2_plan_t stage2;
} pm1_plan_t;

extern void pm1_plan_init(pm1_plan_t *plan, u_int32_t B1, u_int32_t B2);
extern void pm1_plan_clear(pm1_plan_t *plan);

typedef struct {
	u_int8_t *bc; /* Bytecode for the Lucas chain for stage 1 */
	u_int32_t bc_len; /* Number of bytes in bytecode */
	u_int32_t exp2; /* Exponent of 2 in stage 1 primes */
	u_int32_t B1;
	stage2_plan_t stage2;
} pp1_plan_t;

extern void pp1_plan_init(pp1_plan_t *plan, u_int32_t B1, u_int32_t B2);
extern void pp1_plan_clear(pp1_plan_t *plan);

#define BRENT12 0
#define MONTY12 1

typedef struct {
	u_int8_t *bc; /* Bytecode for the Lucas chain for stage 1 */
	u_int32_t bc_len; /* Number of bytes in bytecode */
	u_int32_t exp2; /* Exponent of 2 in stage 1 primes */
	u_int32_t B1;
	int parameterization; /* BRENT12 or MONTY12 */
	u_int64_t sigma; /* Sigma parameter for Brent curves, or multiplier for Montgomery torsion-12 curves */
	stage2_plan_t stage2;
} ecm_plan_t;

extern void ecm_plan_init(ecm_plan_t *plan, u_int32_t B1, u_int32_t B2, int parameterization, u_int64_t sigma);
extern void ecm_plan_clear(ecm_plan_t *plan);

typedef struct {
	// u_int8_t *bc; /* Bytecode for the Lucas chain for stage 1 */
	u_int32_t bc_len; /* Number of bytes in bytecode */
	u_int32_t exp2; /* Exponent of 2 in stage 1 primes */
	u_int32_t B1;
	int parameterization; /* BRENT12 or MONTY12 */
	u_int64_t sigma; /* Sigma parameter for Brent curves, or multiplier for Montgomery torsion-12 curves */
	// stage2_plan_t stage2;
} ocl_ecm_plan_t;

typedef struct ocl_pm1_plan_s {
	u_int64_t E_mask;
	u_int32_t E_n_words;
	u_int32_t exp2;
	u_int32_t B1;
	u_int32_t B2;
} ocl_pm1_plan_t;

typedef struct ocl_pp1_plan_s {
	u_int32_t bc_len; /* Number of bytes in bytecode */
	u_int32_t exp2; /* Exponent of 2 in stage 1 primes */
	u_int32_t B1;
} ocl_pp1_plan_t;

typedef struct ocl_stage2_plan_s
{
	unsigned int B2;
	unsigned int d;
	unsigned int i0, i1;
	unsigned int n_S1;
	/* u_int32_t *S1; */
	/* unsigned char *pairs; */
	unsigned int n_pairs;
} ocl_stage2_plan_t;

#endif /* COFACT_PLAN_H_ */
