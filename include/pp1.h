/*
 * pp1.h
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */

#ifndef PP1_H_
#define PP1_H_
#include "ul64.h"
#include "ul128.h"
#include "ulmpz.h"
#include "cofact_plan.h"
#include "pp1.h"

extern int pp1_stage1_ul64(ul64 f, ul64 X, mod64 m, pp1_plan_t *plan);
extern void pp1_stage2_ul64(ul64 f, ul64 X, mod64 m, stage2_plan_t *plan);

extern int pp1_stage1_ul128(ul128 f, ul128 X, mod128 m, pp1_plan_t *plan);
extern void pp1_stage2_ul128(ul128 f, ul128 X, mod128 m, stage2_plan_t *plan);

extern int pp1_stage1_ulmpz(mpz_t f, mpz_t X, modmpz_t m, pp1_plan_t *plan);
extern void pp1_stage2_ulmpz(mpz_t f, mpz_t X, modmpz_t m, stage2_plan_t *plan);

#endif /* PP1_H_ */
