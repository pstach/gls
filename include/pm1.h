/*
 * pm1.h
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */

#ifndef PM1_H_
#define PM1_H_
#include "ul64.h"
#include "ul128.h"
#include "ulmpz.h"
#include "cofact_plan.h"
#include "pp1.h"

extern int pm1_stage1_ul64(ul64 f, ul64 X, mod64 m, pm1_plan_t *plan);
#define pm1_stage2_ul64 pp1_stage2_ul64

extern int pm1_stage1_ul128(ul128 f, ul128 X, mod128 m, pm1_plan_t *plan);
#define pm1_stage2_ul128 pp1_stage2_ul128

extern int pm1_stage1_mpz(mpz_t f, mpz_t X, modmpz_t m, pm1_plan_t *plan);
#define pm1_stage2_mpz pp1_stage2_mpz

#endif /* PM1_H_ */
