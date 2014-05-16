/*
 * tests.h
 *
 *  Created on: Nov 12, 2013
 *      Author: tcarstens
 */

#ifndef TESTS_H_
#define TESTS_H_

#include "state.h"


/* Here you can configure which sizes you want to test */
#define TEST_32		(1 << 0)
#define TEST_64     (1 << 1)
#define TEST_96     (1 << 2)
#define TEST_128    (1 << 3)
#define TEST_160    (1 << 4)
#define TEST_192    (1 << 5)
#define TEST_224    (1 << 6)
#define TEST_256    (1 << 7)

#define TEST_ALL    (TEST_32 | TEST_64 | TEST_96 | TEST_128 | TEST_160 | TEST_192 | TEST_224 | TEST_256)
#define TEST_SET    (TEST_64 | TEST_128)


/* And here you can configure which operations to test */
#define TESTOP_ADD      (1 << 0)
#define TESTOP_SUB      (1 << 1)
#define TESTOP_MUL      (1 << 2)
#define TESTOP_CMP      (1 << 3)
#define TESTOP_MODADD   (1 << 4)
#define TESTOP_MODSUB   (1 << 5)
#define TESTOP_MODMUL   (1 << 6)
#define TESTOP_MODMUL2  (1 << 7)
#define TESTOP_DIVREM   (1 << 8)
#define TESTOP_MODINV   (1 << 9)

#define TESTOP_PM1   (1 << 10)
#define TESTOP_PP1   (1 << 11)
#define TESTOP_ECM   (1 << 12)

#define TESTOP_ALL  (TESTOP_ADD | TESTOP_SUB | TESTOP_MUL | TESTOP_CMP | TESTOP_MODADD | TESTOP_MODSUB | TESTOP_MODMUL | TESTOP_MODMUL2 | TESTOP_DIVREM | TESTOP_MODINV | TESTOP_PM1 | TESTOP_PP1 | TESTOP_ECM)
#define TESTOP_SET	(TESTOP_MODMUL | TESTOP_MODMUL2 | TESTOP_PM1 | TESTOP_PP1 | TESTOP_ECM)

/*
 * Causes the test harness to check the randomly-generated
 * test inputs for certain arithmetic invariants.
 */
#undef TEST_CHECK_RAND_THREE

/*
 * We have two methods of computing the Montgomery
 * product of two values modulo a third. You can
 * enable one or the other, or both, and in the
 * latter case the two reference values will be
 * checked against one another.
 */
#undef TEST_GMPD_A
#define TEST_GMPD_B


/* Forward declarations for the testop routines */
int ul32_test_all(struct state_t *state);
int ul64_test_all(struct state_t *state);
int ul96_test_all(struct state_t *state);
int ul128_test_all(struct state_t *state);
int ul160_test_all(struct state_t *state);
int ul192_test_all(struct state_t *state);
int ul224_test_all(struct state_t *state);
int ul256_test_all(struct state_t *state);

#endif /* TESTS_H_ */
