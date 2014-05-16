/*
 * ul_dummy.h
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */

#ifndef UL_DUMMY_H_
#define UL_DUMMY_H_

#ifndef UL_DEFINES
#error "don't include ul_dummy.h, its sole existance is to make intellesense happy"

typedef u_int64_t ul[1];
typedef u_int64_t ul2x[2];

typedef struct mod_s {
	ul r;
	ul r2;
	ul n;
} mod[1];

/* setters/getters */
extern void ul_set_ui(ul dst, u_int64_t src);
extern u_int64_t ul_get_ui(ul src);
extern void ul_set(ul dst, ul src);
extern int ul_cmp(ul64 src1, ul64 src2);
extern int ul_cmp_ui(ul64 src1, u_int64_t src2);

/* mpz routines */
extern void mpz_set_ul(mpz_t dst, ul src);
extern void mpz_get_ul(ul dst, mpz_t src);

/* arithmetic routines */
extern void ul_add(ul dst, ul src1, ul src2);
extern void ul_sub(ul dst, ul src1, ul src2);
extern void ul_mul(ul dst, ul src1, ul src2);
extern void ul_mul_ul2x(ul2x dst, ul src1, ul src2);
extern void ul_divrem(ul d, ul r, ul a, ul n);
extern void ul_gcd(ul dst, ul x, ul y);

/* modular arithmetic routines */
extern void ul_modadd(ul dst, ul src1, ul src2, mod n);
extern void ul_modsub(ul dst, ul src1, ul src2, mod n);
extern void ul_modmul(ul dst, ul src1, ul src2, mod n);
extern void ul_modinv(ul dst, ul src, mod n);

/* randomization routines */
extern void ul_rand(ul dst);
extern void ul_modrand(ul dst, mod n);

#endif /* UL_DEFINES */
#endif /* UL_DUMMY_H_ */
