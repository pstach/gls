/*
 * ul64_def.h
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */

#ifndef UL64_DEF_H_
#define UL64_DEF_H_

#define UL_DEFINES
#define USE_UL64

typedef ul64 ul;
typedef mod64 mod;
typedef ul128 ul2x;

#define ul_init ul64_init
#define ul_clear ul64_clear
#define ul2x_init ul128_init
#define ul2x_clear ul128_clear

#define ul_set_ui ul64_set_ui
#define ul_get_ui ul64_get_ui
#define ul_set ul64_set
#define ul_cmp ul64_cmp
#define ul_cmp_ui ul64_cmp_ui
#define mpz_set_ul mpz_set_ul64
#define mpz_get_ul mpz_get_ul64

#define ul_print ul64_print

#define ul_setbit ul64_setbit
#define ul_tstbit ul64_tstbit

#define ul_add ul64_add
#define ul_sub ul64_sub
#define ul_mul ul64_mul
#define ul_mod ul64_mod
#define ul_mul_ul2x ul64_mul_ul128
#define ul_divrem ul64_divrem

#define ul_bscan_fwd ul64_bscan_fwd
#define ul_bscan_rev ul64_bscan_rev
#define ul_lshift ul64_lshift
#define ul_rshift ul64_rshift

#define mod_init mod64_init
#define mod_clear mod64_clear
#define mod_set mod64_set
#define ul_to_montgomery ul64_to_montgomery
#define ul_from_montgomery ul64_from_montgomery

#define ul_gcd ul64_gcd

#define ul_modadd ul64_modadd
#define ul_modsub ul64_modsub
#define ul_modmul ul64_modmul
#define ul_moddiv2 ul64_moddiv2
#define ul_moddiv3 ul64_moddiv3
#define ul_moddiv5 ul64_moddiv5
#define ul_moddiv7 ul64_moddiv7
#define ul_moddiv11 ul64_moddiv11
#define ul_moddiv13 ul64_moddiv13
#define ul_modinv ul64_modinv

#define ul_rand ul64_rand
#define ul_modrand ul64_modrand

#define mpz_set_ul2x mpz_set_ul128
#define mpz_get_ul2x mpz_get_ul128

#endif /* UL64_DEF_H_ */
