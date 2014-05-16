/*
 * ul128_def.h
 *
 *  Created on: Aug 26, 2013
 *      Author: pstach
 */

#ifndef UL128_DEF_H_
#define UL128_DEF_H_

#define UL_DEFINES
#define USE_UL128

typedef ul128 ul;
typedef mod128 mod;
typedef ul128 ul2x;

#define ul_init ul128_init
#define ul_clear ul128_clear
#define ul2x_init ul256_init
#define ul2x_clear ul256_clear

#define ul_set_ui ul128_set_ui
#define ul_get_ui ul128_get_ui
#define ul_set ul128_set
#define ul_cmp ul128_cmp
#define ul_cmp_ui ul128_cmp_ui
#define mpz_set_ul mpz_set_ul128
#define mpz_get_ul mpz_get_ul128

#define ul_print ul128_print
#define ul_setbit ul128_setbit
#define ul_tstbit ul128_tstbit

#define ul_add ul128_add
#define ul_sub ul128_sub
#define ul_mul ul128_mul
#define ul_mod ul128_mod
#define ul_mul_ul2x ul128_mul_ul256
#define ul_divrem ul128_divrem
#define ul_bscan_fwd ul128_bscan_fwd
#define ul_bscan_rev ul128_bscan_rev
#define ul_lshift ul128_lshift
#define ul_rshift ul128_rshift

#define mod_init mod128_init
#define mod_clear mod128_clear
#define mod_set mod128_set
#define ul_to_montgomery ul128_to_montgomery
#define ul_from_montgomery ul128_from_montgomery

#define ul_gcd ul128_gcd

#define ul_modadd ul128_modadd
#define ul_modsub ul128_modsub
#define ul_modmul ul128_modmul
#define ul_moddiv2 ul128_moddiv2
#define ul_moddiv3 ul128_moddiv3
#define ul_moddiv5 ul128_moddiv5
#define ul_moddiv7 ul128_moddiv7
#define ul_moddiv11 ul128_moddiv11
#define ul_moddiv13 ul128_moddiv13
#define ul_modinv ul128_modinv

#define ul_rand ul128_rand
#define ul_modrand ul128_modrand

#define mpz_set_ul2x mpz_set_ul256
#define mpz_get_ul2x mpz_get_ul256

#endif /* UL128_DEF_H_ */

