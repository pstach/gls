/*
 * ulmpz_def.h
 *
 *  Created on: Sep 15, 2013
 *      Author: pstach
 */

#ifndef ULMPZ_DEF_H_
#define ULMPZ_DEF_H_

#define UL_DEFINES

typedef mpz_t ul;
typedef modmpz_t mod;
typedef mpz_t ul2x;

#define ul_init mpz_init
#define ul_clear mpz_clear

#define ul_set_ui mpz_set_ui
#define ul_get_ui mpz_get_ui
#define ul_set mpz_set
#define ul_cmp mpz_cmp
#define ul_cmp_ui mpz_cmp_ui
#define mpz_set_ul mpz_set
#define mpz_get_ul mpz_set

#define ul_print mpz_printer
#define ul_setbit mpz_setbit
#define ul_tstbit mpz_tstbit

#define ul_add mpz_add
#define ul_sub mpz_sub
#define ul_mul mpz_mul
#define ul_mod mpz_mod
#define ul_mul_ul2x mpz_mul
#define ul_divrem mpz_fdiv_qr
#define ul_bscan_fwd mpz_bscan_fwd
#define ul_bscan_rev mpz_bscan_rev
#define ul_lshift mpz_mul_2exp
#define ul_rshift mpz_div_2exp

#define mod_init mpzmod_init
#define mod_clear mpzmod_clear
#define mod_set mpzmod_set
#define ul_to_montgomery mpz_to_montgomery
#define ul_from_montgomery mpz_from_montgomery

#define ul_gcd mpz_gcd

#define ul_modadd mpz_modadd
#define ul_modsub mpz_modsub
#define ul_modmul mpz_modmul
#define ul_moddiv2 mpz_moddiv2
#define ul_moddiv3 mpz_moddiv3
#define ul_moddiv5 mpz_moddiv5
#define ul_moddiv7 mpz_moddiv7
#define ul_moddiv11 mpz_moddiv11
#define ul_moddiv13 mpz_moddiv13
#define ul_modinv mpz_invert

#define ul_rand mpz_rand
#define ul_modrand mpz_modrand

#endif /* ULMPZ_DEF_H_ */
