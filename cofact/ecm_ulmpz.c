/*
 * ecm_ulmpz.c
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */
#include <stdio.h>
#include <stdlib.h>
#include "ulmpz.h"
#include "ulmpz_def.h"

#define ecm_stage1 ecm_stage1_mpz
#define ecm_stage2 ecm_stage2_mpz

#include "ecm_common.c"



