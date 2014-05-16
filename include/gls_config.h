/*
 * gls_config.h
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */

#ifndef GLS_CONFIG_H_
#define GLS_CONFIG_H_
#include "mpzpoly.h"

#define POLY_CNT	2
#define RPOLY_IDX	0
#define APOLY_IDX	1

typedef struct gls_config_s {
	mpz_t N;
	mpz_t m;
	double skew;
	mpzpoly_t poly[POLY_CNT];
	u_int64_t lim[POLY_CNT];
	u_int64_t lpb[POLY_CNT];
	u_int64_t mfb[POLY_CNT];
	double lambda[POLY_CNT];
	/* the rest are computed */
	u_int8_t bound[POLY_CNT];
	double scale[POLY_CNT];
	double cexp2[POLY_CNT][257];
	double logmax[POLY_CNT];
	mpzpoly_t fij[POLY_CNT];
	double *fijd[POLY_CNT];
	u_int64_t log_steps[POLY_CNT][256];
	u_int8_t log_steps_max[POLY_CNT];
	mpz_t BB[POLY_CNT];
	mpz_t BBB[POLY_CNT];
	mpz_t BBBB[POLY_CNT];
} gls_config_t[1];

extern int gls_config_init(gls_config_t cfg);
extern void gls_config_free(gls_config_t cfg);
extern int polyfile_read(gls_config_t cfg, const char *fname);

#endif /* GLS_CONFIG_H_ */
