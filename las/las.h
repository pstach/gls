/*
 * las.h
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */

#ifndef LAS_H_
#define LAS_H_
#include "gls_config.h"
#include "fb.h"

#define TEST_OCL_QLATTICE31 0 /* test OCL root_in_qlattice31 in line sieve */
#define TEST_OCL_VEC_RED    0 /* test OCL root_in_qlattice31_reduce_lattice31 in vector sieve */
//#define TEST_TRANSLATE 1 /* test factor base translation */
//#define BUCKET_DEBUG 1 /* test bucketed ideals */
//#define TEST_FACTORS 1 /* test factors as they are added to a candidate */



#define SIEVE_SHIFT 16
#define SIEVE_SIZE (1 << SIEVE_SHIFT)
#define MEMSET_MIN 64

typedef struct lat_s {
	int64_t a0;
	int64_t b0;
	int64_t a1;
	int64_t b1;
} lat_t[1];

#define BUCKET_SHIFT (SIEVE_SHIFT)
#define BUCKET_BYTES (256ULL * 1024)
#define BUCKET_SIZE (1 << BUCKET_SHIFT)
#define BUCKET_MASK (BUCKET_SIZE - 1)

typedef struct bucket_s {
	u_int32_t logp : 8;
	u_int32_t r : 24;
	u_int32_t p;
} bucket_t;

#define BUCKET_START(x) ((bucket_t *) ((u_int64_t) (x) & ~(BUCKET_BYTES - 1)))

typedef struct candidate_s {
	int64_t a;
	int64_t b;
	u_int32_t r; /* index into bucket */
	int side;
	mpz_t rem[POLY_CNT];
	mpz_t *factors[POLY_CNT];
	u_int32_t n_factors[POLY_CNT];
	struct candidate_s *next;
} candidate_t;

extern FILE *output;
extern double ltrans_time, vtrans_time, lsieve_time, vsieve_time, scheck_time, lresieve_time, vresieve_time;
extern double ltrans_total, vtrans_total, lsieve_total, vsieve_total, scheck_total, lresieve_total, vresieve_total;
extern double pm1_total, pp1_total, ecm_total;
extern u_int64_t rel_count;

static inline double dbltime(void)
{
	double ret;
	struct timeval tv;

	gettimeofday(&tv, NULL);
	ret = ((double) tv.tv_usec / 1000000.) + tv.tv_sec;
	return ret;
}

static inline void candidate_print(candidate_t *cand);

static inline candidate_t *candidate_add(candidate_t **list, gls_config_t cfg, int64_t a, u_int64_t b, u_int32_t r)
{
	u_int32_t i;
	candidate_t *ret;
	mpz_t t0, t1;

	ret = (candidate_t *) malloc(sizeof(candidate_t));
	memset(ret, 0, sizeof(candidate_t));

	ret->a = a;
	ret->b = b;
	ret->r = r;
	ret->next = *list;
	mpz_init(t0);
	mpz_init(t1);

	for(i = 0; i < sizeof(ret->rem) / sizeof(ret->rem[0]); i++)
	{
		mpz_init(ret->rem[i]);
		mpz_set_si(t0, a);
		mpz_set_ui(t1, b);
		mpzpoly_eval_ab(ret->rem[i], cfg->poly[i], t0, t1);
		mpz_abs(ret->rem[i], ret->rem[i]);
	}
	mpz_clear(t0);
	mpz_clear(t1);
	*list = ret;
	return ret;
}

static inline void candidate_free(candidate_t *cand)
{
	u_int32_t i, j;

	for(i = 0; i < sizeof(cand->rem) / sizeof(cand->rem[0]); i++)
	{
		mpz_clear(cand->rem[i]);
		if(cand->factors[i])
		{
			for(j = 0; j < cand->n_factors[i]; j++)
				mpz_clear(cand->factors[i][j]);
			free(cand->factors[i]);
		}
	}
	free(cand);
	return;
}

static inline void candidate_print(candidate_t *cand)
{
	int side;
	u_int32_t i;

	gmp_fprintf(output, "%lld,%lld:", cand->a, cand->b);
	for(side = 0; side < sizeof(cand->factors) / sizeof(cand->factors[0]); side++)
	{
		if(cand->n_factors[side])
		{
			gmp_fprintf(output, "%Zx", cand->factors[side][0]);
			for(i = 1; i < cand->n_factors[side]; i++)
			{
				gmp_fprintf(output, ",%Zx", cand->factors[side][i]);
			}
		}
		if(side < sizeof(cand->factors) / sizeof(cand->factors[0]) - 1)
			gmp_fprintf(output, ":");
	}
	gmp_fprintf(output, "\n");
	return;
}

static inline void candidate_add_factor_mpz(candidate_t *cand, int side, mpz_t f)
{
	mpz_t *factors;
	u_int32_t n_factors;

	n_factors = cand->n_factors[side];
	factors = cand->factors[side];

	n_factors++;
	factors = (mpz_t *) realloc(factors, n_factors * sizeof(mpz_t));
	mpz_init_set(factors[n_factors - 1], f);
#ifdef TEST_FACTORS
	{
		mpz_t tmp;
		mpz_init(tmp);
		mpz_mod(tmp, cand->rem[side], f);
		if(mpz_cmp_ui(tmp, 0) != 0)
		{
			gmp_fprintf(output, "bad %s factor %Zd for candidate a=%lld b=%lld r=%u\n", (side == RPOLY_IDX) ? "rational" : "algebraic", f, cand->a, cand->b, cand->r);
			__asm__ volatile("int3\n\t");
		}
		mpz_clear(tmp);
	}
#endif
	mpz_div(cand->rem[side], cand->rem[side], f);
	cand->factors[side] = factors;
	cand->n_factors[side] = n_factors;
	return;
}

#endif /* LAS_H_ */
