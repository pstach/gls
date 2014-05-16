/*
 * cofact.c
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/mman.h>
#include <gmp.h>
#include <smmintrin.h>
#include "las.h"
#include "qgen.h"
#include "cofact.h"
#include "cofact_plan.h"
#include "pm1.h"
#include "pp1.h"
#include "ecm.h"

#include "ocl_cofact.h"
#define PRINTF_FACTOR_FOUND 0

#if USE_OPENCL
#include <time.h>
#define MAX(a,b) ((a) > (b) ? a : b)
#define COFACT_SIZES 9 /* 32, 64, 96, 128, 160, 192, 224, 256, mpz */
#define COFACT_BATCH_SIZE (1024 * 1024)
#define SIZE_INCREMENT 32
#else
#define COFACT_SIZES 3 /* 64, 128, mpz */
#define COFACT_BATCH_SIZE (1024 * 8)
#define SIZE_INCREMENT 64
#endif
#define MPZ_SIZE_IDX (COFACT_SIZES - 1)

static cofact_algo_t **cofact_algos;
static int n_cofact_algos;
static cofact_queue_t *cofact_queue;
static int cand_lpb[POLY_CNT];

static inline void queue_free(cofact_queue_t *x)
{
	if(x->batch)
		free(x->batch);
	free(x);
	return;
}

static inline cofact_queue_t *queue_alloc()
{
	cofact_queue_t *ret;

	ret = (cofact_queue_t *) malloc(sizeof(cofact_queue_t));
	assert(ret != NULL);
	memset(ret, 0, sizeof(cofact_queue_t));
	ret->batch = (candidate_t **) malloc(COFACT_BATCH_SIZE * sizeof(candidate_t *));
	assert(ret != NULL);
	return ret;
}

void cofact_process_queue(void)
{
	cofact_queue_t *queue_item;

	while(cofact_queue)
	{
		queue_item = cofact_queue;
		cofact_queue = queue_item->next;

		queue_item->algo->process(queue_item->algo, queue_item->batch, queue_item->n_batch);
		queue_free(queue_item);
	}
	return;
}

void cofact_queue_algo(cofact_algo_t *algo)
{
	cofact_queue_t *queue_item;

	queue_item = algo->queue;
	if(!queue_item->n_batch)
		return;
	queue_item->algo = algo;
	queue_item->next = cofact_queue;
	cofact_queue = queue_item;

	algo->queue = queue_alloc();
	return;
}

void cofact_flush(void)
{
	int size_idx, algo_idx;
	int has_work;

	do
	{
		has_work = 0;
		for(size_idx = MPZ_SIZE_IDX; size_idx >= 0; size_idx--)
		for(algo_idx = 0; algo_idx < n_cofact_algos; algo_idx++)
		{
			if(cofact_algos[size_idx][algo_idx].queue->n_batch)
				has_work++;
			else
				continue;
			cofact_queue_algo(&cofact_algos[size_idx][algo_idx]);
			cofact_process_queue();
		}
	} while(has_work);
	return;
}

static void pm1_ul128_process(cofact_algo_t *algo, candidate_t **batch, int n_batch);

void cofact_next_algo(candidate_t *cand, int algo_idx)
{
	int size, size_idx;
	cofact_algo_t *algo;
	cofact_queue_t *queue;

	if(cand->side >= sizeof(cand->rem) / sizeof(cand->rem[0]))
	{
		rel_count++;
		candidate_print(cand);
		candidate_free(cand);
		return;
	}
	if(mpz_cmp_ui(cand->rem[cand->side], 1) == 0)
	{
		cand->side++;
		cofact_next_algo(cand, 0);
		return;
	}

	size = mpz_sizeinbase(cand->rem[cand->side], 2);
	if(size <= cand_lpb[cand->side])
	{
		candidate_add_factor_mpz(cand, cand->side, cand->rem[cand->side]);
		cand->side++;
		cofact_next_algo(cand, 0);
		return;
	}

	if(algo_idx >= n_cofact_algos)
	{
		candidate_free(cand);
		return;
	}

	size_idx = (size + 1) / SIZE_INCREMENT;
	if(size_idx > MPZ_SIZE_IDX)
		size_idx = MPZ_SIZE_IDX;

	algo = &cofact_algos[size_idx][algo_idx];
	queue = algo->queue;

	queue->batch[queue->n_batch] = cand;
	queue->n_batch++;

	if(queue->n_batch >= COFACT_BATCH_SIZE)
		cofact_queue_algo(algo);
	return;
}

void cofact_add_candidate(candidate_t *cand)
{
	cand->side = 0;
	cofact_next_algo(cand, 0);
	return;
}

#if !(USE_OPENCL)
static void pm1_ul64_process(cofact_algo_t *algo, candidate_t **batch, int n_batch)
{
	mod64 n;
	ul64 f, X;
	candidate_t *cand;
	int i;
	pm1_plan_t *plan;
	mpz_t gf;
	double tin, tout;

	plan = (pm1_plan_t *) algo->plan;
	ul64_init(f);
	ul64_init(X);
	mod64_init(n);
	mpz_init(gf);

	tin = dbltime();
	for(i = 0; i < n_batch; i++)
	{
		cand = batch[i];

		mpz_get_ul64(n->n, cand->rem[cand->side]);
		mod64_set(n, n->n);
		ul64_set_ui(X, 0);
		pm1_stage1_ul64(f, X, n, plan);
		if(ul64_cmp_ui(f, 1) != 0 && ul64_cmp(f, n->n) != 0)
		{
			/* factor found in stage1 */
			mpz_set_ul64(gf, f);
#if PRINTF_FACTOR_FOUND
            gmp_printf("pm1s1(64) factor found: %Zd (mod %Zd)\n", gf, cand->rem[cand->side]);
#endif
			candidate_add_factor_mpz(cand, cand->side, gf);
			cofact_next_algo(cand, algo->algo_idx + 1);
			continue;
		}

		pm1_stage2_ul64(f, X, n, &plan->stage2);
		if(ul64_cmp_ui(f, 1) != 0 && ul64_cmp(f, n->n) != 0)
		{
			/* factor found in stage2 */
			mpz_set_ul64(gf, f);
#if PRINTF_FACTOR_FOUND
            gmp_printf("pm1s2(64) factor found: %Zd (mod %Zd)\n", gf, cand->rem[cand->side]);
#endif
			candidate_add_factor_mpz(cand, cand->side, gf);
		}
		cofact_next_algo(cand, algo->algo_idx + 1);
	}
	tout = dbltime();
	pm1_total += (tout - tin);

	mpz_clear(gf);
	mod64_clear(n);
	ul64_clear(X);
	ul64_clear(f);
	return;
}

static void pm1_ul128_process(cofact_algo_t *algo, candidate_t **batch, int n_batch)
{
	mod128 n;
	ul128 f, X;
	candidate_t *cand;
	int i;
	pm1_plan_t *plan;
	mpz_t gf;
	double tin, tout;

	plan = (pm1_plan_t *) algo->plan;
	ul128_init(f);
	ul128_init(X);
	mod128_init(n);
	mpz_init(gf);

	tin = dbltime();
	for(i = 0; i < n_batch; i++)
	{
		cand = batch[i];

		mpz_get_ul128(n->n, cand->rem[cand->side]);
		mod128_set(n, n->n);
		ul128_set_ui(X, 0);
		pm1_stage1_ul128(f, X, n, plan);
		if(ul64_cmp_ui(f, 1) != 0 && ul128_cmp(f, n->n) != 0)
		{
			/* factor found in stage1 */
			mpz_set_ul128(gf, f);
#if PRINTF_FACTOR_FOUND
            gmp_printf("pm1s1(128) factor found: %Zd (mod %Zd)\n", gf, cand->rem[cand->side]);
#endif
			candidate_add_factor_mpz(cand, cand->side, gf);
			cofact_next_algo(cand, algo->algo_idx + 1);
			continue;
		}

		pm1_stage2_ul128(f, X, n, &plan->stage2);
		if(ul128_cmp_ui(f, 1) != 0 && ul128_cmp(f, n->n) != 0)
		{
			/* factor found in stage2 */
			mpz_set_ul128(gf, f);
#if PRINTF_FACTOR_FOUND
            gmp_printf("pm1s2(128) factor found: %Zd (mod %Zd)\n", gf, cand->rem[cand->side]);
#endif
			candidate_add_factor_mpz(cand, cand->side, gf);
		}
		cofact_next_algo(cand, algo->algo_idx + 1);
	}
	tout = dbltime();
	pm1_total += (tout - tin);

	mpz_clear(gf);
	mod128_clear(n);
	ul128_clear(X);
	ul128_clear(f);
	return;
}
#endif /* !(USE_OPENCL) */

static void pm1_mpz_process(cofact_algo_t *algo, candidate_t **batch, int n_batch)
{
	modmpz_t n;
	mpz_t f, X;
	candidate_t *cand;
	int i;
	pm1_plan_t *plan;
	double tin, tout;

	plan = (pm1_plan_t *) algo->plan;
	mpz_init(f);
	mpz_init(X);
	mpzmod_init(n);

	tin = dbltime();
	for(i = 0; i < n_batch; i++)
	{
		cand = batch[i];

		mpz_set(n->n, cand->rem[cand->side]);
		mpzmod_set(n, n->n);
		mpz_set_ui(X, 0);
		pm1_stage1_mpz(f, X, n, plan);
		if(mpz_cmp_ui(f, 1) != 0 && mpz_cmp(f, n->n) != 0)
		{
			/* factor found in stage1 */
			candidate_add_factor_mpz(cand, cand->side, f);
			cofact_next_algo(cand, algo->algo_idx + 1);
			continue;
		}

		pm1_stage2_mpz(f, X, n, &plan->stage2);
		if(mpz_cmp_ui(f, 1) != 0 && mpz_cmp(f, n->n) != 0)
		{
			/* factor found in stage2 */
			candidate_add_factor_mpz(cand, cand->side, f);
		}
		cofact_next_algo(cand, algo->algo_idx + 1);
	}
	tout = dbltime();
	pm1_total += (tout - tin);

	mpzmod_clear(n);
	mpz_clear(X);
	mpz_clear(f);
	return;
}

#if !(USE_OPENCL)
static void pp1_ul64_process(cofact_algo_t *algo, candidate_t **batch, int n_batch)
{
	mod64 n;
	ul64 f, X;
	candidate_t *cand;
	int i;
	pp1_plan_t *plan;
	mpz_t gf;
	double tin, tout;

	plan = (pp1_plan_t *) algo->plan;
	ul64_init(f);
	ul64_init(X);
	mod64_init(n);
	mpz_init(gf);

	tin = dbltime();
	for(i = 0; i < n_batch; i++)
	{
		cand = batch[i];

		mpz_get_ul64(n->n, cand->rem[cand->side]);
		mod64_set(n, n->n);
		ul64_set_ui(X, 0);
		pp1_stage1_ul64(f, X, n, plan);
		if(ul64_cmp_ui(f, 1) != 0 && ul64_cmp(f, n->n) != 0)
		{
			/* factor found in stage1 */
			mpz_set_ul64(gf, f);
#if PRINTF_FACTOR_FOUND
            gmp_printf("pp1s1(64) factor found: %Zd (mod %Zd)\n", gf, cand->rem[cand->side]);
#endif
			candidate_add_factor_mpz(cand, cand->side, gf);
			cofact_next_algo(cand, algo->algo_idx + 1);
			continue;
		}

		pp1_stage2_ul64(f, X, n, &plan->stage2);
		if(ul64_cmp_ui(f, 1) != 0 && ul64_cmp(f, n->n) != 0)
		{
			/* factor found in stage2 */
			mpz_set_ul64(gf, f);
#if PRINTF_FACTOR_FOUND
            gmp_printf("pp1s2(64) factor found: %Zd (mod %Zd)\n", gf, cand->rem[cand->side]);
#endif
			candidate_add_factor_mpz(cand, cand->side, gf);
		}
		cofact_next_algo(cand, algo->algo_idx + 1);
	}
	tout = dbltime();
	pp1_total += (tout - tin);

	mpz_clear(gf);
	mod64_clear(n);
	ul64_clear(X);
	ul64_clear(f);
	return;
}

static void pp1_ul128_process(cofact_algo_t *algo, candidate_t **batch, int n_batch)
{
	mod128 n;
	ul128 f, X;
	candidate_t *cand;
	int i;
	pp1_plan_t *plan;
	mpz_t gf;
	double tin, tout;

	plan = (pp1_plan_t *) algo->plan;
	ul128_init(f);
	ul128_init(X);
	mod128_init(n);
	mpz_init(gf);

	tin = dbltime();
	for(i = 0; i < n_batch; i++)
	{
		cand = batch[i];

		mpz_get_ul128(n->n, cand->rem[cand->side]);
		mod128_set(n, n->n);
		ul128_set_ui(X, 0);
		pp1_stage1_ul128(f, X, n, plan);
		if(ul64_cmp_ui(f, 1) != 0 && ul128_cmp(f, n->n) != 0)
		{
			/* factor found in stage1 */
			mpz_set_ul128(gf, f);
#if PRINTF_FACTOR_FOUND
            gmp_printf("pp1s1(128) factor found: %Zd (mod %Zd)\n", gf, cand->rem[cand->side]);
#endif
			candidate_add_factor_mpz(cand, cand->side, gf);
			cofact_next_algo(cand, algo->algo_idx + 1);
			continue;
		}

		pp1_stage2_ul128(f, X, n, &plan->stage2);
		if(ul128_cmp_ui(f, 1) != 0 && ul128_cmp(f, n->n) != 0)
		{
			/* factor found in stage2 */
			mpz_set_ul128(gf, f);
#if PRINTF_FACTOR_FOUND
            gmp_printf("pp1s2(128) factor found: %Zd (mod %Zd)\n", gf, cand->rem[cand->side]);
#endif
			candidate_add_factor_mpz(cand, cand->side, gf);
		}
		cofact_next_algo(cand, algo->algo_idx + 1);
	}
	tout = dbltime();
	pp1_total += (tout - tin);

	mpz_clear(gf);
	mod128_clear(n);
	ul128_clear(X);
	ul128_clear(f);
	return;
}
#endif /* !(USE_OPENCL) */

static void pp1_mpz_process(cofact_algo_t *algo, candidate_t **batch, int n_batch)
{
	modmpz_t n;
	mpz_t f, X;
	candidate_t *cand;
	int i;
	pp1_plan_t *plan;
	double tin, tout;

	plan = (pp1_plan_t *) algo->plan;
	mpz_init(f);
	mpz_init(X);
	mpzmod_init(n);

	tin = dbltime();
	for(i = 0; i < n_batch; i++)
	{
		cand = batch[i];

		mpz_set(n->n, cand->rem[cand->side]);
		mpzmod_set(n, n->n);
		mpz_set_ui(X, 0);
		pp1_stage1_mpz(f, X, n, plan);
		if(mpz_cmp_ui(f, 1) != 0 && mpz_cmp(f, n->n) != 0)
		{
			/* factor found in stage1 */
			candidate_add_factor_mpz(cand, cand->side, f);
			cofact_next_algo(cand, algo->algo_idx + 1);
			continue;
		}

		pp1_stage2_mpz(f, X, n, &plan->stage2);
		if(mpz_cmp_ui(f, 1) != 0 && mpz_cmp(f, n->n) != 0)
		{
			/* factor found in stage2 */
			candidate_add_factor_mpz(cand, cand->side, f);
		}
		cofact_next_algo(cand, algo->algo_idx + 1);
	}
	tout = dbltime();
	pp1_total += (tout - tin);

	mpzmod_clear(n);
	mpz_clear(X);
	mpz_clear(f);
	return;
}

#if !(USE_OPENCL)
static void ecm_ul64_process(cofact_algo_t *algo, candidate_t **batch, int n_batch)
{
	mod64 n;
	ul64 f, b;
	ellM64_point_t X;
	candidate_t *cand;
	int i;
	ecm_plan_t *plan;
	mpz_t gf;
	double tin, tout;

	plan = (ecm_plan_t *) algo->plan;
	ul64_init(f);
	ul64_init(b);
	ul64_init(X->x);
	ul64_init(X->z);
	mod64_init(n);
	mpz_init(gf);

	tin = dbltime();
	for(i = 0; i < n_batch; i++)
	{
		cand = batch[i];

		mpz_get_ul64(n->n, cand->rem[cand->side]);
		mod64_set(n, n->n);
		ecm_stage1_ul64(f, X, b, n, plan);
		if(ul64_cmp_ui(f, 1) != 0 && ul64_cmp(f, n->n) != 0)
		{
			/* factor found in stage1 */
			mpz_set_ul64(gf, f);
			candidate_add_factor_mpz(cand, cand->side, gf);
			cofact_next_algo(cand, algo->algo_idx + 1);
			continue;
		}

		ecm_stage2_ul64(f, X, b, n, &plan->stage2);
		if(ul64_cmp_ui(f, 1) != 0 && ul64_cmp(f, n->n) != 0)
		{
			/* factor found in stage2 */
			mpz_set_ul64(gf, f);
			candidate_add_factor_mpz(cand, cand->side, gf);
		}
		cofact_next_algo(cand, algo->algo_idx + 1);
	}
	tout = dbltime();
	ecm_total += (tout - tin);

	mpz_clear(gf);
	mod64_clear(n);
	ul64_clear(X->x);
	ul64_clear(X->z);
	ul64_clear(b);
	ul64_clear(f);
	return;
}

static void ecm_ul128_process(cofact_algo_t *algo, candidate_t **batch, int n_batch)
{
	mod128 n;
	ul128 f, b;
	ellM128_point_t X;
	candidate_t *cand;
	int i;
	ecm_plan_t *plan;
	mpz_t gf;
	double tin, tout;

	plan = (ecm_plan_t *) algo->plan;
	ul128_init(f);
	ul128_init(b);
	ul128_init(X->x);
	ul128_init(X->z);
	mod128_init(n);
	mpz_init(gf);

	tin = dbltime();
	for(i = 0; i < n_batch; i++)
	{
		cand = batch[i];

		mpz_get_ul128(n->n, cand->rem[cand->side]);
		mod128_set(n, n->n);
		ecm_stage1_ul128(f, X, b, n, plan);
		if(ul128_cmp_ui(f, 1) != 0 && ul128_cmp(f, n->n) != 0)
		{
			/* factor found in stage1 */
			mpz_set_ul128(gf, f);
			candidate_add_factor_mpz(cand, cand->side, gf);
			cofact_next_algo(cand, algo->algo_idx + 1);
			continue;
		}

		ecm_stage2_ul128(f, X, b, n, &plan->stage2);
		if(ul128_cmp_ui(f, 1) != 0 && ul128_cmp(f, n->n) != 0)
		{
			/* factor found in stage2 */
			mpz_set_ul128(gf, f);
			candidate_add_factor_mpz(cand, cand->side, gf);
		}
		cofact_next_algo(cand, algo->algo_idx + 1);
	}
	tout = dbltime();
	ecm_total += (tout - tin);

	mpz_clear(gf);
	mod128_clear(n);
	ul128_clear(X->x);
	ul128_clear(X->z);
	ul128_clear(b);
	ul128_clear(f);
	return;
}
#endif /* !(USE_OPENCL) */

static void ecm_mpz_process(cofact_algo_t *algo, candidate_t **batch, int n_batch)
{
	modmpz_t n;
	mpz_t f, b;
	ellMmpz_point_t X;
	candidate_t *cand;
	int i;
	ecm_plan_t *plan;
	double tin, tout;

	plan = (ecm_plan_t *) algo->plan;
	mpz_init(f);
	mpz_init(b);
	mpz_init(X->x);
	mpz_init(X->z);
	mpzmod_init(n);

	tin = dbltime();
	for(i = 0; i < n_batch; i++)
	{
		cand = batch[i];

		mpz_set(n->n, cand->rem[cand->side]);
		mpzmod_set(n, n->n);
		ecm_stage1_mpz(f, X, b, n, plan);
		if(mpz_cmp_ui(f, 1) != 0 && mpz_cmp(f, n->n) != 0)
		{
			/* factor found in stage1 */
			candidate_add_factor_mpz(cand, cand->side, f);
			cofact_next_algo(cand, algo->algo_idx + 1);
			continue;
		}

		ecm_stage2_mpz(f, X, b, n, &plan->stage2);
		if(mpz_cmp_ui(f, 1) != 0 && mpz_cmp(f, n->n) != 0)
		{
			/* factor found in stage2 */
			candidate_add_factor_mpz(cand, cand->side, f);
		}
		cofact_next_algo(cand, algo->algo_idx + 1);
	}
	tout = dbltime();
	ecm_total += (tout - tin);

	mpzmod_clear(n);
	mpz_clear(X->x);
	mpz_clear(X->z);
	mpz_clear(b);
	mpz_clear(f);
	return;
}

#define LPB_MAX 33

static int nb_curves(const unsigned int lpb)
{
	/* the following table, computed with the proba_cofactor() function in the
	 * facul.sage file, ensures a probability of at least about 90% to find a
	 * factor below 2^lpb with n = T[lpb]
	 */
	int T[LPB_MAX+1] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, /* 0-9 */
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, /* 10-19 */
		0, 0, 1 /*22:0.9074*/, 2 /*23:0.9059*/, 3 /*24:0.8990*/,
		5 /*25:0.9194*/, 6 /*26:0.9065*/, 8 /*27:0.9053*/,
		10 /*28:0.9010*/, 13 /*29:0.9091*/, 16 /*30:0.9134*/,
		18 /*31:0.9039*/, 21 /*32:0.9076*/, 24/*33:0.8963*/ };
	return (lpb <= LPB_MAX) ? T[lpb] : T[LPB_MAX];
}

void cofact_init(gls_config_t cfg)
{
	unsigned int i, j, lpb, n;

	cofact_queue = NULL;
	for(i = 0; i < sizeof(cfg->lpb) / sizeof(cfg->lpb[0]); i++)
	{
		cand_lpb[i] = cfg->lpb[i];
		if(lpb < cfg->lpb[i])
			lpb = cfg->lpb[i];
	}
	n = nb_curves(cfg->lpb[APOLY_IDX]);
	n_cofact_algos = n + 3;

	cofact_algos = (cofact_algo_t **) malloc(COFACT_SIZES * sizeof(cofact_algo_t *));
	for(i = 0; i < COFACT_SIZES; i++)
	{
		cofact_algos[i] = (cofact_algo_t *) malloc(n_cofact_algos * sizeof(cofact_algo_t));
		memset(cofact_algos[i], 0, n_cofact_algos * sizeof(cofact_algo_t));
		for(j = 0; j < n_cofact_algos; j++)
		{
			cofact_algos[i][j].queue = queue_alloc();
		}
	}

#if USE_OPENCL
	int PP1_STAGE2_XJ_LEN = 0;
	int ECM_COMMONZ_T_LEN = 0;
	int ECM_STAGE2_PID_LEN = 0;
	int ECM_STAGE2_PJ_LEN = 0;

	/* pm1 */
	cofact_algos[0][0].process = pm1_ul32_process_ocl;
	cofact_algos[0][0].plan = malloc(sizeof(pm1_plan_t));
	pm1_plan_init(cofact_algos[0][0].plan, 315, 2205);
	cofact_algos[0][0].algo_idx = 0;
    PP1_STAGE2_XJ_LEN = ((pm1_plan_t *)cofact_algos[0][0].plan)->stage2.n_S1;

    cofact_algos[1][0].process = pm1_ul64_process_ocl;
    cofact_algos[1][0].plan = cofact_algos[0][0].plan;
    cofact_algos[1][0].algo_idx = 0;

	cofact_algos[2][0].process = pm1_ul96_process_ocl;
	cofact_algos[2][0].plan = cofact_algos[0][0].plan;
	cofact_algos[2][0].algo_idx = 0;

	cofact_algos[3][0].process = pm1_ul128_process_ocl;
	cofact_algos[3][0].plan = cofact_algos[0][0].plan;
	cofact_algos[3][0].algo_idx = 0;

	cofact_algos[4][0].process = pm1_ul160_process_ocl;
	cofact_algos[4][0].plan = cofact_algos[0][0].plan;
	cofact_algos[4][0].algo_idx = 0;

	cofact_algos[5][0].process = pm1_ul192_process_ocl;
	cofact_algos[5][0].plan = cofact_algos[0][0].plan;
	cofact_algos[5][0].algo_idx = 0;

	cofact_algos[6][0].process = pm1_ul224_process_ocl;
	cofact_algos[6][0].plan = cofact_algos[0][0].plan;
	cofact_algos[6][0].algo_idx = 0;

	cofact_algos[7][0].process = pm1_ul256_process_ocl;
	cofact_algos[7][0].plan = cofact_algos[0][0].plan;
	cofact_algos[7][0].algo_idx = 0;

	cofact_algos[8][0].process = pm1_mpz_process;
	cofact_algos[8][0].plan = cofact_algos[0][0].plan;
	cofact_algos[8][0].algo_idx = 0;

	/* pp1 */
	cofact_algos[0][1].process = pp1_ul32_process_ocl;
	cofact_algos[0][1].plan = malloc(sizeof(pp1_plan_t));
	pp1_plan_init(cofact_algos[0][1].plan, 525, 3255);
	cofact_algos[0][1].algo_idx = 1;
    PP1_STAGE2_XJ_LEN = MAX(PP1_STAGE2_XJ_LEN, ((pp1_plan_t *)cofact_algos[0][1].plan)->stage2.n_S1);

	cofact_algos[1][1].process = pp1_ul64_process_ocl;
	cofact_algos[1][1].plan = cofact_algos[0][1].plan;
	cofact_algos[1][1].algo_idx = 1;

	cofact_algos[2][1].process = pp1_ul96_process_ocl;
	cofact_algos[2][1].plan = cofact_algos[0][1].plan;
	cofact_algos[2][1].algo_idx = 1;

	cofact_algos[3][1].process = pp1_ul128_process_ocl;
	cofact_algos[3][1].plan = cofact_algos[0][1].plan;
	cofact_algos[3][1].algo_idx = 1;

	cofact_algos[4][1].process = pp1_ul160_process_ocl;
	cofact_algos[4][1].plan = cofact_algos[0][1].plan;
	cofact_algos[4][1].algo_idx = 1;

	cofact_algos[5][1].process = pp1_ul192_process_ocl;
	cofact_algos[5][1].plan = cofact_algos[0][1].plan;
	cofact_algos[5][1].algo_idx = 1;

	cofact_algos[6][1].process = pp1_ul224_process_ocl;
	cofact_algos[6][1].plan = cofact_algos[0][1].plan;
	cofact_algos[6][1].algo_idx = 1;

	cofact_algos[7][1].process = pp1_ul256_process_ocl;
	cofact_algos[7][1].plan = cofact_algos[0][1].plan;
	cofact_algos[7][1].algo_idx = 1;

	cofact_algos[8][1].process = pp1_mpz_process;
	cofact_algos[8][1].plan = cofact_algos[0][1].plan;
	cofact_algos[8][1].algo_idx = 1;

	/* ecm */
	cofact_algos[0][2].process = ecm_ul32_process_ocl;
	cofact_algos[0][2].plan = malloc(sizeof(ecm_plan_t));
	ecm_plan_init(cofact_algos[0][2].plan, 105, 3255, MONTY12, 2);
	cofact_algos[0][2].algo_idx = 2;
    {
        ecm_plan_t *_ecm_plan = (ecm_plan_t *)cofact_algos[0][2].plan;
        ECM_COMMONZ_T_LEN = (_ecm_plan->stage2.n_S1) + (_ecm_plan->stage2.i1 - _ecm_plan->stage2.i0 - ((_ecm_plan->stage2.i0 == 0) ? 1 : 0));
        ECM_STAGE2_PID_LEN = _ecm_plan->stage2.i1 - _ecm_plan->stage2.i0;
        ECM_STAGE2_PJ_LEN = _ecm_plan->stage2.n_S1;
    }

	cofact_algos[1][2].process = ecm_ul64_process_ocl;
	cofact_algos[1][2].plan = cofact_algos[0][2].plan;
	cofact_algos[1][2].algo_idx = 2;

	cofact_algos[2][2].process = ecm_ul96_process_ocl;
	cofact_algos[2][2].plan = cofact_algos[0][2].plan;
	cofact_algos[2][2].algo_idx = 2;

	cofact_algos[3][2].process = ecm_ul128_process_ocl;
	cofact_algos[3][2].plan = cofact_algos[0][2].plan;
	cofact_algos[3][2].algo_idx = 2;

	cofact_algos[4][2].process = ecm_ul160_process_ocl;
	cofact_algos[4][2].plan = cofact_algos[0][2].plan;
	cofact_algos[4][2].algo_idx = 2;

	cofact_algos[5][2].process = ecm_ul192_process_ocl;
	cofact_algos[5][2].plan = cofact_algos[0][2].plan;
	cofact_algos[5][2].algo_idx = 2;

	cofact_algos[6][2].process = ecm_ul224_process_ocl;
	cofact_algos[6][2].plan = cofact_algos[0][2].plan;
	cofact_algos[6][2].algo_idx = 2;

	cofact_algos[7][2].process = ecm_ul256_process_ocl;
	cofact_algos[7][2].plan = cofact_algos[0][2].plan;
	cofact_algos[7][2].algo_idx = 2;

	cofact_algos[8][2].process = ecm_mpz_process;
	cofact_algos[8][2].plan = cofact_algos[0][2].plan;
	cofact_algos[8][2].algo_idx = 2;

	if(n > 0)
	{
		cofact_algos[0][3].process = ecm_ul32_process_ocl;
		cofact_algos[0][3].plan = malloc(sizeof(ecm_plan_t));
		ecm_plan_init(cofact_algos[0][3].plan, 315, 5355, BRENT12, 11);
		cofact_algos[0][3].algo_idx = 3;
        {
            ecm_plan_t *_ecm_plan = (ecm_plan_t *)cofact_algos[0][3].plan;
            int _ECM_COMMONZ_T_LEN = (_ecm_plan->stage2.n_S1) + (_ecm_plan->stage2.i1 - _ecm_plan->stage2.i0 - ((_ecm_plan->stage2.i0 == 0) ? 1 : 0));
            int _ECM_STAGE2_PID_LEN = _ecm_plan->stage2.i1 - _ecm_plan->stage2.i0;
            int _ECM_STAGE2_PJ_LEN = _ecm_plan->stage2.n_S1;
            ECM_COMMONZ_T_LEN = MAX(ECM_COMMONZ_T_LEN, _ECM_COMMONZ_T_LEN);
            ECM_STAGE2_PID_LEN = MAX(ECM_STAGE2_PID_LEN, _ECM_STAGE2_PID_LEN);
            ECM_STAGE2_PJ_LEN = MAX(ECM_STAGE2_PJ_LEN, _ECM_STAGE2_PJ_LEN);
        }

	    cofact_algos[1][3].process = ecm_ul64_process_ocl;
	    cofact_algos[1][3].plan = cofact_algos[0][3].plan;
	    cofact_algos[1][3].algo_idx = 3;

	    cofact_algos[2][3].process = ecm_ul96_process_ocl;
	    cofact_algos[2][3].plan = cofact_algos[0][3].plan;
	    cofact_algos[2][3].algo_idx = 3;

	    cofact_algos[3][3].process = ecm_ul128_process_ocl;
	    cofact_algos[3][3].plan = cofact_algos[0][3].plan;
	    cofact_algos[3][3].algo_idx = 3;

	    cofact_algos[4][3].process = ecm_ul160_process_ocl;
	    cofact_algos[4][3].plan = cofact_algos[0][3].plan;
	    cofact_algos[4][3].algo_idx = 3;

	    cofact_algos[5][3].process = ecm_ul192_process_ocl;
	    cofact_algos[5][3].plan = cofact_algos[0][3].plan;
	    cofact_algos[5][3].algo_idx = 3;

	    cofact_algos[6][3].process = ecm_ul224_process_ocl;
	    cofact_algos[6][3].plan = cofact_algos[0][3].plan;
	    cofact_algos[6][3].algo_idx = 3;

	    cofact_algos[7][3].process = ecm_ul256_process_ocl;
	    cofact_algos[7][3].plan = cofact_algos[0][3].plan;
	    cofact_algos[7][3].algo_idx = 3;

	    cofact_algos[8][3].process = ecm_mpz_process;
	    cofact_algos[8][3].plan = cofact_algos[0][3].plan;
	    cofact_algos[8][3].algo_idx = 3;
	}
#else /* USE_OPENCL */
	/* pm1 */
	cofact_algos[0][0].process = pm1_ul64_process;
	cofact_algos[0][0].plan = malloc(sizeof(pm1_plan_t));
	pm1_plan_init(cofact_algos[0][0].plan, 315, 2205);
	cofact_algos[0][0].algo_idx = 0;

	cofact_algos[1][0].process = pm1_ul128_process;
	cofact_algos[1][0].plan = cofact_algos[0][0].plan;
	cofact_algos[1][0].algo_idx = 0;

	cofact_algos[2][0].process = pm1_mpz_process;
	cofact_algos[2][0].plan = cofact_algos[0][0].plan;
	cofact_algos[2][0].algo_idx = 0;

	/* pp1 */
	cofact_algos[0][1].process = pp1_ul64_process;
	cofact_algos[0][1].plan = malloc(sizeof(pp1_plan_t));
	pp1_plan_init(cofact_algos[0][1].plan, 525, 3255);
	cofact_algos[0][1].algo_idx = 1;

	cofact_algos[1][1].process = pp1_ul128_process;
	cofact_algos[1][1].plan = cofact_algos[0][1].plan;
	cofact_algos[1][1].algo_idx = 1;

	cofact_algos[2][1].process = pp1_mpz_process;
	cofact_algos[2][1].plan = cofact_algos[0][1].plan;
	cofact_algos[2][1].algo_idx = 1;

	/* ecm */
	cofact_algos[0][2].process = ecm_ul64_process;
	cofact_algos[0][2].plan = malloc(sizeof(ecm_plan_t));
	ecm_plan_init(cofact_algos[0][2].plan, 105, 3255, MONTY12, 2);
	cofact_algos[0][2].algo_idx = 2;

	cofact_algos[1][2].process = ecm_ul128_process;
	cofact_algos[1][2].plan = cofact_algos[0][2].plan;
	cofact_algos[1][2].algo_idx = 2;

	cofact_algos[2][2].process = ecm_mpz_process;
	cofact_algos[2][2].plan = cofact_algos[0][2].plan;
	cofact_algos[2][2].algo_idx = 2;

	if(n > 0)
	{
		cofact_algos[0][3].process = ecm_ul64_process;
		cofact_algos[0][3].plan = malloc(sizeof(ecm_plan_t));
		ecm_plan_init(cofact_algos[0][3].plan, 315, 5355, BRENT12, 11);
		cofact_algos[0][3].algo_idx = 3;

		cofact_algos[1][3].process = ecm_ul128_process;
		cofact_algos[1][3].plan = cofact_algos[0][3].plan;
		cofact_algos[1][3].algo_idx = 3;

		cofact_algos[2][3].process = ecm_mpz_process;
		cofact_algos[2][3].plan = cofact_algos[0][3].plan;
		cofact_algos[2][3].algo_idx = 3;
	}
#endif /* USE_OPENCL */

	/* heuristic strategy where B1 is increased by sqrt(B1) at each curve */
	double B1 = 105.0;
	for (i = 4; i < n + 3; i++)
	{
		double B2;
		unsigned int k;

		B1 += sqrt (B1);
		B2 = 17.0 * B1;
		/* we round B2 to (2k+1)*105, thus k is the integer nearest to B2/210-0.5 */
		k = B2 / 210.0;

#if USE_OPENCL
		cofact_algos[0][i].process = ecm_ul32_process_ocl;
		cofact_algos[0][i].plan = malloc(sizeof(ecm_plan_t));
		ecm_plan_init(cofact_algos[0][i].plan, (unsigned int) B1, (2 * k + 1) * 105, MONTY12, i - 1);
		cofact_algos[0][i].algo_idx = i;
        {
            ecm_plan_t *_ecm_plan = (ecm_plan_t *)cofact_algos[0][i].plan;
            int _ECM_COMMONZ_T_LEN = (_ecm_plan->stage2.n_S1) + (_ecm_plan->stage2.i1 - _ecm_plan->stage2.i0 - ((_ecm_plan->stage2.i0 == 0) ? 1 : 0));
            int _ECM_STAGE2_PID_LEN = _ecm_plan->stage2.i1 - _ecm_plan->stage2.i0;
            int _ECM_STAGE2_PJ_LEN = _ecm_plan->stage2.n_S1;
            ECM_COMMONZ_T_LEN = MAX(ECM_COMMONZ_T_LEN, _ECM_COMMONZ_T_LEN);
            ECM_STAGE2_PID_LEN = MAX(ECM_STAGE2_PID_LEN, _ECM_STAGE2_PID_LEN);
            ECM_STAGE2_PJ_LEN = MAX(ECM_STAGE2_PJ_LEN, _ECM_STAGE2_PJ_LEN);
        }

		cofact_algos[1][i].process = ecm_ul64_process_ocl;
		cofact_algos[1][i].plan = cofact_algos[0][i].plan;
		cofact_algos[1][i].algo_idx = i;

		cofact_algos[2][i].process = ecm_ul96_process_ocl;
		cofact_algos[2][i].plan = cofact_algos[0][i].plan;
		cofact_algos[2][i].algo_idx = i;

		cofact_algos[3][i].process = ecm_ul128_process_ocl;
		cofact_algos[3][i].plan = cofact_algos[0][i].plan;
		cofact_algos[3][i].algo_idx = i;

		cofact_algos[4][i].process = ecm_ul160_process_ocl;
		cofact_algos[4][i].plan = cofact_algos[0][i].plan;
		cofact_algos[4][i].algo_idx = i;

		cofact_algos[5][i].process = ecm_ul192_process_ocl;
		cofact_algos[5][i].plan = cofact_algos[0][i].plan;
		cofact_algos[5][i].algo_idx = i;

		cofact_algos[6][i].process = ecm_ul224_process_ocl;
		cofact_algos[6][i].plan = cofact_algos[0][i].plan;
		cofact_algos[6][i].algo_idx = i;

		cofact_algos[7][i].process = ecm_ul256_process_ocl;
		cofact_algos[7][i].plan = cofact_algos[0][i].plan;
		cofact_algos[7][i].algo_idx = i;

		cofact_algos[8][i].process = ecm_mpz_process;
		cofact_algos[8][i].plan = cofact_algos[0][i].plan;
		cofact_algos[8][i].algo_idx = i;
#else /* USE_OPENCL */
		cofact_algos[0][i].process = ecm_ul64_process;
		cofact_algos[0][i].plan = malloc(sizeof(ecm_plan_t));
		ecm_plan_init(cofact_algos[0][i].plan, (unsigned int) B1, (2 * k + 1) * 105, MONTY12, i - 1);
		cofact_algos[0][i].algo_idx = i;

		cofact_algos[1][i].process = ecm_ul128_process;
		cofact_algos[1][i].plan = cofact_algos[0][i].plan;
		cofact_algos[1][i].algo_idx = i;

		cofact_algos[2][i].process = ecm_mpz_process;
		cofact_algos[2][i].plan = cofact_algos[0][i].plan;
		cofact_algos[2][i].algo_idx = i;
#endif /* USE_OPENCL */
	}
	assert(i == n_cofact_algos);

#if USE_OPENCL
    /* Construct build arguments */
	const char *config_mp_source = "las/ocl/las.cl";            /* File name of kernel source */

    char *build_opts = NULL;
    {
		int build_opts_len =
				snprintf(NULL, 0, "%s -D PP1_STAGE2_XJ_LEN=%d -D ECM_COMMONZ_T_LEN=%d -D ECM_STAGE2_PID_LEN=%d -D ECM_STAGE2_PJ_LEN=%d",
						 ocl_state.buildopts,
						 PP1_STAGE2_XJ_LEN,
						 ECM_COMMONZ_T_LEN,
						 ECM_STAGE2_PID_LEN,
						 ECM_STAGE2_PJ_LEN);

		build_opts_len++; /* snprintf does not include null byte in ret value */
		build_opts = (char *)malloc(build_opts_len);

		snprintf(build_opts, build_opts_len, "%s -D PP1_STAGE2_XJ_LEN=%d -D ECM_COMMONZ_T_LEN=%d -D ECM_STAGE2_PID_LEN=%d -D ECM_STAGE2_PJ_LEN=%d",
				 ocl_state.buildopts,
				 PP1_STAGE2_XJ_LEN,
				 ECM_COMMONZ_T_LEN,
				 ECM_STAGE2_PID_LEN,
				 ECM_STAGE2_PJ_LEN);
		printf("build_opts=%s\n", build_opts);
    }
    
    /* Now do the build */
    ocl_build(&ocl_state, config_mp_source, build_opts);
    
    free(build_opts);
#endif /* USE_OPENCL */

	return;
}
