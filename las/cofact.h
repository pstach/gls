/*
 * cofact.h
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */

#ifndef COFACT_H_
#define COFACT_H_
#include <stdint.h>
#include "las.h"
#include "gls_config.h"
#include "cofact_plan.h"

struct cofact_algo_s;
typedef struct cofact_algo_s cofact_algo_t;
struct cofact_queue_s;
typedef struct cofact_queue_s cofact_queue_t;

struct cofact_algo_s {
	void (*process)(cofact_algo_t *algo, candidate_t **batch, int n_batch);
	void *plan;
	cofact_queue_t *queue;
	int algo_idx;
};

struct cofact_queue_s {
	cofact_algo_t *algo;
	candidate_t **batch;
	int n_batch;
	cofact_queue_t *next;
};

/* Return -1 if the leftover norm n cannot yield a relation.
 *
 * Possible cases, where qj represents a prime in [B,L], and rj a prime > L:
 * (0) n >= 2^mfb
 * (a) n < L:           1 or q1
 * (b) L < n < B^2:     r1 -> cannot yield a relation
 * (c) B^2 < n < B*L:   r1 or q1*q2
 * (d) B*L < n < L^2:   r1 or q1*q2 or q1*r2
 * (e) L^2 < n < B^3:   r1 or q1*r2 or r1*r2 -> cannot yield a relation
 * (f) B^3 < n < B^2*L: r1 or q1*r2 or r1*r2 or q1*q2*q3
 * (g) B^2*L < n < L^3: r1 or q1*r2 or r1*r2
 * (h) L^3 < n < B^4:   r1 or q1*r2, r1*r2 or q1*q2*r3 or q1*r2*r3 or r1*r2*r3
 */

static inline int check_leftover_norm(candidate_t *cand, gls_config_t cfg, int side)
{
	size_t s = mpz_sizeinbase(cand->rem[side], 2);
	unsigned int lpb = cfg->lpb[side];
	unsigned int mfb = cfg->mfb[side];

	if(s > mfb)
		return -1; /* n has more than mfb bits, which is the given limit */
	/* now n < 2^mfb */
	if(s <= lpb)
		return 0; /* case (a) */
	/* Note also that in the descent case where L > B^2, if we're below L
	 * it's still fine of course, but we have no guarantee that our
	 * cofactor is prime...
	 */
	/* now n >= L=2^lpb */
	if(mpz_cmp(cand->rem[side], cfg->BB[side]) < 0)
		return -1; /* case (b) */
	/* now n >= B^2 */
	if(2 * lpb < s)
	{
		if(mpz_cmp(cand->rem[side], cfg->BBB[side]) < 0)
			return -1; /* case (e) */
		if(3 * lpb < s && mpz_cmp(cand->rem[side], cfg->BBBB[side]) < 0)
			return -1; /* case (h) */
	}

	if(mpz_probab_prime_p(cand->rem[side], 1))
		return -1; /* n is a pseudo-prime larger than L */
	return 0;
}

extern void cofact_process_queue(void);
extern void cofact_next_algo(candidate_t *cand, int algo_idx);
extern void cofact_flush(void);
extern void cofact_add_candidate(candidate_t *cand);
extern void cofact_init(gls_config_t cfg);

#endif  /* COFACT_H_ */
