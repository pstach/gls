/*
 * cofact_plan.c
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>
#include "cofact_plan.h"
#include "prac_bc.h"

//#define VERBOSE_PLAN 1

static inline u_int64_t euler_phi(u_int64_t n)
{
	u_int64_t p, r;

	if(!n)
		return 0;
	r = 1;

	if(n % 2 == 0)
	{
		n /= 2;
		while((n % 2) == 0)
		{
			n /= 2;
			r *= 2;
		}
	}

	for(p = 3; p * p <= n; p += 2)
	{
		if((n % p) == 0)
		{
			n /= p;
			r *= (p - 1);
			while((n % p) == 0)
			{
				n /= p;
				r *= p;
			}
		}
	}
	if(n > 1)
		r *= (n - 1);
	return r;
}

static inline u_int64_t gcd(u_int64_t x, u_int64_t y)
{
	u_int64_t a, b;
	u_int64_t tmp;

	if(x < y)
	{
		a = x;
		b = y;
	}
	else
	{
		a = y;
		b = x;
	}

	while(b)
	{
		tmp = a % b;
		a = b;
		b = tmp;
	}
	return a;
}

/* Tries to find a factor p of n such that primes[p] != 0.
 * We assume p is the largest prime factor of n.
 * If successful, return p, otherwise return 0.
 */
static unsigned int find_prime_div(const unsigned int n, const unsigned int min_f, const unsigned char *primes)
{
	unsigned int p = n, f = min_f;

	while (p >= f * f)
	{
		while (p % f == 0)
		{
			p /= f;
			if(primes[p])
				return p;
		}
		f += 2; /* Assumes n even */
	}

	if(primes[p])
		return p;
	return 0;
}

void stage2_plan_init(stage2_plan_t *plan, u_int32_t B2_min, u_int32_t B2)
{
	unsigned int p, n_primes, n_pairs;
	unsigned int i, i_min, i_max, j, d, m, n, not_in_d;
	u_int8_t *primes;
	int need_NEXT_D;
	const int composite_pairing = 1;
	mpz_t P;

	plan->B2 = B2;
	if(B2 <= B2_min)
	{
		plan->d = 0;
		plan->n_S1 = 0;
		plan->S1 = NULL;
		plan->pairs = NULL;
		return;
	}

	mpz_init(P);
	/* Choose a value for d. Should depend on B2-B2min, for a start we fix d=210 */
	d = 210;
	plan->d = d;
	not_in_d = 11; /* the smallest prime not in d */

	/* List of the j values for which we need to precompute V_j(x + 1/x) */
	plan->n_S1 = (u_int32_t) (euler_phi((u_int64_t) d) / 2);
	/* TODO: can these indexes be 16 bit given more appropriate values of d? saves on GPU shared memory */
	plan->S1 = (u_int32_t *) malloc(plan->n_S1 * sizeof(u_int32_t));

	for(i = 0, j = 1; j < d / 2; j += 2)
	{
		if(gcd(j, d) == 1)
			plan->S1[i++] = j;
	}
#ifdef VERBOSE_PLAN
	printf("S_1 = {");
	for(i = 0; i < plan->n_S1; i++)
		printf("%d%s", plan->S1[i], (i + 1 < plan->n_S1) ? ", " : "");
	printf("}\n");
#endif
	/* Preliminary choice for the smallest and largest i value we might need.
	 * The smallest may increase yet due to pairing with composite values.
	 * p > B2min, so i*d > B2min - max(S_1), or i >= ceil((B2min - max(S_1)) / d).
	 * For now we have max(S_1) < d/2, so we can write i >= floor((B2min + d/2) / d)
	 * p <= B2, so i*d +- j <= B2, so i*d - j <= B2, so i*d <= B2 + max(S_1),
	 * or i <= floor((B2 + max(S_1)) / d).
	 */
	i_min = (B2_min + d / 2) / d;
	i_max = (B2 + d / 2) / d;

#ifdef VERBOSE_PLAN
	printf("Initial choice for min_i = %u, max_i = %u\n", i_min, i_max);
#endif

	/* Generate the list of pairs for stage 2 */
	/* For each prime B2min < p <= B2, we write p = id-j, j in S1,
	 * and write into *pairs the index of j within the S_2 array.
	 * When it's time to increase i, NEXT_D is written to *pairs.
	 * NEXT_PASS is the signal to end stage 2.
	 */

	/* Make array where entry at index p, prime p with B2min < p <= B2, are
	 * set to 1. The largest value we can write as i*d+j with i < max_i and
	 * j < d/2 is max_i * d + d/2 - 1
	 */
	primes = (u_int8_t *) malloc(i_max * d + d/2);
	memset(primes, 0, i_max * d + d/2);
	n_primes = 0;

	mpz_set_ui(P, B2_min);
	for(mpz_nextprime(P, P); mpz_cmp_ui(P, B2) <= 0; mpz_nextprime(P, P))
	{
		n_primes++;
		primes[mpz_get_ui(P)] = 1;
	}

	/* We need at most one pair per prime, plus the number of NEXT_D and NEXT_PASS codes */
    plan->n_pairs = n_primes + (B2 - B2_min) / d + 1;
	plan->pairs = (u_int8_t *) malloc(plan->n_pairs);

	/* Lower max_i so that max_i * d +- j, 0 < j < d/2, actually includes any primes */
	for(i = i_max; i >= i_min; i--)
	{
		for (m = 0; m < plan->n_S1; m++)
		{
			j = plan->S1[m];
			if ((i * d >= j && primes[i * d - j]) || primes[i * d + j])
			{
#ifdef VERBOSE_PLAN
				printf("Final max_i = %u, found prime %u or %u\n",
						i, i*d - j, i*d + j);
#endif
				break;
			}
		}
		if (m < plan->n_S1)
			break;
	}
	i_max = i;

	n = 0;
	n_pairs = 0;
	need_NEXT_D = 0;

	if(composite_pairing)
	{
		/* Do a pass over the primes in reverse, flagging off those primes
		 * that are included as composite values id+-j, where id-+j is prime.
		 * The smallest prime not in d is not_in_d, so any proper divisor of
		 * i*d +- j is at most (i*d+d/2+1) / not_in_d
		 */
		for(p = B2; p >= (i_max * d + d/2 - 1) / not_in_d; p--)
		{
			if(primes[p])
			{
				unsigned int q, r;
				int sj;
				/* p = id + j or id - j with 0 < j < d/2 */
				j = p % d;
				sj = (int) j - ((j > d / 2) ? (int) d : 0);
				/* p = i*d + sj, q = i*d - sj = p - 2*sj */
				if((int) p < 2 * sj)
					continue;
				q = (int) p - 2 * sj;

				if(!primes[q]) /* If q is not a prime, see if a stage 2 prime divides q */
				{
					r = find_prime_div (q, not_in_d, primes);
					if(r)
					{
#ifdef VERBOSE_PLAN
						printf("Including %u as factor of %u = %u * %u "
								"+ %d which pairs with prime %u = %u"
								" * %u + %d\n",
								r, q, (p - sj) / d, d, -sj, p,
								(p-sj)/d, d, sj);
#endif
						primes[r] = 0;
					}
				}
			}
		}

		/* Some small primes may remain. Go through all possible (i,j)
		 * values in order of decreasing id+j, and choose those that
		 * cover two small primes
		 */
		for(i = i_max; i >= i_min && i > 0; i--)
		{
			for(m = 0; m < plan->n_S1; m++)
			{
				unsigned int q;

				j = plan->S1[m];
				p = i * d - j;
				q = i * d + j;

				if(!primes[q] && !primes[q])
				{
					unsigned int r1, r2;

					r1 = find_prime_div(p, not_in_d, primes);
					r2 = find_prime_div(q, not_in_d, primes);

					if(r1 && r2)
					{
						/* Flag on one of the two "primes" that include these two composite values */
						primes[p] = 1;
						primes[r1] = 0;
						primes[r2] = 0;
#ifdef VERBOSE_PLAN
						printf("Including %u * %u +- %u which includes "
								"%u and %u as factors\n",
								i, d, j, r1, r2);
#endif
					}
				}
			}
		}

		/* Some small primes may still remain. Go through all possible (i,j)
		 * values in order of decreasing id+j again, and choose those that
		 * cover at least one prime
		 */
		for(i = i_max; i >= i_min && i > 0; i--)
		{
			for(m = 0; m < plan->n_S1; m++)
			{
				unsigned int q;

				j = plan->S1[m];
				p = i * d - j;
				q = i * d + j;

				if(!primes[p] && !primes[q])
				{
					unsigned int r1, r2;

					r1 = find_prime_div(p, not_in_d, primes);
					r2 = find_prime_div(q, not_in_d, primes);

					if(r1 || r2)
					{
						/* Flag on one of the two "primes" that include this composite value */
						primes[p] = 1;
						primes[r1] = 0;
						primes[r2] = 0;
#ifdef VERBOSE_PLAN
						printf("Including %u * %u +- %u which includes "
								"%u as factors\n",
								i, d, j, (r1 == 0) ? r2 : r1);
#endif
					}
				}
			}
		}
	}

	/* Increase min_i so that min_i * d +- j, 0 < j < d/2, actually includes any primes */
	for(i = i_min; i <= i_max; i++)
	{
		for (m = 0; m < plan->n_S1; m++)
		{
			j = plan->S1[m];
			if((i * d >= j && primes[i * d - j]) || primes[i * d + j])
			{
#ifdef VERBOSE_PLAN
				printf("Final min_i = %u, found prime %u or %u\n",
					i, i*d - j, i*d + j);
#endif
				break;
			}
		}
		if(m < plan->n_S1)
			break;
	}
	i_min = i;


	/* For the remaining primes, write the required (i,j)-pairs to a list */
	for(i = i_min; i <= i_max; i++)
	{
		for(m = 0; m < plan->n_S1; m++)
		{
			j = plan->S1[m];
			/* See if this is a i*d +- j we need to include */
			if((i * d >= j && primes[i * d - j]) || primes[i * d + j])
			{
				while (need_NEXT_D)
				{
					plan->pairs[n++] = NEXT_D;
#ifdef VERBOSE_PLAN
					printf("Adding NEXT_D to list\n");
#endif
					need_NEXT_D--;
				}

#ifdef VERBOSE_PLAN
				printf("Adding %d*d +- %d (=S1[%d]) to list, includes primes ",
						i, j, m);
				if(i * d >= j && primes[i * d - j])
					printf("%d ", i*d - j);
				if(i * d + j <= B2 && primes[i * d + j])
					printf("%d", i*d + j);
				printf("\n");
#endif

				plan->pairs[n++] = (u_int8_t) m;
				n_pairs++;
				if(i * d >= j)
					primes[i * d - j] = 0;

				if(i * d + j <= B2)
					primes[i * d + j] = 0;
			}
		}

		need_NEXT_D++;
	}

	plan->pairs[n++] = NEXT_PASS;
#ifdef VERBOSE_PLAN
	printf("Adding NEXT_PASS to list\n");
#endif
	plan->i0 = i_min;
	plan->i1 = i_max + 1;

#ifdef VERBOSE_PLAN
	printf("pairs = ");
	for(i = 0; i < n; i++)
	{
		if(plan->pairs[i] == NEXT_D)
			printf("NEXT_D ");
		else if(plan->pairs[i] == NEXT_PASS)
			printf ("NEXT_PASS ");
		else
			printf ("%d ", plan->pairs[i]);
	}
	printf ("\n");

	printf("Used %u pairs to include %u primes, avg. %.2f primes/pair\n",
		n_pairs, n_primes, (double)n_primes / (double) n_pairs);

	for(i = B2_min + 1; i <= B2; i++)
	{
		if(primes[i])
		{
			fprintf(stderr, "Error, prime %d is still set\n", i);
			exit(-1);
		}
	}
#endif
	free(primes);
	return;
}

void stage2_plan_clear(stage2_plan_t *plan)
{
	if(plan->pairs)
		free(plan->pairs);
	if(plan->S1)
		free(plan->S1);
	memset(plan, 0, sizeof(*plan));
	return;
}

void pm1_plan_init(pm1_plan_t *plan, u_int32_t B1, u_int32_t B2)
{
	u_int64_t p, q, q_max;
	size_t tmp_E_n_words;
	mpz_t E, P;

	mpz_init(E);
	mpz_init(P);

	plan->B1 = B1;
	plan->exp2 = 0;

	for(p = 1; p <= B1 / 2; p <<= 1)
		plan->exp2++;

	mpz_set_ui(P, 2);
	mpz_set_ui(E, 1);

	for(mpz_nextprime(P, P); mpz_cmp_ui(P, B1) <= 0; mpz_nextprime(P, P))
	{
		p = mpz_get_ui(P);
		q_max = B1 / (p - 1);

		for(q = 1; q <= q_max; q *= p)
			mpz_mul_ui(E, E, p);
	}
#ifdef VERBOSE_PLAN
	gmp_printf("E: %Zd\n", E);
#endif
	plan->E = mpz_export(NULL, &tmp_E_n_words, -1, sizeof(u_int64_t), 0, 0, E);
	plan->E_n_words = (u_int32_t) tmp_E_n_words;

	plan->E_mask = 1ULL << 63;
	while((plan->E[plan->E_n_words - 1] & plan->E_mask) == 0)
		plan->E_mask >>= 1;

	stage2_plan_init(&plan->stage2, B1, B2);
	mpz_clear(E);
	mpz_clear(P);
	return;
}

void pm1_plan_clear(pm1_plan_t *plan)
{
	stage2_plan_clear(&plan->stage2);
	if(plan->E)
		free(plan->E);
	memset(plan, 0, sizeof(*plan));
	return;
}

#define PP1_DICT_NRENTRIES 6
static size_t pp1_dict_len[PP1_DICT_NRENTRIES] = {1, 1, 2, 2, 3, 4};
static literal_t *pp1_dict_entry[PP1_DICT_NRENTRIES] =
  {"\xB", "\xA", "\xB\xA", "\x3\x0", "\x3\xB\xA", "\x3\x0\x3\x0"};
static code_t pp1_dict_code[PP1_DICT_NRENTRIES] = {0, 0, 10, 11, 13, 14};
static bc_dict_t pp1_dict =
  {PP1_DICT_NRENTRIES, pp1_dict_len, pp1_dict_entry, pp1_dict_code};

void pp1_plan_init(pp1_plan_t *plan, u_int32_t B1, u_int32_t B2)
{
	u_int64_t p, q, q_max;
	mpz_t P;
	const double addcost = 10, doublecost = 10, bytecost = 1, changecost = 1;
	const unsigned int compress = 1;
	bc_state_t *bc_state;

	mpz_init(P);
	/* Make bytecode for stage 1 */
	plan->exp2 = 0;
	for(p = 1; p <= B1 / 2; p *= 2)
		plan->exp2++;

	plan->B1 = B1;
	bc_state = bytecoder_init(compress ? &pp1_dict : NULL);

	for(mpz_set_ui(P, 3); mpz_cmp_ui(P, B1) <= 0; mpz_nextprime(P, P))
	{
		p = mpz_get_ui(P);
		q_max = B1 / p;
		for(q = 1; q <= q_max; q *= p)
			prac_bytecode(p, addcost, doublecost, bytecost, changecost, bc_state);
	}

	bytecoder((literal_t) 12, bc_state);
	bytecoder_flush(bc_state);
	plan->bc_len = bytecoder_size(bc_state);
	plan->bc = (char *) malloc(plan->bc_len);
	bytecoder_read (plan->bc, bc_state);
	bytecoder_clear (bc_state);

	if(!compress)
	{
	  /* The very first chain init and very last chain end are hard-coded
	 in the stage 1 code and must be removed from the byte code. */
	  size_t i;
	  if(plan->bc[0] != 10) /* check that first code is chain init */
	  {
		  printf("pp1_plan_init: first code not chain init\n");
		  exit(-1);
	  }
	  if(plan->bc[plan->bc_len - 2] != 11) /* check that next-to-last code is chain end */
	  {
		  printf("pp1_plan_init: next to last code is not chain end\n");
		  exit(-1);
	  }
	  /* check that last code is bytecode end */
	  if(plan->bc[plan->bc_len - 1] != (literal_t) 12)
	  {
		  printf("pp1_plan_init: last code is not bytecode end\n");
		  exit(-1);
	  }
	  /* Remove first code 10 and last code 11 */
	  for(i = 1; i < plan->bc_len; i++)
		  plan->bc[i - 1] = plan->bc[i];
	  plan->bc[plan->bc_len - 3] = plan->bc[plan->bc_len - 2];
	  plan->bc_len -= 2;
	}

#ifdef VERBOSE_PLAN
	{
	  int changes = 0;
	  printf ("Byte code for stage 1: ");
	  for(p = 0; p < plan->bc_len; p++)
	  {
		  printf("%s%d", (p == 0) ? "" : ", ", (int) (plan->bc[p]));
		  changes += (p > 0 && plan->bc[p-1] != plan->bc[p]);
	  }
	  printf("\n");
	  printf("Length %d, %d code changes\n", plan->bc_len, changes);
	}
#endif

	/* Make stage 2 plan */
	stage2_plan_init(&plan->stage2, B1, B2);
	return;
}

void pp1_plan_clear(pp1_plan_t *plan)
{
	stage2_plan_clear(&plan->stage2);
	if(plan->bc)
		free(plan->bc);
	memset(plan, 0, sizeof(*plan));
	return;
}

#define ECM_DICT_NRENTRIES 4
static size_t ecm_dict_len[ECM_DICT_NRENTRIES] = {1, 1, 2, 2};
static literal_t *ecm_dict_entry[ECM_DICT_NRENTRIES] =
{"\xB", "\xA", "\xB\xA", "\x3\x0"};
static code_t ecm_dict_code[ECM_DICT_NRENTRIES] = {0, 0, 10, 11};

static bc_dict_t ecm_dict =
	{ECM_DICT_NRENTRIES, ecm_dict_len, ecm_dict_entry, ecm_dict_code};

void ecm_plan_init(ecm_plan_t *plan, u_int32_t B1, u_int32_t B2, int parameterization, u_int64_t sigma)
{
	unsigned int p, q;
	const double addcost = 6., doublecost = 5., /* TODO: find good ratio */
			bytecost = 1, changecost = 1;
	bc_state_t *bc_state;
	double totalcost = 0.;
	mpz_t P;

#ifdef VERBOSE_PLAN
    printf("Making plan for ECM with B1=%u, B2=%u, parameterization = %d, sigma=%lu\n",
            B1, B2, parameterization, sigma);
#endif

	/* If group order is divisible by 12, add two 2s to stage 1 */
	plan->exp2 = 2;
	for(q = 1; q <= B1 / 2; q *= 2)
		plan->exp2++;
	totalcost += plan->exp2 * doublecost;

	/* Make bytecode for stage 1 */
	plan->B1 = B1;
	plan->parameterization = parameterization;
	plan->sigma = sigma;
	bc_state = bytecoder_init(&ecm_dict);

	/* Group order is divisible by 12, add another 3 to stage 1 primes */
	totalcost += prac_bytecode(3, addcost, doublecost, bytecost,
			changecost, bc_state);

	mpz_init(P);
	for(mpz_set_ui(P, 3); mpz_cmp_ui(P, B1) <= 0; mpz_nextprime(P, P))
	{
		p = mpz_get_ui(P);
		for(q = 1; q <= B1 / p; q *= p)
			totalcost += prac_bytecode(p, addcost, doublecost, bytecost,
					changecost, bc_state);
	}
	mpz_clear(P);

	bytecoder((literal_t) 12, bc_state);
	bytecoder_flush(bc_state);
	plan->bc_len = bytecoder_size(bc_state);
	plan->bc = (u_int8_t *) malloc(plan->bc_len);
	bytecoder_read(plan->bc, bc_state);
	bytecoder_clear(bc_state);

#ifdef VERBOSE_PLAN
	{
		int changes = 0;
		printf("Byte code for stage 1: ");
		for(p = 0; p < plan->bc_len; p++)
		{
			printf("%s%d", (p == 0) ? "" : ", ", (int) (plan->bc[p]));
			changes += (p > 0 && plan->bc[p-1] != plan->bc[p]);
		}
		printf("\n");
		printf("Length %d, %d code changes, total cost: %f\n",
				plan->bc_len, changes, totalcost);
	}
#endif

	/* Make stage 2 plan */
	stage2_plan_init(&plan->stage2, B1, B2);
	return;
}

void ecm_plan_clear(ecm_plan_t *plan)
{
	stage2_plan_clear(&plan->stage2);
	if(plan->bc)
		free(plan->bc);
	memset(plan, 0, sizeof(*plan));
	return;
}
