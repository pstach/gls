/*
 * qgen.h
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */

#ifndef QGEN_H_
#define QGEN_H_

typedef struct qgen_s {
	int type;
	mpz_t qpos;
	mpz_t q0;
	mpz_t q1;
	FILE *fd;
} qgen_t;

#define QGEN_RANGE	1
#define QGEN_LIST	2

static inline qgen_t *qgen_list(const char *fname)
{
	qgen_t *ret;

	ret = (qgen_t *) malloc(sizeof(qgen_t));
	memset(ret, 0, sizeof(*ret));
	ret->type = QGEN_LIST;
	mpz_init(ret->qpos);
	mpz_init(ret->q0);
	mpz_init(ret->q1);
	ret->fd = fopen(fname, "r");
	if(!ret->fd)
	{
		perror("qgen_list:fopen");
		free(ret);
		return NULL;
	}
	return ret;
}

static inline qgen_t *qgen_range(mpz_t q0, mpz_t q1)
{
	qgen_t *ret;

	ret = (qgen_t *) malloc(sizeof(qgen_t));
	memset(ret, 0, sizeof(*ret));
	ret->type = QGEN_RANGE;
	mpz_init(ret->q0);
	mpz_init(ret->q1);
	mpz_init(ret->qpos);

	mpz_set(ret->q0, q0);
	mpz_set(ret->q1, q1);
	mpz_set(ret->qpos, q0);
	return ret;
}

static inline void qgen_clear(qgen_t *qgen)
{
	if(qgen->fd)
		fclose(qgen->fd);
	mpz_clear(qgen->q0);
	mpz_clear(qgen->q1);
	mpz_clear(qgen->qpos);
	return;
}

static inline int qgen_peek_q(qgen_t *qgen, mpz_t q)
{
	if(qgen->type == QGEN_RANGE)
	{
		mpz_nextprime(qgen->qpos, qgen->qpos);
		mpz_set(q, qgen->qpos);
		if(mpz_cmp(q, qgen->q1) < 0)
			return 1;
		mpz_sub_ui(qgen->qpos, qgen->qpos, 1);
		return 0;
	}
	if(qgen->type == QGEN_LIST)
	{
		long off;
		char buf[1024];

		off = ftell(qgen->fd);
		memset(buf, 0, sizeof(buf));
		if(fgets(buf, sizeof(buf) - 1, qgen->fd) == NULL)
			return 0;
		mpz_set_str(q, buf, 10);
		fseek(qgen->fd, off, SEEK_SET);
		return 1;
	}
	return 0;
}

static inline int qgen_next_q(qgen_t *qgen, mpz_t q)
{
	if(qgen->type == QGEN_RANGE)
	{
		mpz_nextprime(qgen->qpos, qgen->qpos);
		mpz_set(q, qgen->qpos);
		if(mpz_cmp(q, qgen->q1) < 0)
			return 1;
		return 0;
	}
	if(qgen->type == QGEN_LIST)
	{
		char buf[1024];

		memset(buf, 0, sizeof(buf));
		if(fgets(buf, sizeof(buf) - 1, qgen->fd) == NULL)
			return 0;
		mpz_set_str(q, buf, 10);
		return 1;
	}
	return 0;
}

#endif /* QGEN_H_ */
