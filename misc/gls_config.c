/*
 * gls_config.c
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"
#include "mpzpoly.h"
#include "gls_config.h"

int gls_config_init(gls_config_t cfg)
{
	size_t i;

	memset(cfg, 0, sizeof(cfg[0]));
	mpz_init(cfg->N);
	mpz_init(cfg->m);
	for(i = 0; i < sizeof(cfg->poly) / sizeof(cfg->poly[0]); i++)
		mpzpoly_init(cfg->poly[i]);
	return 0;
}

void gls_config_free(gls_config_t cfg)
{
	size_t i;

	mpz_clear(cfg->N);
	mpz_clear(cfg->m);
	for(i = 0; i < sizeof(cfg->poly) / sizeof(cfg->poly[0]); i++)
		mpzpoly_clear(cfg->poly[i]);
	return;
}

int polyfile_read(gls_config_t cfg, const char *fname)
{
	int ret;
	FILE *fd;
	char buf[8192], *ptr, last_char;
	unsigned long idx;
	char *name, *value;
	size_t i, line_num;

	ret = -1;
	fd = file_open(fname, "r");
	if(!fd)
	{
		fprintf(stderr, "failed to open polynomial file\n");
		return -1;
	}

	line_num = 0;
	while(!feof(fd))
	{
		memset(buf, 0, sizeof(buf));
		if(fgets(buf, sizeof(buf) - 1, fd) == NULL)
			break;
		line_num++;

		/* trim trailing carriage returns and newlines */
		while(strlen(buf) > 0 && strchr("\r\n", buf[strlen(buf) - 1]))
			buf[strlen(buf) - 1] = 0;
		/* skip whitespace */
		for(ptr = buf; *ptr && strchr(" \t", *ptr); ptr++);
		/* skip blank lines or comments */
		if(!*ptr || *ptr == '#')
			continue;
		name = ptr;
		/* skip to next whitespace or ':' */
		for(; *ptr && strchr(" \t:", *ptr) == NULL; ptr++);
		if(!*ptr)
		{
			fprintf(stderr, "malformed line #%d in polynomial file\n", line_num);
			goto cleanup;
		}
		/* find value */
		last_char = *ptr;
		*(ptr++) = 0;
		/* if we're not at ':', skip to it */
		if(last_char != ':')
			for(; *ptr && *ptr != ':'; ptr++);
		if(*ptr) ptr++;

		/* skip whitespace to value */
		for(; *ptr && strchr(" \t", *ptr); ptr++);
		if(!*ptr)
		{
			fprintf(stderr, "malformed line #%d in polynomial file\n", line_num);
			goto cleanup;
		}
		value = ptr;

		/* lowercase name */
		for(i = 0; i < strlen(name); i++)
			name[i] = tolower(name[i]);

		/* handle N */
		if(strcmp(name, "n") == 0)
		{
			mpz_set_str(cfg->N, value, 0);
			continue;
		}

		/* handle m */
		if(strcmp(name, "m") == 0)
		{
			mpz_set_str(cfg->m, value, 0);
			continue;
		}

		/* handle skew */
		if(strcmp(name, "skew") == 0 || strcmp(name, "skewness") == 0)
		{
			cfg->skew = strtof(value, NULL);
			continue;
		}

		/* handle coefficients of algebraic polynomial */
		if(name[0] == 'c')
		{
			idx = strtoul(&name[1], NULL, 0);
			if(idx >= MAX_POLY_DEGREE)
			{
				fprintf(stderr, "coefficient index %d on line #%d of polynomial file too large\n", idx, line_num);
				goto cleanup;
			}
			mpz_set_str(cfg->poly[APOLY_IDX]->c[idx], value, 0);
			continue;
		}

		/* handle coefficients of algebraic polynomial */
		if(name[0] == 'y')
		{
			idx = strtoul(&name[1], NULL, 0);
			if(idx >= MAX_POLY_DEGREE)
			{
				fprintf(stderr, "coefficient index %d on line #%d of polynomial file too large\n", idx, line_num);
				goto cleanup;
			}
			mpz_set_str(cfg->poly[RPOLY_IDX]->c[idx], value, 0);
			continue;
		}

		/* handle limits */
		if(strcmp(&name[1], "lim") == 0)
		{
			if(name[0] == 'r')
				idx = RPOLY_IDX;
			else if(name[0] == 'a')
				idx = APOLY_IDX;
			else
				goto line_invalid;
			cfg->lim[idx] = strtof(value, NULL);
			continue;
		}

		/* handle large prime bounds */
		if(strncmp(name, "lpb", strlen("lpb")) == 0 && strlen(name) == strlen("lpb") + 1)
		{
			if(name[strlen("lpb")] == 'r')
				idx = RPOLY_IDX;
			else if(name[strlen("lpb")] == 'a')
				idx = APOLY_IDX;
			else
				goto line_invalid;
			cfg->lpb[idx] = strtoul(value, NULL, 0);
			continue;
		}

		/* handle cofactorization bounds */
		if(strncmp(name, "mfb", strlen("mfb")) == 0 && strlen(name) == strlen("mfb") + 1)
		{
			if(name[strlen("mfb")] == 'r')
				idx = RPOLY_IDX;
			else if(name[strlen("mfb")] == 'a')
				idx = APOLY_IDX;
			else
				goto line_invalid;
			cfg->mfb[idx] = strtoul(value, NULL, 0);
			continue;
		}

		/* handle lambda */
		if(strcmp(&name[1], "lambda") == 0)
		{
			if(name[0] == 'r')
				idx = RPOLY_IDX;
			else if(name[0] == 'a')
				idx = APOLY_IDX;
			else
				goto line_invalid;
			cfg->lambda[idx] = strtof(value, NULL);
			continue;
		}

line_invalid:
		fprintf(stderr, "invalid line #%d in polynomial file\n", line_num);
		goto cleanup;
	}

	for(idx = 0; idx < sizeof(cfg->poly) / sizeof(cfg->poly[0]); idx++)
	{
		cfg->poly[idx]->deg = MAX_POLY_DEGREE;
		mpzpoly_fix_degree(cfg->poly[idx]);
	}
	ret = 0;

cleanup:
	fclose(fd);
	return ret;
}


