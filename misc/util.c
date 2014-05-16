/*
 * util.c
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */
#include <stdio.h>
#include <stdlib.h>
#include <wordexp.h>
#include "util.h"

FILE *file_open(const char *fname, const char *mode)
{
	FILE *fd;
	wordexp_t p;

	fd = NULL;
	wordexp(fname, &p, 0);
	if(p.we_wordc < 1)
		goto error;
	fd = fopen(p.we_wordv[0], mode);
error:
	wordfree(&p);
	return fd;
}


