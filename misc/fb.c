/*
 * fb.c
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "util.h"
#include "fb.h"

void fb_init(fb_t fb)
{
	memset(fb, 0, sizeof(fb[0]));
	return;
}

void fb_clear(fb_t fb)
{
	if(!fb->map_ptr)
	{
		if(fb->p_small)
			free(fb->p_small);
		if(fb->r_small)
			free(fb->r_small);
		if(fb->p_large)
			free(fb->p_large);
		if(fb->r_large)
			free(fb->r_large);
	}
	if(fb->map_ptr)
		munmap(fb->map_ptr, fb->map_len);
	if(fb->fd)
		fclose(fb->fd);
	memset(fb, 0, sizeof(fb[0]));
	return;
}

int fb_fileread(fb_t fb, const char *fname)
{
	int fd;
	u_int8_t *ptr, *end_ptr;
	off_t align;

	fb->fd = file_open(fname, "rb");
	if(!fb->fd)
	{
		perror("fb_fileread:fopen");
		return -1;
	}

	fseek(fb->fd, 0, SEEK_END);
	fb->map_len = ftell(fb->fd);
	fseek(fb->fd, 0, SEEK_SET);

	fb->map_ptr = mmap(NULL, fb->map_len, PROT_READ, MAP_SHARED | MAP_POPULATE, fileno(fb->fd), 0);
	if(fb->map_ptr == MAP_FAILED)
	{
		perror("fb_fileread:mmap");
		return -1;
	}

	ptr = (u_int8_t *) fb->map_ptr;
	end_ptr = ptr + fb->map_len;
	memcpy(&fb->n_small, ptr, sizeof(fb->n_small));
	memcpy(&fb->n_large, &ptr[sizeof(fb->n_small)], sizeof(fb->n_large));
	align = (sizeof(fb->n_small) + sizeof(fb->n_large) + ALIGN_LEN - 1) & ~(ALIGN_LEN - 1);

	fb->p_small = (u_int32_t *) &ptr[align];
	align = ((sizeof(fb->p_small[0]) * fb->n_small + ALIGN_LEN - 1) & ~(ALIGN_LEN - 1)) + align;
	fb->r_small = (u_int32_t *) &ptr[align];
	align = ((sizeof(fb->r_small[0]) * fb->n_small + ALIGN_LEN - 1) & ~(ALIGN_LEN - 1)) + align;

	fb->p_large = (u_int64_t *) &ptr[align];
	align = ((sizeof(fb->p_large[0]) * fb->n_large + ALIGN_LEN - 1) & ~(ALIGN_LEN - 1)) + align;
	fb->r_large = (u_int64_t *) &ptr[align];
	align = ((sizeof(fb->p_large[0]) * fb->n_large + ALIGN_LEN - 1) & ~(ALIGN_LEN - 1)) + align;

	if(&ptr[align] > end_ptr)
	{
		fprintf(stderr, "fb_fileread: factor base appears truncated\n");
		return -1;
	}
/*
	if(fb->n_small)
	{
		fb->lr_small = (u_int32_t *) mmap(NULL, (sizeof(fb->r_small[0]) * fb->n_small + ALIGN_LEN - 1) & ~(ALIGN_LEN - 1),
				PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	}
	if(fb->n_large)
	{
		fb->lr_large = (u_int64_t *) mmap(NULL, (sizeof(fb->r_large[0]) * fb->n_large + ALIGN_LEN - 1) & ~(ALIGN_LEN - 1),
				PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	}
*/
	return 0;
}

int fb_filesave(fb_t fb, const char *fname)
{
	int fd, ret;
	off_t align;

	ret = -1;
	fd = open(fname, O_CREAT | O_WRONLY, 00644);
	if(fd < 0)
	{
		perror("fb_filesave:open");
		return -1;
	}

	if(write(fd, &fb->n_small, sizeof(fb->n_small)) != sizeof(fb->n_small))
	{
		perror("fb_filesave:write");
		goto cleanup;
	}
	if(write(fd, &fb->n_large, sizeof(fb->n_large)) != sizeof(fb->n_large))
	{
		perror("fb_filesave:write");
		goto cleanup;
	}

	align = (sizeof(fb->n_small) + sizeof(fb->n_large) + ALIGN_LEN - 1) & ~(ALIGN_LEN - 1);
	if(lseek(fd, align, SEEK_SET) != align)
	{
		perror("fb_filesave:lseek");
		goto cleanup;
	}

	if(write(fd, fb->p_small, sizeof(fb->p_small[0]) * fb->n_small) != sizeof(fb->p_small[0]) * fb->n_small)
	{
		perror("fb_filesave:write");
		goto cleanup;
	}

	align = ((sizeof(fb->p_small[0]) * fb->n_small + ALIGN_LEN - 1) & ~(ALIGN_LEN - 1)) + align;
	if(lseek(fd, align, SEEK_SET) != align)
	{
		perror("fb_filesave:lseek");
		goto cleanup;
	}

	if(write(fd, fb->r_small, sizeof(fb->r_small[0]) * fb->n_small) != sizeof(fb->r_small[0]) * fb->n_small)
	{
		perror("fb_filesave:write");
		goto cleanup;
	}

	align = ((sizeof(fb->r_small[0]) * fb->n_small + ALIGN_LEN - 1) & ~(ALIGN_LEN - 1)) + align;
	if(lseek(fd, align, SEEK_SET) != align)
	{
		perror("fb_filesave:lseek");
		goto cleanup;
	}

	if(write(fd, fb->p_large, sizeof(fb->p_large[0]) * fb->n_large) != sizeof(fb->p_large[0]) * fb->n_large)
	{
		perror("fb_filesave:write");
		goto cleanup;
	}

	align = ((sizeof(fb->p_large[0]) * fb->n_large + ALIGN_LEN - 1) & ~(ALIGN_LEN - 1)) + align;
	if(lseek(fd, align, SEEK_SET) != align)
	{
		perror("fb_filesave:lseek");
		goto cleanup;
	}

	if(write(fd, fb->r_large, sizeof(fb->r_large[0]) * fb->n_large) != sizeof(fb->r_large[0]) * fb->n_large)
	{
		perror("fb_filesave:write");
		goto cleanup;
	}

	align = ((sizeof(fb->p_large[0]) * fb->n_large + ALIGN_LEN - 1) & ~(ALIGN_LEN - 1)) + align;
	if(lseek(fd, align, SEEK_SET) != align)
	{
		perror("fb_filesave:lseek");
		goto cleanup;
	}
	write(fd, "\x00", 1);
	ftruncate(fd, align);
	ret = 0;
cleanup:
	close(fd);
	return ret;
}


