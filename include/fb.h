/*
 * fb.h
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */

#ifndef FB_H_
#define FB_H_

#define ALIGN_LEN 4096ULL
#define MAX_SMALL_PRIME	(1ULL << 31)

typedef struct fb_s {
	u_int64_t n_small; /* number of fb elements less than MAX_SMALL_PRIME, on disk */
	u_int64_t n_large; /* number of fb elements less than MAX_SMALL_PRIME, on disk */
	u_int32_t *r_small;
	u_int32_t *p_small;
	u_int32_t *lr_small;
	u_int32_t *pos_small;
	u_int64_t *r_large;
	u_int64_t *p_large;
	u_int64_t *lr_large;
	u_int64_t n_line_sieve;
	void *map_ptr;
	size_t map_len;
	FILE *fd;
} fb_t[1];

extern void fb_init(fb_t fb);
extern void fb_clear(fb_t fb);
extern int fb_fileread(fb_t fb, const char *fname);
extern int fb_filesave(fb_t fb, const char *fname);

#endif /* FB_H_ */
