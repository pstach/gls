/*
 * las.cl
 *
 *  Created on: January 28, 2014
 *      Author: tcarstens
 */
#include "include/ocl/ul/ul32.hl"
#include "las/ocl/kernels.hl"

#include "include/ocl/ul/ul64.hl"
#include "las/ocl/kernels.hl"

#include "include/ocl/ul/ul96.hl"
#include "las/ocl/kernels.hl"

#include "include/ocl/ul/ul128.hl"
#include "las/ocl/kernels.hl"

#include "include/ocl/ul/ul160.hl"
#include "las/ocl/kernels.hl"

#include "include/ocl/ul/ul192.hl"
#include "las/ocl/kernels.hl"

#include "include/ocl/ul/ul224.hl"
#include "las/ocl/kernels.hl"

#include "include/ocl/ul/ul256.hl"
#include "las/ocl/kernels.hl"


typedef struct lat_s {
	int64_t a0;
	int64_t b0;
	int64_t a1;
	int64_t b1;
} lat_t;


uint32_t root_in_qlattice31(uint32_t root, uint32_t p, __global lat_t *lat)
{
	int64_t a0, b0, a1, b1;
	int64_t tmp1, tmp2;
	uint32_t numer, denom, inv;

	a0 = lat->a0;
	b0 = lat->b0;
	a1 = lat->a1;
	b1 = lat->b1;

	tmp1 = ((int64_t) b1 * (int64_t) root) - a1;
	tmp1 %= p;
	if(tmp1 < 0)
		tmp1 += p;
	numer = tmp1;
	tmp2 = ((int64_t) b0 * (int64_t) root) - a0;
	tmp2 %= p;
	if(tmp2 < 0)
		tmp2 += p;
	denom = tmp2;
	if(denom == 0) /* projective */
		return 0;

    ul32 ul_p;
    ul32 ul_denom;
    ul32 ul_inv;
    ul32_set_ui(ul_p, p);
    ul32_set_ui(ul_denom, denom);
    ul32_modinv(ul_inv, ul_denom, ul_p);
    inv = ul32_get_ui(ul_inv);
    
	// inv = modinv32(denom, p);
	tmp1 = (int64_t) inv * (int64_t) numer;
	tmp1 %= p;
	return (p - tmp1);
}


#define RPA do {							\
    a0 += b0; a1 += b1;							\
    if ((a0 + b0 * 4 > 0)) {					\
      int32_t c0 = a0, c1 = a1;						\
      c0 += b0; c1 += b1; if ((c0 <= 0)) { a0 = c0; a1 = c1; }	\
      c0 += b0; c1 += b1; if ((c0 <= 0)) { a0 = c0; a1 = c1; }	\
      c0 += b0; c1 += b1; if ((c0 <= 0)) { a0 = c0; a1 = c1; }	\
    } else								\
      RPC;								\
  } while (0)
#define RPB do {							\
    b0 += a0; b1 += a1;							\
    if ((b0 + a0 * 4 < 0)) {					\
      int32_t c0 = b0, c1 = b1;						\
      c0 += a0; c1 += a1; if ((c0 >= 0)) { b0 = c0; b1 = c1; }	\
      c0 += a0; c1 += a1; if ((c0 >= 0)) { b0 = c0; b1 = c1; }	\
      c0 += a0; c1 += a1; if ((c0 >= 0)) { b0 = c0; b1 = c1; }	\
    } else								\
      RPD;								\
    } while (0)
#define RPC do {					\
    int32_t k = a0 / b0; a0 %= b0; a1 -= k * b1;	\
  } while (0)
#define RPD do {					\
    int32_t k = b0 / a0; b0 %= a0; b1 -= k * a1;	\
  } while (0)

int reduce_lattice31(__global lat_t *dst, uint32_t r, uint32_t p, uint32_t I)
{
  const int32_t hI = (int32_t) I;
  int32_t a0, b0, a1, b1, k;

  a0 = -(int32_t) p;
  b0 = (int32_t) r;
  a1 = 0;
  b1 = 1;

  if ((b0 >= hI)) {
    const int32_t mhI = -hI;
    RPC;
    while ((a0 < -0X7FFFFFFF / 5)) {
      RPD;
      if ((b0 < 0X7FFFFFFF / 5)) goto p15;
      RPC;
    }
    if ((a0 <= mhI))
      do {
	RPB;
      p15:
	if ((b0 < hI)) break;
	RPA;
      } while ((a0 <= mhI));
  }

  k = b0 - hI - a0;
  if (b0 > -a0) {
    if ((!a0)) return -1;
    k /= a0; b0 -= k * a0; b1 -= k * a1;
  } else {
    if ((!b0)) return -1;
    k /= b0; a0 += k * b0; a1 += k * b1;
  }
  dst->a0 = (int32_t) a0; dst->a1 = (uint32_t) b0; dst->b0 = (int32_t) a1; dst->b1 = (uint32_t) b1;
  return 0;
}


__kernel void kern_root_in_qlattice31(__global uint32_t *root,
                                      __global uint32_t *p,
                                      __global lat_t *lat,
                                      __global uint32_t *out,
                                      int n_batch) {
    int my_gid = get_global_id(0);
    if (my_gid >= n_batch)
        return;
    
    out[my_gid] = root_in_qlattice31(root[my_gid], p[my_gid], lat);
}


__kernel void kern_root_in_qlattice31_reduce_lattice31(
                                    __global uint32_t *root,
                                    __global uint32_t *p,
                                    __global lat_t *lat,
                                    
                                    __global lat_t *ef_lat,
                                    __global int *out_red,
                                    unsigned int I,
                                    int n_batch) {
    int my_gid = get_global_id(0);
    uint32_t lat_r;
    if (my_gid >= n_batch)
        return;
    
    lat_r = root_in_qlattice31(root[my_gid], p[my_gid], lat);
    if (lat_r) {
        out_red[my_gid] = reduce_lattice31(&ef_lat[my_gid], lat_r, p[my_gid], I);
    }
    else {
        out_red[my_gid] = -1;
    }
}
