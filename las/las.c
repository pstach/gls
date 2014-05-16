/*
 * las.c
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
#include "ocl_cofact.h"


#define BAD_ANGLE 8.

#define LOG_SCALE 1.4426950408889634

#define abs_int(x) (((x) < 0) ? -(x) : (x))

#define LAT_MAX ((double) (1ULL << 32))
#define LAT_MIN (-LAT_MAX)

FILE *output;

double ltrans_time, vtrans_time, lsieve_time, vsieve_time, scheck_time, lresieve_time, vresieve_time;
double ltrans_total, vtrans_total, lsieve_total, vsieve_total, scheck_total, lresieve_total, vresieve_total;
double pm1_total, pp1_total, ecm_total;
u_int64_t rel_count;

static inline u_int8_t fb_log(double n, double log_scale)
{
  const long l = floor(log(n) * log_scale + 0.5);
  return (u_int8_t) l;
}

u_int8_t fb_make_steps(u_int64_t *steps, const u_int32_t fbb, const double scale)
{
	unsigned char i;
	const double base = exp(1. / scale);

	//fprintf(stdout, "fbb = %lu, scale = %f, base = %f\n", (unsigned long) fbb, scale, base);
	const u_int8_t max = fb_log(fbb, scale);

	for(i = 0; i <= max; i++)
	{
		steps[i] = ceil(pow(base, i + 0.5)) - 1.;
		//fprintf(stdout, "steps[%u] = %lu\n", (unsigned int) i, (long unsigned) steps[i]);
		/* fb_log(n, scale) = floor (log (n) * scale + 0.5)
		 * = floor (log (floor(pow(base, i + 0.5))) * scale + 0.5)
		 * = floor (log (ceil(pow(e^(1. / scale), i + 0.5)-1)) * scale + 0.5)
		 * < floor (log (pow(e^(1. / scale), i + 0.5)) * scale + 0.5)
		 * = floor (log (e^((i+0.5) / scale)) * scale + 0.5)
		 * = floor (((i+0.5) / scale) * scale + 0.5)
		 * = floor (i + 1)
		 * Thus, fb_log(n, scale) < floor (i + 1) <= i
		 */
		/* We have to use <= in the first assert, as for very small i, multiple
		 * steps[i] can have the same value
		 */
		assert(fb_log(steps[i], scale) <= i);
		assert(fb_log(steps[i] + 1, scale) > i);
	}
	return max;
}

static inline u_int8_t find_logp(const u_int64_t *log_steps, const u_int8_t log_steps_max, const u_int32_t p)
{
	u_int8_t i;
	for (i = log_steps_max; i > 1; i--)
	{
		if(log_steps[i - 1] < p)
			return i;
	}
	fprintf(stderr, "invalid logp\n");
	exit(-1);
	return 0;
}

#if 0
int reduce_lattice_skew(lat_t dst, u_int64_t r, u_int64_t p, double sigma)
{
	double a0sq, a1sq, s, k;
	double a0, b0, a1, b1;

	a0 = (double) p;
	b0 = 0.;
	a1 = (double) r;
	b1 = 1.;

	a0sq = a0 * a0;
	a1sq = (a1 * a1) + sigma;

	for(;;)
	{
		s = a0 * a1;
		s += (sigma * b0 * b1);

		if(a0sq < a1sq)
		{
			//assert(((s / a0sq) >= LAT_MIN) && ((s / a0sq) <= LAT_MAX));
			k = rint(s / a0sq);
			if(k == 0.)
				break;
			a1 -= (k * a0);
			b1 -= (k * b0);
			a1sq = a1 * a1;
			a1sq += (sigma * b1 * b1);
		}
		else
		{
			//assert(((s / a1sq) >= LAT_MIN) && ((s / a1sq) <= LAT_MAX));
			k = rint(s / a1sq);
			if(k == 0.)
				break;
			a0 -= (k * a1);
			b0 -= (k * b1);
			a0sq = a0 * a0;
			a0sq += (sigma * b0 * b0);
		}
	}

	if(b0 < 0.)
	{
		b0 = -b0;
		a0 = -a0;
	}
	if(b1 < 0.)
	{
		b1 = -b1;
		a1 = -a1;
	}

	if(a0sq < a1sq)
	{
		s = a1;
		k = b1;
		a1 = a0;
		b1 = b0;
		a0 = s;
		b0 = k;
	}

	if(fabs(a0) >= LAT_MAX || fabs(a1) >= LAT_MAX)
		return -1;

	dst->a0 = (int64_t) a0;
	dst->b0 = (int64_t) b0;
	dst->a1 = (int64_t) a1;
	dst->b1 = (int64_t) b1;
	return 0;
}
#endif

/* check that the double x fits into an int32_t */
#define fits_int32_t(x) \
  ((double) INT32_MIN <= (x)) && ((x) <= (double) INT32_MAX)

int mpz_rdiv_q(mpz_ptr q, mpz_t a, mpz_t b)
{
	mpz_t r;
	mpz_init(r);

	mpz_fdiv_qr(q,r,a,b);
	/* b>0: We want -b/2 <= a-bq < b/2 */
	/* b<0: We want  b/2 < a-bq <= -b/2 */
	mpz_mul_2exp(r,r,1);
	if(mpz_cmp(r,b) * mpz_sgn(b) >= 0)
		mpz_add_ui(q, q, 1);
	mpz_clear(r);
	return 1;
}

/* We work with two vectors v0=(a0,b0) and v1=(a1,b1). The quadratic form
 * is proportional to a0^2+skewness^2*b0^2 */
static int generic_skew_gauss(mpz_t a[2], mpz_t b[2], double sigma)
{
	// sigma, or skewness^2, can be written as a one-word (actually, 53 bits)
	// integer times a power of two, which is presumably larger than
	// 2^-53 in the most common case, since we expect skewness > 1.
	double mantissa;
	int64_t mantissa_z;
	int exponent;
	/* These are provided by c99 */
	mantissa = frexp(sigma, &exponent);
	mantissa_z = ldexp(mantissa, 53);
	exponent -= 53;

	mpz_t N[2], S, q, tmp, tmp2;

	mpz_init(N[0]);
	mpz_init(N[1]);
	mpz_init(S);
	mpz_init(q);
	mpz_init(tmp);
	mpz_init(tmp2);

    /* Compute the two norms, and the dot products */
#define QUADFORM(dst, i0, i1) do { \
	mpz_mul(dst, a[i0], a[i1]); \
	mpz_mul(tmp, b[i0], b[i1]); \
	if (exponent < 0) mpz_mul_2exp(dst, dst, -exponent); \
	if (exponent > 0) mpz_mul_2exp(tmp, tmp, exponent); \
	mpz_set_si(tmp2, mantissa_z); \
	mpz_addmul(dst, tmp, tmp2); \
} while (0)

	QUADFORM(N[0], 0, 0);
	QUADFORM(N[1], 1, 1);
	QUADFORM(S,    0, 1);

	/* After a reduction step (e.g. v0-=q*v1), N[0], N[1], and S are
	 * updated using the following algorithm.
	 * new_N0 = old_N0 + q^2old_N1 - 2q old_S
	 * new_S = old_S - q old_N1
	 *
	 * i.e.
	 *
	 * new_N0 = old_N0 - q old_S - q*(old_S-q old_N1)
	 * new_N0 = old_N0 - q * (old_S + new_S)
	 *
	 * cost: two products only
 	 */
    for(;;)
    {
        /* reduce v0 with respect to v1 */
        mpz_rdiv_q(q, S, N[1]);
        if(mpz_cmp_ui(q, 0) == 0) break;
        mpz_submul(a[0], q, a[1]);
        mpz_submul(b[0], q, b[1]);
        /* update */
        mpz_set(tmp, S);
        mpz_submul(S, q, N[1]);
        mpz_add(tmp, tmp, S);
        mpz_submul(N[0], q, tmp);

        /* reduce v1 with respect to v0 */
        mpz_rdiv_q(q, S, N[0]);
        if(mpz_cmp_ui(q, 0) == 0) break;
        mpz_submul(a[1], q, a[0]);
        mpz_submul(b[1], q, b[0]);
        /* update */
        mpz_set(tmp, S);
        mpz_submul(S, q, N[0]);
        mpz_add(tmp, tmp, S);
        mpz_submul(N[1], q, tmp);
    }

    /* We don't care about the sign of b. Down the road, there's an
     * IJToAB function which guarantees positive b */

    /* However we do care about vector 0 being the ``smallest'' in some
     * sense. The trick is that the comparison criterion used previously
     * by the code was the skewed L^\infty norm max(a,b*skewness). We
     * want the reduction step here to be oblivious to this L^\infty /
     * L^2 mess, so we provide something which is L^2 minimal. Wrapper
     * code around this function is still guaranteeing L^\infinity
     * minimality for compatibility, though (see sieve_info_adjust_IJ)
     */
    if(mpz_cmp(N[0], N[1]) > 0)
    {
        mpz_swap(a[0], a[1]);
        mpz_swap(b[0], b[1]);
    }
    mpz_clear(N[0]);
    mpz_clear(N[1]);
    mpz_clear(S);
    mpz_clear(q);
    mpz_clear(tmp);
    mpz_clear(tmp2);
    return 1;
}

#define mpz_fits_int64_p(x) (mpz_sizeinbase((x), 2) < 64)

int reduce_lattice_skew(lat_t dst, u_int64_t r, u_int64_t p, double sigma)
{
    mpz_t a[2], b[2];
    int fits;

    mpz_init_set_ui(a[0], p);
    mpz_init_set_ui(a[1], r);
    mpz_init_set_ui(b[0], 0);
    mpz_init_set_ui(b[1], 1);
    generic_skew_gauss(a, b, sigma);
    fits = mpz_fits_int64_p(a[0]);
    fits = fits & mpz_fits_int64_p(b[0]);
    fits = fits & mpz_fits_int64_p(a[1]);
    fits = fits & mpz_fits_int64_p(b[1]);
    if(fits)
    {
    	dst->a0 = mpz_get_si(a[0]);
        dst->a1 = mpz_get_si(a[1]);
        dst->b0 = mpz_get_si(b[0]);
        dst->b1 = mpz_get_si(b[1]);
    }
    mpz_clear(a[0]);
    mpz_clear(a[1]);
    mpz_clear(b[0]);
    mpz_clear(b[1]);
    return (fits - 1);
}

static int reject_lattice(lat_t lat)
{
	double ang;
	double a1, b1;

	a1 = lat->a1;
	b1 = lat->b1;
	if(fabs(a1) < 1.)
		a1 = 0.000001;
	if(fabs(b1) < 1.)
		b1 = 0.000001;
	ang = fabs(180.0 - (atan(-lat->a0 / a1) - atan(-lat->b0 / b1)) * 180.0 / 3.14159);
	if(ang < BAD_ANGLE)
		return -1;
	return 0;
}

u_int32_t modinv32(u_int32_t a, u_int32_t p)
{
    u_int32_t ps1, ps2, dividend, divisor, rem, q, t;
    u_int32_t parity;

    q = 1;
    rem = a;
    dividend = p;
    divisor = a;
    ps1 = 1;
    ps2 = 0;
    parity = 0;

    while(divisor > 1)
    {
    	rem = dividend - divisor;
    	t = rem - divisor;
		if(rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if(rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if(rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if(rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if(rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if(rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if(rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if(rem >= divisor) { q += ps1; rem = t;
		if(rem >= divisor) {
			q = dividend / divisor;
			rem = dividend % divisor;
			q = q * ps1;
		} } } } } } } } }

		q += ps2;
		parity = ~parity;
		dividend = divisor;
		divisor = rem;
		ps2 = ps1;
		ps1 = q;
    }

    if(parity == 0)
    	return ps1;
    return (p - ps1);
}

static inline u_int64_t gcd64(u_int64_t x, u_int64_t y)
{
	u_int64_t tmp;

	if(y < x)
	{
		tmp = x;
		x = y;
		y = tmp;
	}

	while(y > 0)
	{
		x = x % y;
		tmp = x;
		x = y;
		y = tmp;
	}
	return x;
}

u_int32_t root_in_qlattice31(u_int32_t root, u_int32_t p, lat_t lat)
{
	int64_t a0, b0, a1, b1;
	int64_t tmp1, tmp2;
	u_int32_t numer, denom, inv;

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

	inv = modinv32(denom, p);
	tmp1 = (int64_t) inv * (int64_t) numer;
	tmp1 %= p;
	return (p - tmp1);
}

u_int64_t root_in_qlattice63(u_int64_t root, u_int64_t p, lat_t lat)
{
	int64_t a0, b0, a1, b1;
	int64_t tmp1, tmp2;
	u_int32_t numer, denom, inv;

	a0 = lat->a0;
	b0 = lat->b0;
	a1 = lat->a1;
	b1 = lat->b1;

	tmp1 = (b1 * (int64_t) root) - a1;
	tmp1 %= p;
	if(tmp1 < 0)
		tmp1 += p;
	numer = tmp1;

	tmp2 = (b0 * (int64_t) root) - a0;
	tmp2 %= p;
	if(tmp2 < 0)
		tmp2 += p;
	denom = tmp2;

	if(denom == 0)
		return 0;

	inv = modinv32(denom, p);
	assert(inv != 0);
	assert(inv * denom % p == 1);
	tmp1 = inv * numer;
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

int reduce_lattice31(lat_t dst, u_int32_t r, u_int32_t p, u_int32_t I)
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
  dst->a0 = (int32_t) a0; dst->a1 = (u_int32_t) b0; dst->b0 = (int32_t) a1; dst->b1 = (u_int32_t) b1;
  return 0;
}

#if DEAD_CODE
#define MAX_REDUCE_ITER 16

int reduce_lattice31(lat_t dst, u_int32_t r, u_int32_t p, int32_t I)
{
	double a0sq, a1sq, s, k;
	double a0, b0, a1, b1;
	int i;

	a0 = (double) p;
	b0 = 0.;
	a1 = (double) r;
	b1 = 1.;

	a0sq = a0 * a0;
	a1sq = a1 * a1;

	for(i = 0; i < MAX_REDUCE_ITER; i++)
	{
		s = (a0 * a1) + (b0 * b1);
		if(a0sq < a1sq)
		{
			//assert(((s / a0sq) >= LAT_MIN) && ((s / a0sq) <= LAT_MAX));
			k = rint(s / a0sq);
			if(k >= I_half)
				break;
			a1 -= (k * a0);
			b1 -= (k * b0);
			a1sq = (a1 * a1) + (b1 * b1);
		}
		else
		{
			//assert(((s / a1sq) >= LAT_MIN) && ((s / a1sq) <= LAT_MAX));
			k = rint(s / a1sq);
			if(k >= I_half)
				break;
			a0 -= (k * a1);
			b0 -= (k * b1);
			a0sq = (a0 * a0) + (b0 * b0);
		}
	}
	if(i >= MAX_REDUCE_ITER)
		return -1;

	if(b0 < 0.)
	{
		b0 = -b0;
		a0 = -a0;
	}
	if(b1 < 0.)
	{
		b1 = -b1;
		a1 = -a1;
	}

	if(a0sq < a1sq)
	{
		s = a1;
		k = b1;
		a1 = a0;
		b1 = b0;
		a0 = s;
		b0 = k;
	}

	if(fabs(a0) >= LAT_MAX || fabs(a1) >= LAT_MAX)
		return -1;

	dst->a0 = (int64_t) a0;
	dst->b0 = (int64_t) b0;
	dst->a1 = (int64_t) a1;
	dst->b1 = (int64_t) b1;
	return 0;
}
#endif

static const u_int32_t coprime30_tab[30] = {
        0x00000000, 0x2aaaaaaa, 0x24924924, 0x2aaaaaaa,
        0x21084210, 0x2ebaebae, 0x00000000, 0x2aaaaaaa,
        0x24924924, 0x2baaeaba, 0x00000000, 0x2ebaebae,
        0x00000000, 0x2aaaaaaa, 0x259a4b34, 0x2aaaaaaa,
        0x00000000, 0x2ebaebae, 0x00000000, 0x2baaeaba,
        0x24924924, 0x2aaaaaaa, 0x00000000, 0x2ebaebae,
        0x21084210, 0x2aaaaaaa, 0x24924924, 0x2aaaaaaa,
        0x00000000, 0x2fbaebbe
};

/* returns non-zero when gcd(a,b) = 2x, 3x, 5x */
int coprime30(u_int64_t a, u_int64_t b)
{
	u_int32_t a_idx, b_idx;

	a_idx = (a - 1) % 30;
	b_idx = (b - 1) % 30;
	return (coprime30_tab[b_idx] & (1 << a_idx));
}

static int bucket_cmp(const void *p1, const void *p2)
{
	bucket_t *b1, *b2;

	b1 = (bucket_t *) p1;
	b2 = (bucket_t *) p2;

	if(b1->r < b2->r)
		return -1;
	if(b1->r > b2->r)
		return 1;
	return 0;
}

/* S1 = S1 - S2, with "-" in saturated arithmetical,
 * and memset(S2, 0, EndS1-S1).
 */
void SminusS(u_int8_t *S1, u_int8_t *EndS1, u_int8_t *S2)
{
  __m128i *S1i = (__m128i *) S1, *EndS1i = (__m128i *) EndS1, *S2i = (__m128i *) S2,
    z = _mm_setzero_si128();
  while (S1i < EndS1i) {
    __m128i x0, x1, x2, x3;
    __asm__ __volatile__
      ("prefetcht0 0x1000(%0)\n"
       "prefetcht0 0x1000(%1)\n"
       "movdqa (%0),%2\n"
       "movdqa 0x10(%0),%3\n"
       "movdqa 0x20(%0),%4\n"
       "movdqa 0x30(%0),%5\n"
       "psubusb (%1),%2\n"
       "psubusb 0x10(%1),%3\n"
       "psubusb 0x20(%1),%4\n"
       "psubusb 0x30(%1),%5\n"
       "movdqa %6,(%1)\n"
       "movdqa %6,0x10(%1)\n"
       "movdqa %6,0x20(%1)\n"
       "movdqa %6,0x30(%1)\n"
       "movdqa %2,(%0)\n"
       "movdqa %3,0x10(%0)\n"
       "movdqa %4,0x20(%0)\n"
       "movdqa %5,0x30(%0)\n"
       "add $0x40,%0\n"
       "add $0x40,%1\n"
       : "+&r"(S1i), "+&r"(S2i), "=&x"(x0), "=&x"(x1), "=&x"(x2), "=&x"(x3) : "x"(z));
  }
}

void usage(const char *prog, const char *error)
{
	fprintf(stderr, "%s\n", error);
	exit(1);
}

#include "cofact_plan.h"

int main(int argc, char *argv[])
{
	int i, j;
#if USE_OPENCL
	cl_uint config_ocl_platform = 0;
	cl_uint config_ocl_device = 0;
	unsigned int config_ocl_buildopts = 0;
#endif
	u_int32_t I_dim, I, I_half, J, I_mask;
	u_int64_t IJ;
	gls_config_t cfg;
	char *qlist, *polyfile;
	fb_t fb[2];
	char fname[4096];
	mpz_t q0, q1, q, tmp;
	int side, q_side;
	double sigma;
	mpz_t roots[MAX_POLY_DEGREE];
	u_int32_t root_idx, n_roots;
	u_int64_t fb_idx;
	qgen_t *qgen;
	mpz_t cofact_mul[POLY_CNT];
	u_int8_t *bucket_map[POLY_CNT];
	bucket_t **buckets[POLY_CNT];
#ifdef BUCKET_DEBUG
	size_t *bucket_counts[POLY_CNT];
#endif
	size_t n_buckets, bucket_alloc;
	u_int8_t *S[POLY_CNT];
	u_int8_t *SS;
	double tin, tout;

	output = stdout;
	mpz_init_set_ui(q0, 0);
	mpz_init_set_ui(q1, 0);
	qlist = NULL;
	polyfile = NULL;
	q_side = APOLY_IDX;
	I_dim = 15;

	for(i = 1; i < argc; i++)
	{
		if(strcmp(argv[i], "-h") == 0)
			usage(argv[0], NULL);
#if USE_OPENCL
        if(strcmp(argv[i], "-oclp") == 0)
        {
            if (i + 1 >= argc)
                usage(argv[0], "-oclp missing argument");
            config_ocl_platform = atoi(argv[i+1]);
            i++;
            continue;
        }
        if(strcmp(argv[i], "-ocld") == 0)
        {
            if (i + 1 >= argc)
                usage(argv[0], "-ocld missing argument");
            config_ocl_device = atoi(argv[i+1]);
            i++;
            continue;
        }
        if(strcmp(argv[i], "-nvidia") == 0)
        {
            config_ocl_buildopts |= OCL_BUILDOPT_NVIDIA;
            i++;
            continue;
        }
#endif
		if(strcmp(argv[i], "-out") == 0)
		{
			if(i + 1 >= argc)
				usage(argv[0], "-out missing argument");
			output = fopen(argv[i + 1], "w");
			if(!output)
			{
				fprintf(stderr, "failed to open output: \"%s\"\n", output);
				return -1;
			}
			i++;
			continue;
		}
		if(strcmp(argv[i], "-I") == 0)
		{
			if(i + 1 >= argc)
				usage(argv[0], "-I missing argument");
			I_dim = strtoul(argv[i + 1], NULL, 10);
			i++;
			continue;
		}
		if(strcmp(argv[i], "-qr") == 0)
		{
			q_side = RPOLY_IDX;
			continue;
		}
		if(strcmp(argv[i], "-qa") == 0)
		{
			q_side = APOLY_IDX;
			continue;
		}
		if(strcmp(argv[i], "-q0") == 0)
		{
			if(i + 1 >= argc)
				usage(argv[0], "-q0 missing argument");
			if(mpz_cmp_ui(q0, 0) != 0)
				usage(argv[0], "multiple q0 values specified");
			mpz_set_str(q0, argv[i + 1], 10);
			i++;
			continue;
		}
		if(strcmp(argv[i], "-q1") == 0)
		{
			if(i + 1 >= argc)
				usage(argv[0], "-q1 missing argument");
			if(mpz_cmp_ui(q1, 0) != 0)
				usage(argv[0], "multiple q1 values specified");
			mpz_set_str(q1, argv[i + 1], 10);
			i++;
			continue;
		}
		if(strcmp(argv[i], "-qlist") == 0)
		{
			if(i + 1 >= argc)
				usage(argv[0], "-qlist missing argument");
			if(qlist)
				usage(argv[0], "multiple qlists specified");
			qlist = argv[i + 1];
			i++;
			continue;
		}
		if(strcmp(argv[i], "-poly") == 0)
		{
			if(i + 1 >= argc)
				usage(argv[0], "-poly missing argument");
			if(polyfile)
				usage(argv[0], "multiple poly specified");
			polyfile = argv[i + 1];
			i++;
			continue;
		}
		fprintf(stderr, "invalid argument: %s\n", argv[i]);
		usage(argv[0], "");
	}

	if(mpz_cmp_ui(q0, 0) != 0 && qlist)
		usage(argv[0], "q range and qlist specified");
	if(mpz_cmp_ui(q0, 0) == 0 && !qlist)
		usage(argv[0], "both q range and qlist not specified");

#if USE_OPENCL
    ocl_init(&ocl_state, config_ocl_platform, config_ocl_device, config_ocl_buildopts);
#endif

	gls_config_init(cfg);
	if(polyfile_read(cfg, polyfile) < 0)
		return 1;

	if(qlist)
		qgen = qgen_list(qlist);
	if(mpz_cmp_ui(q0, 0) != 0)
		qgen = qgen_range(q0, q1);
	if(!qgen)
		return 1;

	for(fb_idx = 0; fb_idx < sizeof(fb) / sizeof(fb[0]); fb_idx++)
	{
		memset(fname, 0, sizeof(fname));
		snprintf(fname, sizeof(fname) - 1, "%s.fb.%u", polyfile, fb_idx);
		if(fb_fileread(fb[fb_idx], fname) < 0)
			return 1;
	}

	I = 1 << I_dim;
	I_half = I >> 1;
	J = I >> 1;
	I_mask = I - 1;
	IJ = J << I_dim;
	sigma = cfg->skew * cfg->skew;
	mpz_init(q);
	mpz_init(tmp);

	for(side = 0; side < sizeof(fb) / sizeof(fb[0]); side++)
	{
		S[side] = (u_int8_t *) malloc(SIEVE_SIZE + MEMSET_MIN);

		mpz_init_set_ui(cofact_mul[side], 1);
		for(fb_idx = 0; fb_idx < fb[side]->n_small; fb_idx++)
		{
			u_int32_t p, pp;

			p = fb[side]->p_small[fb_idx];
			if(p > SIEVE_SIZE * 8)
				break;
			if(p <= 2)
				continue;
			for(pp = p; pp < SIEVE_SIZE; pp *= p);
			mpz_mul_ui(cofact_mul[side], cofact_mul[side], pp);
		}
		fb[side]->n_line_sieve = fb_idx;

		fb[side]->lr_small = (u_int32_t *) mmap(NULL, (fb_idx * sizeof(u_int32_t) + ALIGN_LEN - 1) & ~(ALIGN_LEN - 1),
				PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, 0, 0);
		fb[side]->pos_small = (u_int32_t *) mmap(NULL, (fb_idx * sizeof(u_int32_t) + ALIGN_LEN - 1) & ~(ALIGN_LEN - 1),
				PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, 0, 0);
	}
	SS = (u_int8_t *) malloc(SIEVE_SIZE);
	memset(SS, 0, SIEVE_SIZE);

	for(i = 0; i < sizeof(roots) / sizeof(roots[0]); i++)
		mpz_init(roots[i]);

	n_buckets = IJ >> BUCKET_SHIFT;
	bucket_alloc = (n_buckets + 1) * BUCKET_BYTES;

	for(side = 0; side < sizeof(buckets) / sizeof(buckets[0]); side++)
	{
		u_int8_t *ptr;
		bucket_map[side] = (u_int8_t *) mmap(NULL, bucket_alloc, PROT_READ | PROT_WRITE,
				MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
		if(bucket_map[side] == (u_int8_t *) MAP_FAILED)
		{
			fprintf(stderr, "failed to map space for buckets\n");
			exit(-1);
		}
		madvise(bucket_map[side], bucket_alloc, MADV_RANDOM);

		buckets[side] = (bucket_t **) malloc(n_buckets * sizeof(bucket_t *));
#ifdef BUCKET_DEBUG
		bucket_counts[side] = (size_t *) malloc(n_buckets * sizeof(size_t));
#endif

		ptr = (u_int8_t *) BUCKET_START(bucket_map[side]);
		ptr += BUCKET_BYTES;

		for(i = 0; i < n_buckets; i++, ptr += BUCKET_BYTES)
		{
			buckets[side][i] = (bucket_t *) ptr;
			*ptr = 0; /* prefault the first page of each bucket */
		}
	}

	sieve_info_init_norm_data(cfg);

	for(side = 0; side < POLY_CNT; side++)
	{
		unsigned long lim = cfg->lim[side];

		mpz_init (cfg->BB[side]);
		mpz_init (cfg->BBB[side]);
		mpz_init (cfg->BBBB[side]);
		mpz_ui_pow_ui(cfg->BB[side], lim, 2);
		mpz_mul_ui(cfg->BBB[side], cfg->BB[side], lim);
		mpz_mul_ui(cfg->BBBB[side], cfg->BBB[side], lim);
	}

	cofact_init(cfg);

	ltrans_total = vtrans_total = lsieve_total = vsieve_total = 0.;
	scheck_total = lresieve_total = vresieve_total = 0.;
	pm1_total = pp1_total = ecm_total = 0.;
	rel_count = 0;

	while(qgen_next_q(qgen, q))
	{
		n_roots = mpzpoly_get_roots(roots, cfg->poly[q_side], q);
		if(!n_roots)
			continue;

		gmp_fprintf(output, "# %Zd, n_roots=%u\n", q, n_roots);
		for(root_idx = 0; root_idx < n_roots; root_idx++)
		{
			lat_t cd_lat;

			if(reduce_lattice_skew(cd_lat, mpz_get_ui(roots[root_idx]), mpz_get_ui(q), sigma) < 0)
			{
				gmp_fprintf(output, "# reduction of (%Zd,%Zd) failed\n", roots[root_idx], q);
				continue;
			}
			gmp_fprintf(output, "# (%Zd,%Zd) -> (%lld,%lld),(%lld,%lld)\n", roots[root_idx], q, cd_lat->a0, cd_lat->b0, cd_lat->a1, cd_lat->b1);
			if(reject_lattice(cd_lat) < 0)
			{
				gmp_fprintf(output, "# lattice reduction of (%Zd,%Zd) rejected\n", roots[root_idx], q);
				continue;
			}

			sieve_info_update_norm_data(cfg, cd_lat, q_side, q, I_half);
			for(side = 0; side < POLY_CNT; side++)
			{
				cfg->log_steps_max[side] = fb_make_steps(cfg->log_steps[side], cfg->lim[side], cfg->scale[side] * LOG_SCALE);
			}

#ifdef TEST_TRANSLATE
			mpz_set_si(q0, cd_lat->a0);
			mpz_set_si(q1, cd_lat->b0);
			mpzpoly_eval_ab(tmp, cfg->poly[q_side], q0, q1);
			mpz_mod(tmp, tmp, q);
			if(mpz_cmp_ui(tmp, 0) != 0)
			{
				gmp_fprintf(output, "BAD EVAL a0,b0 = %Zd\n", tmp);
				continue;
			}

			mpz_set_si(q0, cd_lat->a0);
			mpz_mul_ui(q0, q0, 43);
			mpz_set_si(tmp, cd_lat->a1);
			mpz_mul_ui(tmp, tmp, 313);
			mpz_add(q0, q0, tmp);
			mpz_set_si(q1, cd_lat->b0);
			mpz_mul_ui(q1, q1, 43);
			mpz_set_si(tmp, cd_lat->b1);
			mpz_mul_ui(tmp, tmp, 313);
			mpz_add(q1, q1, tmp);
			mpzpoly_eval_ab(tmp, cfg->poly[q_side], q0, q1);
			mpz_mod(tmp, tmp, q);
			if(mpz_cmp_ui(tmp, 0) != 0)
			{
				gmp_fprintf(output, "BAD EVAL a1,b1 = %Zd\n", tmp);
				exit(-1);
			}
#endif
			for(side = 0; side < sizeof(fb) / sizeof(fb[0]); side++)
			{
				u_int32_t *p_small;
				u_int32_t *r_small;
				u_int32_t *lr_small;
				u_int64_t n_small;
				u_int64_t n_line_sieve;
				bucket_t **buckets_ptr;
				u_int64_t *log_steps;
				u_int8_t log_steps_max;
				p_small = fb[side]->p_small;
				r_small = fb[side]->r_small;
				lr_small = fb[side]->lr_small;
				n_small = fb[side]->n_small;
				n_line_sieve = fb[side]->n_line_sieve;
				buckets_ptr = buckets[side];
				log_steps = cfg->log_steps[side];
				log_steps_max = cfg->log_steps_max[side];


				/* translate small prime below line sieving threshold */
				tin = dbltime();
#if USE_OPENCL
                {
                    int ret = 0;
                	cl_int err = CL_SUCCESS;
#if PRINTF_PROGRESS_REPORT
                    printf("factor base translation: lines, n_small=%d\n", n_line_sieve);
#endif

                	/* the kernel */
                	ocl_kern_state_t kern_root_in_qlattice31 = { 0 };
                    ret = ocl_kern_setup(&ocl_state, "kern_root_in_qlattice31", &kern_root_in_qlattice31, n_line_sieve);
                    if (ret) {
                        fprintf(stderr, "error ocl_kern_setup kern_root_in_qlattice31\n");
                        exit(-1);
                    }
#if PRINTF_PROGRESS_REPORT
                    printf("root_in_qlattice31: local=%d, global=%d\n", kern_root_in_qlattice31.local_work_size, kern_root_in_qlattice31.global_work_size);
#endif
                    
                    /* the buffers */
                    cl_mem dev_root = 0;    /* u_int32_t */
                    cl_mem dev_p = 0;       /* u_int32_t */
                    cl_mem dev_lat = 0;     /* lat_t     */
                    cl_mem dev_out = 0;     /* u_int32_t */
                    
                    dev_root = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(u_int32_t) * n_line_sieve, NULL, &err);
                    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x %s:%d\n", err, __FILE__, __LINE__); exit(-1); }
                    dev_p = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(u_int32_t) * n_line_sieve, NULL, &err);
                    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x %s:%d\n", err, __FILE__, __LINE__); exit(-1); }
                    dev_lat = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(lat_t), NULL, &err);
                    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x %s:%d\n", err, __FILE__, __LINE__); exit(-1); }
                    dev_out = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_WRITE, sizeof(u_int32_t) * n_line_sieve, NULL, &err);
                    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x %s:%d\n", err, __FILE__, __LINE__); exit(-1); }

                    err = clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
                        dev_root, CL_TRUE, 0, sizeof(u_int32_t) * n_line_sieve,
                        r_small, 0, NULL, NULL);
                    err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
                        dev_p, CL_TRUE, 0, sizeof(u_int32_t) * n_line_sieve,
                        p_small, 0, NULL, NULL);
                    err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
                        dev_lat, CL_TRUE, 0, sizeof(lat_t),
                        cd_lat, 0, NULL, NULL);
                    if (err != CL_SUCCESS) { fprintf(stderr, "clEnqueueWriteBuffer error %.8x %s:%d\n", err, __FILE__, __LINE__); exit(-1); }

                    int arg = 0;
                    err  = clSetKernelArg(kern_root_in_qlattice31.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_root);
                    err |= clSetKernelArg(kern_root_in_qlattice31.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_p);
                    err |= clSetKernelArg(kern_root_in_qlattice31.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_lat);
                    err |= clSetKernelArg(kern_root_in_qlattice31.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_out);
                    err |= clSetKernelArg(kern_root_in_qlattice31.ckKernel, arg++, sizeof(cl_int), (void*)&n_line_sieve);
                    if (CL_SUCCESS != err) { fprintf(stderr, "Error %.8x in clSetKernelArg at %s:%d\n", err, __FILE__, __LINE__); exit(-1); }

                    ret = ocl_kern_run(&ocl_state, &kern_root_in_qlattice31);
                    if (ret) { fprintf(stderr, "error in ocl_kern_run %s:%d\n", __FILE__, __LINE__); exit(-1); }
                    ocl_kern_clear(&kern_root_in_qlattice31);
                    
                    err = clEnqueueReadBuffer(ocl_state.cqCommandQueue, dev_out, CL_TRUE, 0, sizeof(u_int32_t) * n_line_sieve, lr_small, 0, NULL, NULL);
                    if (CL_SUCCESS != err) { fprintf(stderr, "Error %.8x in clEnqueueReadBuffer at %s:%d\n", err, __FILE__, __LINE__); exit(-1); }

                    if (dev_root) { clReleaseMemObject(dev_root); dev_root = 0; }
                    if (dev_p) { clReleaseMemObject(dev_p); dev_p = 0; }
                    if (dev_lat) { clReleaseMemObject(dev_lat); dev_lat = 0; }
                    if (dev_out) { clReleaseMemObject(dev_out); dev_out = 0; }
                    
#if PRINTF_PROGRESS_REPORT
                    double elapse = (double)(kern_root_in_qlattice31.stop.tv_sec - kern_root_in_qlattice31.start.tv_sec) +
                                    ((double)(kern_root_in_qlattice31.stop.tv_nsec - kern_root_in_qlattice31.start.tv_nsec))/1000000000;
                    printf("kern_root_in_qlattice31 complete n_line_sieve=%d wall=%f s ops/s = %e\n", n_line_sieve, elapse, n_line_sieve/elapse);
#endif
                }
#endif /* USE_OPENCL */
#if !USE_OPENCL || defined(TEST_TRANSLATE) || (USE_OPENCL && TEST_OCL_QLATTICE31)
#if (USE_OPENCL && TEST_OCL_QLATTICE31)
                printf("Testing ocl_root_in_qlattice31...\n");
                int skip = 0;
                int test_ocl_riq_correct = 0;
                double test_riq = dbltime();
#endif
				for(fb_idx = 0; fb_idx < n_line_sieve; fb_idx++)
				{
					u_int32_t p, r, lat_r;

					r = r_small[fb_idx];
					p = p_small[fb_idx];
#if USE_OPENCL
                    lat_r = lr_small[fb_idx];
                    if (lat_r == 0) {
                        /* FIXME */
                        skip++;
                        continue;
                    }
#if TEST_OCL_QLATTICE31
                    u_int32_t _lat_r = root_in_qlattice31(r, p, cd_lat);
                    if (_lat_r != lat_r) {
                        fprintf(stderr, "error: lat_r %d != %d = _lat_r at fb_idx=%d\n", lat_r, _lat_r, fb_idx);
                        fprintf(stderr, "  p=%d, r=%d\n", p, r);
                        fprintf(stderr, "  (a0, b0) = (%d, %d)\n", cd_lat->a0, cd_lat->b0);
                        fprintf(stderr, "  (a1, b1) = (%d, %d)\n", cd_lat->a1, cd_lat->b1);
                    }
                    else
                        test_ocl_riq_correct++;
#endif /* TEST_OCL_QLATTICE31 */
#else /* !USE_OPENCL */
					_mm_prefetch(&r_small[fb_idx + 16], _MM_HINT_T0);
					_mm_prefetch(&p_small[fb_idx + 16], _MM_HINT_T0);
					_mm_prefetch(&lr_small[fb_idx + 16], _MM_HINT_T0);
					lat_r = root_in_qlattice31(r, p, cd_lat);
					if(lat_r == 0)
					{
						/* FIXME */
						continue;
					}
					lr_small[fb_idx] = lat_r;
#endif /* USE_OPENCL */

#ifdef TEST_TRANSLATE
					//printf("# fb(%u,%u) -> lat(%d,%d)\n", p, r, p, lat_r, p);
					if(side == q_side)
					{
						mpz_set_si(q0, cd_lat->a0);
						mpz_set_si(tmp, cd_lat->a1);
						mpz_mul_ui(q0, q0, lat_r);
						mpz_add(q0, q0, tmp);

						mpz_set_si(q1, cd_lat->b0);
						mpz_set_si(tmp, cd_lat->b1);
						mpz_mul_ui(q1, q1, lat_r);
						mpz_add(q1, q1, tmp);

						mpzpoly_eval_ab(tmp, cfg->poly[side], q0, q1);
						mpz_mod(tmp, tmp, q);
						if(mpz_cmp_ui(tmp, 0) != 0)
						{
							gmp_fprintf(output, "BAD EVAL q %Zu\n", tmp);
							exit(-1);
						}
					}

					/* a = (a0 * ((lat_r * y) + (p * x))) + (a1 * y) */
					mpz_set_si(q0, cd_lat->a0);
					mpz_mul_ui(q0, q0, (lat_r * 67) + (p * 3));
					mpz_set_si(tmp, cd_lat->a1);
					mpz_mul_ui(tmp, tmp, 67);
					mpz_add(q0, q0, tmp);

					/* b = (b0 * ((lat_r * y) + (p * x))) + (b1 * y) */
					mpz_set_si(q1, cd_lat->b0);
					mpz_mul_ui(q1, q1, (lat_r * 67) + (p * 3));
					mpz_set_si(tmp, cd_lat->b1);
					mpz_mul_ui(tmp, tmp, 67);
					mpz_add(q1, q1, tmp);

					mpzpoly_eval_ab(tmp, cfg->poly[side], q0, q1);
					mpz_mod_ui(tmp, tmp, p);
					if(mpz_cmp_ui(tmp, 0) != 0)
					{
						gmp_fprintf(output, "BAD EVAL tmp=%Zu p=%u\n", tmp, p);
						mpz_set_ui(q0, r);
						mpz_set_ui(q1, p);
						mpzpoly_eval_mod(tmp, cfg->poly[side], q0, q1);
						if(mpz_cmp_ui(tmp, 0) != 0)
							gmp_fprintf(output, "BAD ROOT\n");
						exit(-1);
					}
#endif /* defined(TEST_TRANSLATE) */
				}
#if USE_OPENCL && TEST_OCL_QLATTICE31
                double elapse = dbltime() - test_riq;
                printf("Verified %d of %d correct\n", test_ocl_riq_correct, n_line_sieve - skip);
                printf("Testing took %f s, ops/sec %e\n", elapse, n_line_sieve/elapse);
#endif
#endif /* !USE_OPENCL || defined(TEST_TRANSLATE) || TEST_OCL_QLATTICE31 */
				tout = dbltime();
				ltrans_time = tout - tin;
				fprintf(output, "# translate line time: %fs\n", ltrans_time);
				ltrans_total += ltrans_time;

				fb_idx = n_line_sieve;
				/* translate rest of small factor base elements onto lattice and bucket vectors */
				tin = dbltime();
				for(i = 0; i < n_buckets; i++)
				{
					buckets_ptr[i] = BUCKET_START(buckets_ptr[i]);
#ifdef BUCKET_DEBUG
					bucket_counts[side][i] = 0;
#endif
				}

#if USE_OPENCL
                lat_t *hst_ef_lat = (lat_t *)malloc(sizeof(lat_t) * n_small);
                if (!hst_ef_lat) { fprintf(stderr, "error: could not malloc hst_ef_lat %s:%d\n", __FILE__, __LINE__); exit(-1); }
                cl_int *hst_out_red = (int *)malloc(sizeof(cl_int) * n_small);
                if (!hst_out_red) { fprintf(stderr, "error: could not malloc hst_out_red %s:%d\n", __FILE__, __LINE__); exit(-1); }
                
                {
                    int ret = 0;
                	cl_int err = CL_SUCCESS;
#if PRINTF_PROGRESS_REPORT
                    printf("factor base translation: vectors, n_small=%d\n", n_small);
#endif
                    u_int64_t base_idx, n_batch;
                    const u_int64_t max_n_batch = 1024*1024;

                    for(base_idx = fb_idx; base_idx < n_small; base_idx += n_batch)
                    {
                    	n_batch = n_small - base_idx;
                    	if(n_batch > max_n_batch)
                    		n_batch = max_n_batch;
                        
                    	/* the kernel */
                    	ocl_kern_state_t kern_root_in_qlattice31_reduce_lattice31 = { 0 };
                        ret = ocl_kern_setup(&ocl_state, "kern_root_in_qlattice31_reduce_lattice31", &kern_root_in_qlattice31_reduce_lattice31, n_batch);
                        if (ret) {
                            fprintf(stderr, "error ocl_kern_setup kern_root_in_qlattice31_reduce_lattice31\n");
                            exit(-1);
                        }
#if PRINTF_PROGRESS_REPORT
                        printf("kern_root_in_qlattice31_reduce_lattice31: local=%d, global=%d\n", kern_root_in_qlattice31_reduce_lattice31.local_work_size, kern_root_in_qlattice31_reduce_lattice31.global_work_size);
#endif
                        
                        /* the buffers */
                        cl_mem dev_root = 0;    /* u_int32_t */
                        cl_mem dev_p = 0;       /* u_int32_t */
                        cl_mem dev_lat = 0;     /* lat_t     */
                        cl_mem dev_ef_lat = 0;  /* lat_t     */
                        cl_mem dev_out_red = 0; /* cl_int    */
                        
                        dev_root = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(u_int32_t) * n_batch, NULL, &err);
                        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x %s:%d\n", err, __FILE__, __LINE__); exit(-1); }
                        dev_p = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(u_int32_t) * n_batch, NULL, &err);
                        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x %s:%d\n", err, __FILE__, __LINE__); exit(-1); }
                        dev_lat = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_ONLY, sizeof(lat_t), NULL, &err);
                        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x %s:%d\n", err, __FILE__, __LINE__); exit(-1); }
                        dev_ef_lat = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_WRITE, sizeof(lat_t) * n_batch, NULL, &err);
                        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x %s:%d\n", err, __FILE__, __LINE__); exit(-1); }
                        dev_out_red = clCreateBuffer(ocl_state.cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_int) * n_batch, NULL, &err);
                        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer error %.8x %s:%d\n", err, __FILE__, __LINE__); exit(-1); }

                        err = clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
                            dev_root, CL_TRUE, 0, sizeof(u_int32_t) * n_batch,
                            &r_small[base_idx], 0, NULL, NULL);
                        err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
                            dev_p, CL_TRUE, 0, sizeof(u_int32_t) * n_batch,
                            &p_small[base_idx], 0, NULL, NULL);
                        err |= clEnqueueWriteBuffer(ocl_state.cqCommandQueue,
                            dev_lat, CL_TRUE, 0, sizeof(lat_t),
                            cd_lat, 0, NULL, NULL);
                        if (err != CL_SUCCESS) { fprintf(stderr, "clEnqueueWriteBuffer error %.8x %s:%d\n", err, __FILE__, __LINE__); exit(-1); }

                        int arg = 0;
                        err  = clSetKernelArg(kern_root_in_qlattice31_reduce_lattice31.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_root);
                        err |= clSetKernelArg(kern_root_in_qlattice31_reduce_lattice31.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_p);
                        err |= clSetKernelArg(kern_root_in_qlattice31_reduce_lattice31.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_lat);
                        err |= clSetKernelArg(kern_root_in_qlattice31_reduce_lattice31.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_ef_lat);
                        err |= clSetKernelArg(kern_root_in_qlattice31_reduce_lattice31.ckKernel, arg++, sizeof(cl_mem), (void*)&dev_out_red);
                        err |= clSetKernelArg(kern_root_in_qlattice31_reduce_lattice31.ckKernel, arg++, sizeof(cl_uint), (void*)&I);
                        err |= clSetKernelArg(kern_root_in_qlattice31_reduce_lattice31.ckKernel, arg++, sizeof(cl_int), (void*)&n_batch);
                        if (CL_SUCCESS != err) { fprintf(stderr, "Error %.8x in clSetKernelArg at %s:%d\n", err, __FILE__, __LINE__); exit(-1); }

                        ret = ocl_kern_run(&ocl_state, &kern_root_in_qlattice31_reduce_lattice31);
                        if (ret) { fprintf(stderr, "error in ocl_kern_run %s:%d\n", __FILE__, __LINE__); exit(-1); }
                        ocl_kern_clear(&kern_root_in_qlattice31_reduce_lattice31);
                        
                        err = clEnqueueReadBuffer(ocl_state.cqCommandQueue, dev_ef_lat, CL_TRUE, 0, sizeof(lat_t) * n_batch, &hst_ef_lat[base_idx], 0, NULL, NULL);
                        err |= clEnqueueReadBuffer(ocl_state.cqCommandQueue, dev_out_red, CL_TRUE, 0, sizeof(cl_int) * n_batch, &hst_out_red[base_idx], 0, NULL, NULL);
                        if (CL_SUCCESS != err) { fprintf(stderr, "Error %.8x in clEnqueueReadBuffer at %s:%d\n", err, __FILE__, __LINE__); exit(-1); }

                        if (dev_root) { clReleaseMemObject(dev_root); dev_root = 0; }
                        if (dev_p) { clReleaseMemObject(dev_p); dev_p = 0; }
                        if (dev_lat) { clReleaseMemObject(dev_lat); dev_lat = 0; }
                        if (dev_ef_lat) { clReleaseMemObject(dev_ef_lat); dev_ef_lat = 0; }
                        if (dev_out_red) { clReleaseMemObject(dev_out_red); dev_out_red = 0; }

#if PRINTF_PROGRESS_REPORT
                        double elapse = (double)(kern_root_in_qlattice31_reduce_lattice31.stop.tv_sec - kern_root_in_qlattice31_reduce_lattice31.start.tv_sec) +
                                        ((double)(kern_root_in_qlattice31_reduce_lattice31.stop.tv_nsec - kern_root_in_qlattice31_reduce_lattice31.start.tv_nsec))/1000000000;
                        printf("kern_root_in_qlattice31_reduce_lattice31 complete n_batch=%d wall=%f s ops/s = %e\n", n_batch, elapse, n_batch/elapse);
#endif
                    }
                }
#endif /* USE_OPENCL */

#if (USE_OPENCL && TEST_OCL_VEC_RED)
                printf("Testing ocl_root_in_qlattice31_reduce_lattice31...\n");
                int skip2 = 0;
                int test_ocl_riq2_correct = 0;
                int test_ocl_riqrl_correct = 0;
                double test_riqrl = dbltime();
#endif
				for(; fb_idx < n_small; fb_idx++)
				{
					u_int32_t p, r, lat_r, logp;
					lat_t ef_lat;

					r = r_small[fb_idx];
					p = p_small[fb_idx];

					if(side == q_side && mpz_cmp_ui(q, p) == 0)
						continue;

					_mm_prefetch(&p_small[fb_idx + 16], _MM_HINT_T0);
#if USE_OPENCL
                    memcpy(ef_lat, &hst_ef_lat[fb_idx], sizeof(lat_t));
#if TEST_OCL_VEC_RED

                    lat_r = root_in_qlattice31(r, p, cd_lat);
                    if(lat_r == 0)
                    {
                    	skip2++;
                    	continue;
                    }
#endif
#else
					_mm_prefetch(&r_small[fb_idx + 16], _MM_HINT_T0);
					lat_r = root_in_qlattice31(r, p, cd_lat);
					if(lat_r == 0)
					{
						continue;
					}
#endif /* USE_OPENCL */
					logp = find_logp(log_steps, log_steps_max, p);

#ifdef TEST_TRANSLATE
					//printf("# fb(%u,%u) -> lat(%d,%d)\n", r, p, lat_r, p);
					if(side == q_side)
					{
						mpz_set_si(q0, cd_lat->a0);
						mpz_set_si(tmp, cd_lat->a1);
						mpz_mul_ui(q0, q0, lat_r);
						mpz_add(q0, q0, tmp);

						mpz_set_si(q1, cd_lat->b0);
						mpz_set_si(tmp, cd_lat->b1);
						mpz_mul_ui(q1, q1, lat_r);
						mpz_add(q1, q1, tmp);

						mpzpoly_eval_ab(tmp, cfg->poly[side], q0, q1);
						mpz_mod(tmp, tmp, q);
						if(mpz_cmp_ui(tmp, 0) != 0)
						{
							gmp_fprintf(output, "BAD EVAL q %Zu\n", tmp);
							exit(-1);
						}
					}

					/* a = (a0 * ((lat_r * y) + (p * x))) + (a1 * y) */
					mpz_set_si(q0, cd_lat->a0);
					mpz_mul_ui(q0, q0, ((u_int64_t) lat_r * 17) + ((u_int64_t) p * 3));
					mpz_set_si(tmp, cd_lat->a1);
					mpz_mul_ui(tmp, tmp, 17);
					mpz_add(q0, q0, tmp);

					/* b = (b0 * ((lat_r * y) + (p * x))) + (b1 * y) */
					mpz_set_si(q1, cd_lat->b0);
					mpz_mul_ui(q1, q1, ((u_int64_t) lat_r * 17) + ((u_int64_t) p * 3));
					mpz_set_si(tmp, cd_lat->b1);
					mpz_mul_ui(tmp, tmp, 17);
					mpz_add(q1, q1, tmp);

					mpzpoly_eval_ab(tmp, cfg->poly[side], q0, q1);
					mpz_mod_ui(tmp, tmp, p);
					if(mpz_cmp_ui(tmp, 0) != 0)
					{
						gmp_fprintf(output, "BAD EVAL tmp=%Zu p=%u\n", tmp, p);
						exit(-1);
					}
#endif
#if USE_OPENCL
#if TEST_OCL_VEC_RED
                    lat_t _ef_lat = { 0 };
                    int _out_red = reduce_lattice31(_ef_lat, lat_r, p, I);
                    if (_out_red != hst_out_red[fb_idx]) {
                        fprintf(stderr, "error: hst_out_red[fb_idx] = %d != %d = _out_red fb_idx=%d %s:%d\n", hst_out_red[fb_idx], _out_red, fb_idx, __FILE__, __LINE__);
                    }
                    else if (_out_red == -1)
                        test_ocl_riqrl_correct++;
                    else if ((_ef_lat->a0 != hst_ef_lat[fb_idx]->a0) ||
                        (_ef_lat->b0 != hst_ef_lat[fb_idx]->b0) ||
                        (_ef_lat->a1 != hst_ef_lat[fb_idx]->a1) ||
                        (_ef_lat->b1 != hst_ef_lat[fb_idx]->b1))
                    {
                        fprintf(stderr, "error: ef_lat disagree fb_idx=%d %s:%d\n", fb_idx, __FILE__, __LINE__);
                        fprintf(stderr, "  %d\t%d\n", _ef_lat->a0, hst_ef_lat[fb_idx]->a0);
                        fprintf(stderr, "  %d\t%d\n", _ef_lat->b0, hst_ef_lat[fb_idx]->b0);
                        fprintf(stderr, "  %d\t%d\n", _ef_lat->a1, hst_ef_lat[fb_idx]->a1);
                        fprintf(stderr, "  %d\t%d\n", _ef_lat->b1, hst_ef_lat[fb_idx]->b1);
                    }
                    else
                        test_ocl_riqrl_correct++;
#endif
                    if(hst_out_red[fb_idx] < 0)
#else
					if(reduce_lattice31(ef_lat, lat_r, p, I) < 0)
#endif
					{
						//printf("# reduce_lattice31 failed for (%u,%u)\n", lat_r, p);
						continue;
					}

					//gmp_printf("# (%u,%u) -> (%d,%d)(%d,%d)\n", lat_r, p, ef_lat->a0, ef_lat->b0, ef_lat->a1, ef_lat->b1);
#ifdef TEST_TRANSLATE
					{
						int k, l;
						int64_t a, b, c, d;

						for(k = 0; k < 500; k++)
						for(l = 0; l < 500; l++)
						{
							c = (k * ef_lat->a0) + (l * ef_lat->a1);
							d = (k * ef_lat->b0) + (l * ef_lat->b1);
							if(c >= I_half || c <= -((int32_t) I_half) || d >= J || d < 0)
								continue;
							a = (c * cd_lat->a0) + (d * cd_lat->a1);
							b = (c * cd_lat->b0) + (d * cd_lat->b1);

							if(b < 0)
							{
								a = -a;
								b = -b;
							}
							if(coprime30(a, b))
								continue;
							//printf("k: %d l: %d c: %lld d: %lld a: %lld b: %lld\n", k, l, c, d, a, b);
							if(side == q_side)
							{
								mpz_set_si(q0, a);
								mpz_set_si(q1, b);
								mpzpoly_eval_ab(tmp, cfg->poly[side], q0, q1);
								mpz_mod(tmp, tmp, q);
								if(mpz_cmp_ui(tmp, 0) != 0)
								{
									gmp_fprintf(output, "BAD VECTOR q %Zd\n", tmp);
									exit(-1);
								}
							}

							mpz_set_si(q0, a);
							mpz_set_si(q1, b);
							mpzpoly_eval_ab(tmp, cfg->poly[side], q0, q1);
							mpz_mod_ui(tmp, tmp, p);
							if(mpz_cmp_ui(tmp, 0) != 0)
							{
								gmp_fprintf(output, "BAD VECTOR p %Zd\n", tmp);
								exit(-1);
							}
						}
					}
#endif
					{
						int64_t a, b, c, d;
						u_int64_t x, inc_a, inc_c;
						u_int32_t v, bound0, bound1;

						inc_a = (((u_int64_t) ef_lat->b0) << I_dim) + ef_lat->a0;
						inc_c = (((u_int64_t) ef_lat->b1) << I_dim) + ef_lat->a1;
						bound0 = (u_int32_t) (-ef_lat->a0);
						bound1 = (u_int32_t) (I - ef_lat->a1);

						x = I_half;
						v = x;
						if(v >= bound1) x += inc_a;
						if(v < bound0) x += inc_c;
						if(x >= IJ) continue;

						do
						{
							v = (x & I_mask);
							c = v;
							c -= I_half;
							d = (int64_t) (x >> I_dim);

							a = (c * cd_lat->a0) + (d * cd_lat->a1);
							b = (c * cd_lat->b0) + (d * cd_lat->b1);
							if(a < 0)
								a = -a;
							if(b < 0)
								b = -b;

							//printf("(c,d) = (%lld,%lld) (a,b) = (%lld,%lld)\n", c, d, a, b);
							if(coprime30(a, b) == 0)
							{
#ifdef TEST_TRANSLATE
								if(side == q_side)
								{
									mpz_set_si(q0, a);
									mpz_set_si(q1, b);
									mpzpoly_eval_ab(tmp, cfg->poly[side], q0, q1);
									mpz_mod(tmp, tmp, q);
									if(mpz_cmp_ui(tmp, 0) != 0)
									{
										gmp_fprintf(output, "BAD VECTOR q %Zd\n", tmp);
										exit(-1);
									}
								}

								mpz_set_si(q0, a);
								mpz_set_si(q1, b);
								mpzpoly_eval_ab(tmp, cfg->poly[side], q0, q1);
								mpz_mod_ui(tmp, tmp, p);
								if(mpz_cmp_ui(tmp, 0) != 0)
								{
									gmp_fprintf(output, "BAD VECTOR p %Zd\n", tmp);
									exit(-1);
								}
#endif
								u_int32_t bucket_idx;
								u_int32_t bucket_off;
								bucket_t *bucket;

								bucket_idx = (x >> BUCKET_SHIFT);
								bucket_off = (x & BUCKET_MASK);

								bucket = buckets_ptr[bucket_idx];
								bucket->logp = logp;
								bucket->r = bucket_off;
								bucket->p = p;
								bucket++;
								buckets_ptr[bucket_idx] = bucket;
								_mm_prefetch(bucket, _MM_HINT_T0);
#ifdef BUCKET_DEBUG
								bucket_counts[side][bucket_idx]++;
#endif
							}
							if(v >= bound1) x += inc_a;
							if(v < bound0) x += inc_c;
						} while(x < IJ);
						//printf("vector hit_cnt: %u\n", hit_cnt);
					}
				}

#if USE_OPENCL
                if (hst_ef_lat) { free(hst_ef_lat); hst_ef_lat = NULL; }
                if (hst_out_red) { free(hst_out_red); hst_out_red = NULL; }
                
#if TEST_OCL_VEC_RED
                printf("Verified %d of %d reduce_lattice31\n", test_ocl_riqrl_correct, n_small - n_line_sieve - skip2);
                double elapse2 = dbltime() - test_riqrl;
                printf("Testing took %f s, ops/sec %e\n", elapse2, n_small/elapse2);
#endif
#endif

				tout = dbltime();
				vtrans_time = tout - tin;
				fprintf(output, "# translate vector time: %fs\n", vtrans_time);
				vtrans_total += vtrans_time;
			} /* end of side loop */

			/* begin sieving */
			{
				int64_t a, b, c, d;
				u_int32_t bucket_idx;
				u_int32_t pos;
				double tin, tout;

				bucket_idx = 0;
				lsieve_time = vsieve_time = scheck_time = lresieve_time = vresieve_time = 0.;

				for(d = 0; d < J; d++)
				{
					/* calculate initial line position for this d */
					tin = dbltime();
					{
						for(side = 0; side < POLY_CNT; side++)
						{
							u_int32_t lat_r, p, pos_r;
							u_int32_t *p_small, *lr_small, *pos_small;

							p_small = fb[side]->p_small;
							lr_small = fb[side]->lr_small;
							pos_small = fb[side]->pos_small;

							for(fb_idx = 0; fb_idx < fb[RPOLY_IDX]->n_line_sieve; fb_idx++)
							{
								u_int32_t lat_r, p;

								lat_r = *(lr_small++);
								p = *(p_small++);
 								pos_r = (((u_int64_t) lat_r * (u_int64_t) d) + I_half) % p;
								*(pos_small++) = pos_r;
							}
						}
					}
					tout = dbltime();
					lsieve_time += (tout - tin);
					for(c = 0; c < I; c += SIEVE_SIZE, bucket_idx++)
					{
						u_int32_t off;
						candidate_t *cand_list, *cand;
						bucket_t **buckets_ptr, *bucket, *bucket_end;

						cand_list = NULL;

						for(side = 0; side < POLY_CNT; side++)
						{
							u_int8_t logp;
							u_int32_t p, pos_r, inc_ptr;
							u_int32_t *p_small, *pos_small;
							u_int8_t *SS_ptr, *SS_end;
							double log_scale;

							p_small = fb[side]->p_small;
							pos_small = fb[side]->pos_small;
							log_scale = cfg->scale[side] * LOG_SCALE;

							/* initialize sieving norms for the side */
							if(side == RPOLY_IDX)
								init_rat_norms_bucket_region(S[side], cfg, bucket_idx, I_dim);
							else
								init_alg_norms_bucket_region(S[side], cfg, bucket_idx, I_dim);

							/* sieve buckets */
							tin = dbltime();
							buckets_ptr = buckets[side];
							bucket_end = buckets_ptr[bucket_idx];
							bucket = BUCKET_START(bucket_end);
#ifdef BUCKET_DEBUG
							assert((bucket_end - bucket) == bucket_counts[side][bucket_idx]);
							/*
							fprintf(output, "COUNT: %u\n", bucket_counts[side][bucket_idx]);
							{
								u_int32_t *ptr, x;

								x = 0x34343434;
								for(ptr = (u_int32_t *) bucket; ptr < (u_int32_t *) bucket_end; ptr++)
									x ^= (((x << 10) | (x >> (32 - 10))) + *ptr);
								fprintf(output, "HASH: 0x%08x\n", x);
							}
							*/
#endif
							for(; bucket < bucket_end; bucket++)
							{
								SS[bucket->r] += bucket->logp;
							}
							SminusS(S[side], &S[side][SIEVE_SIZE], SS);

							tout = dbltime();
							vsieve_time += (tout - tin);

							/* sieve line */
							tin = dbltime();
							for(fb_idx = 0; fb_idx < fb[side]->n_line_sieve; fb_idx++)
							{
								pos_r = *(pos_small);
								p = *(p_small++);
								logp = fb_log(p, log_scale); // TODO: precomute this and shove it off somewhere
								SS_ptr = &SS[pos_r];
								SS_end = &SS[SIEVE_SIZE];
								inc_ptr = (c & 1) ? p : (p << 1); /* skip even c's when d is even */

								for(;;)
								{
									if(SS_ptr >= SS_end)
										break;
									*SS_ptr += logp;
									SS_ptr += inc_ptr;
									if(SS_ptr >= SS_end)
										break;
									*SS_ptr += logp;
									SS_ptr += inc_ptr;
									if(SS_ptr >= SS_end)
										break;
									*SS_ptr += logp;
									SS_ptr += inc_ptr;
									if(SS_ptr >= SS_end)
										break;
									*SS_ptr += logp;
									SS_ptr += inc_ptr;
								}

								/* PROFILE: remove if trial resieve division is faster than gcd
								 * else update this after resieve
								 */
								pos_r = SS_ptr - SS_end;
								*(pos_small++) = pos_r;
							}
							SminusS(S[side], &S[side][SIEVE_SIZE], SS);
							tout = dbltime();
							lsieve_time += (tout - tin);
						} /* end of side sieve loop */

						tin = dbltime();
						for(off = 0; off < SIEVE_SIZE; off++)
						{
				            if(S[RPOLY_IDX][off] > cfg->bound[RPOLY_IDX] || S[APOLY_IDX][off] > cfg->bound[APOLY_IDX])
				            	continue;

				            a = ((c - (int64_t) I_half + (int64_t) off) * cd_lat->a0) + (d * cd_lat->a1);
				            b = ((c - (int64_t) I_half + (int64_t) off) * cd_lat->b0) + (d * cd_lat->b1);
				            if(b < 0)
				            {
				            	a = -a;
				            	b = -b;
				            }

				            if(coprime30(abs_int(a), b))
				            	continue;
				            if(gcd64(abs_int(a), b) != 1)
				            	continue;

				            candidate_add(&cand_list, cfg, a, b, off);
						} /* end candidate loop */
						tout = dbltime();
						scheck_time += (tout - tin);

						if(!cand_list)
							continue;

						tin = dbltime();
						/* sort buckets */
						for(side = 0; side < POLY_CNT; side++)
						{
							buckets_ptr = buckets[side];
							bucket_end = buckets_ptr[bucket_idx];
							bucket = BUCKET_START(bucket_end);
							qsort(bucket, bucket_end - bucket, sizeof(bucket_t), bucket_cmp);
						}
						tout = dbltime();
						vresieve_time += (tout - tin);

						mpz_t f;
						mpz_init(f);

						while(cand_list)
						{
							cand = cand_list;
							cand_list = cand->next;

							for(side = 0; side < POLY_CNT; side++)
							{
								mp_bitcnt_t twos;
								u_int32_t p, pos_r;

								buckets_ptr = buckets[side];

								twos = mpz_scan1(cand->rem[side], 0);
								mpz_div_2exp(cand->rem[side], cand->rem[side], twos);

								if(side == q_side)
									candidate_add_factor_mpz(cand, side, q);

								/* resieve bucket */
								tin = dbltime();
								bucket_end = buckets_ptr[bucket_idx];
								for(bucket = BUCKET_START(bucket_end); bucket < bucket_end; bucket++)
								{
									if(bucket->r < cand->r)
										continue;
									if(bucket->r > cand->r)
										break;
									mpz_set_ui(f, bucket->p);
									candidate_add_factor_mpz(cand, side, f);
								}
								tout = dbltime();
								vresieve_time += (tout - tin);

								/* remove small primes */
								tin = dbltime();
								mpz_gcd(f, cand->rem[side], cofact_mul[side]);
								if(mpz_cmp_ui(f, 1) != 0)
								{
									candidate_add_factor_mpz(cand, side, f);
									for(;;)
									{
										if(mpz_cmp_ui(cand->rem[side], 1) == 0)
											break;
										mpz_gcd(f, f, cand->rem[side]);
										if(mpz_cmp_ui(f, 1) == 0)
											break;
										candidate_add_factor_mpz(cand, side, f);
									}
								}
								tout = dbltime();
								lresieve_time += (tout - tin);

								if(check_leftover_norm(cand, cfg, side) < 0)
									break;
							} /* end resieve side loop */

							if(side < POLY_CNT)
							{
								candidate_free(cand);
								continue;
							}

							/* queue candidate for subsequent cofactorization */
							cofact_add_candidate(cand);
						} /* end cand_list loop */
						mpz_clear(f);
						cofact_process_queue();
					} /* end for c */
				} /* end for d */

				fprintf(output, "# line sieve %fs vector sieve %fs sieve check %fs line resieve %fs vector resieve %fs\n",
						lsieve_time, vsieve_time, scheck_time, lresieve_time, vresieve_time);
				lsieve_total += lsieve_time;
				vsieve_total += vsieve_time;
				scheck_total += scheck_time;
				lresieve_total += lresieve_time;
				vresieve_total += vresieve_time;
				fflush(output);
			} /* end sieving */
		} /* end of root loop */
	} /* end of qgen loop */

	cofact_flush();
	fprintf(output, "# number of relations total %u\n", rel_count);
	fprintf(output, "# line sieve total %fs vector sieve total %fs sieve check total %fs\n",
			lsieve_total, vsieve_total, scheck_total);
	fprintf(output, "# line resieve total %fs vector resieve total %fs\n",
		lresieve_total, vresieve_total);
	fprintf(output, "# pm1 total %fs pp1 total %fs ecm total %fs\n",
			pm1_total, pp1_total, ecm_total);
	
#if USE_OPENCL
    ocl_close(&ocl_state);
#endif

	if(output != stdout)
		fclose(output);
	qgen_clear(qgen);
	for(side = 0; side < sizeof(fb) / sizeof(fb[0]); side++)
	{
		mpz_clear(cofact_mul[side]);
		if(fb[side]->lr_small)
			munmap(fb[side]->lr_small, (fb[side]->n_line_sieve * sizeof(u_int32_t) + ALIGN_LEN - 1) & ~(ALIGN_LEN - 1));
		if(fb[side]->pos_small)
			munmap(fb[side]->pos_small, (fb[side]->n_line_sieve * sizeof(u_int32_t) + ALIGN_LEN - 1) & ~(ALIGN_LEN - 1));
		fb_clear(fb[side]);
	}
	mpz_clear(q0);
	mpz_clear(q1);
	//gls_config_clear(cfg);
	return 0;
}

