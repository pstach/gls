/*
 * test_ulcommon.c
 *
 *  Created on: Aug 25, 2013
 *      Author: pstach
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ul_dummy.h"

int main(int argc, char *argv[])
{
	u_int32_t i, seed;

	seed = time(NULL) ^ (getpid() << 16);
	if(argc > 1)
		seed = strtoul(argv[1], NULL, 0);
	printf("SEED: 0x%08x\n", seed);
	srandom(seed);
#if 0
	{
		ul a, b, c;
		mpz_t ga, gb, gc, tst;

		ul_init(a);
		ul_init(b);
		ul_init(c);
		mpz_init(ga);
		mpz_init(gb);
		mpz_init(gc);
		mpz_init(tst);

		for(i = 0; i < 1000000; i++)
		{
			ul_rand(a);
			ul_rand(b);

			mpz_set_ul(ga, a);
			mpz_set_ul(gb, b);
			ul_add(c, a, b);
			mpz_add(gc, ga, gb);

			mpz_set_ul(tst, c);
			if(mpz_cmp(tst, gc) != 0)
			{
				printf("ul_add failed\n");
				exit(-1);
			}
		}
		mpz_clear(ga);
		mpz_clear(gb);
		mpz_clear(gc);
		mpz_clear(tst);
		ul_clear(a);
		ul_clear(b);
		ul_clear(c);
	}
	printf("ul_add success\n");

	{
		ul a, b, c;
		mpz_t ga, gb, gc, tst;

		ul_init(a);
		ul_init(b);
		ul_init(c);
		mpz_init(ga);
		mpz_init(gb);
		mpz_init(gc);
		mpz_init(tst);

		for(i = 0; i < 1000000; i++)
		{
			ul_rand(a);
			ul_rand(b);
			if(ul_cmp(a, b) < 0)
			{
				ul_set(c, a);
				ul_set(a, b);
				ul_set(b, c);
			}

			mpz_set_ul(ga, a);
			mpz_set_ul(gb, b);

			ul_sub(c, a, b);
			mpz_sub(gc, ga, gb);

			mpz_set_ul(tst, c);
			if(mpz_cmp(tst, gc) != 0)
			{
				printf("ul_sub failed\n");
				exit(-1);
			}
		}
		mpz_clear(ga);
		mpz_clear(gb);
		mpz_clear(gc);
		mpz_clear(tst);
		ul_clear(a);
		ul_clear(b);
		ul_clear(c);
	}
	printf("ul_sub success\n");

	{
		ul a, b, c, tst;
		mpz_t ga, gb, gc;

		ul_init(a);
		ul_init(b);
		ul_init(c);
		ul_init(tst);
		mpz_init(ga);
		mpz_init(gb);
		mpz_init(gc);

		for(i = 0; i < 1000000; i++)
		{
			ul_rand(a);
			ul_rand(b);

			mpz_set_ul(ga, a);
			mpz_set_ul(gb, b);

			ul_mul(c, a, b);
			mpz_mul(gc, ga, gb);

			mpz_get_ul(tst, gc);
			if(ul_cmp(c, tst) != 0)
			{
				printf("ul_mul failed\n");
				exit(-1);
			}
		}
		mpz_clear(ga);
		mpz_clear(gb);
		mpz_clear(gc);
		ul_clear(a);
		ul_clear(b);
		ul_clear(c);
		ul_clear(tst);
	}
	printf("ul_mul success\n");

/*
	{
		ul a, b;
		ul2x c;
		mpz_t ga, gb, gc, tst;

		ul_init(a);
		ul_init(b);
		ul2x_init(c);
		mpz_init(ga);
		mpz_init(gb);
		mpz_init(gc);
		mpz_init(tst);

		for(i = 0; i < 1000000; i++)
		{
			ul_rand(a);
			ul_rand(b);

			mpz_set_ul(ga, a);
			mpz_set_ul(gb, b);

			ul_mul_ul2x(c, a, b);
			printf("PREMUL\n");
			mpz_mul(gc, ga, gb);
			printf("POSTMUL\n");
			printf("HI\n");
			mpz_set_ul2x(tst, c);
			printf("BYE\n");
			if(mpz_cmp(gc, tst) != 0)
			{
				printf("ul_mul_2x failed\n");
				exit(-1);
			}
		}
		mpz_clear(ga);
		mpz_clear(gb);
		mpz_clear(gc);
		mpz_clear(tst);
		ul_clear(a);
		ul_clear(b);
		ul2x_clear(c);
	}
	printf("ul_mul_2x success\n");
*/
	{
		ul a, b, r, d;
		mpz_t ga, gb, gr, gd, tst_r, tst_d;

		ul_init(a);
		ul_init(b);
		ul_init(r);
		ul_init(d);
		mpz_init(ga);
		mpz_init(gb);
		mpz_init(gr);
		mpz_init(gd);
		mpz_init(tst_r);
		mpz_init(tst_d);

		for(i = 0; i < 2000000; i++)
		{
			ul_rand(a);
			do
				ul_set_ui(b, random());
			while(ul_cmp_ui(b, 0) == 0);
			mpz_set_ul(ga, a);
			mpz_set_ul(gb, b);

			ul_divrem(d, r, a, b);
			mpz_set_ul(tst_d, d);
			mpz_set_ul(tst_r, r);

			mpz_fdiv_qr(gd, gr, ga, gb);

			if(mpz_cmp(gd, tst_d) != 0 || mpz_cmp(gr, tst_r) != 0)
			{
				printf("ul_divrem failed\n");
				exit(-1);
			}
		}
		mpz_clear(ga);
		mpz_clear(gb);
		mpz_clear(gr);
		mpz_clear(gd);
		mpz_clear(tst_r);
		mpz_clear(tst_d);
		ul_clear(a);
		ul_clear(b);
		ul_clear(r);
		ul_clear(d);
	}
	printf("ul_divrem success\n");

	{
		ul a;
		mpz_t ga;
		int pos, gpos;

		ul_init(a);
		mpz_init(ga);

		for(i = 0; i < 1000000; i++)
		{
			ul_rand(a);
			mpz_set_ul(ga, a);

			pos = ul_bscan_fwd(a);
			gpos = mpz_scan1(ga, 0);

			if(pos != gpos)
			{
				printf("ul_bscan_fwd failed\n");
				exit(-1);
			}
		}
		mpz_clear(ga);
		ul_clear(a);
	}
	printf("ul_bscan_fwd success\n");
#ifdef BROKEN_AND_UNNEEDED
	{
		ul a;
		mpz_t ga;
		int pos, gpos;

		ul_init(a);
		mpz_init(ga);

		for(i = 0; i < 1000000; i++)
		{
			ul_rand(a);
			mpz_set_ul(ga, a);

			pos = ul_bscan_rev(a);
			gpos = (sizeof(ul) * 8) - (mpz_sizeinbase(ga, 2) - 1);

			if(pos != gpos)
			{
				printf("ul_bscan_rev failed %d %d\n", pos, gpos);
				exit(-1);
			}
		}
		mpz_clear(ga);
		ul_clear(a);
	}
	printf("ul_bscan_rev success\n");
#endif /* BROKEN_AND_UNNEEDED */
	{
		ul a, b, r;
		mpz_t ga, gb, gr, tst;

		ul_init(a);
		ul_init(b);
		ul_init(r);
		mpz_init(ga);
		mpz_init(gb);
		mpz_init(gr);
		mpz_init(tst);

		for(i = 0; i < 1000000; i++)
		{
			ul_rand(a);
			ul_rand(b);
			mpz_set_ul(ga, a);
			mpz_set_ul(gb, b);

			ul_gcd(r, a, b);
			mpz_set_ul(tst, r);

			mpz_gcd(gr, ga, gb);

			if(mpz_cmp(tst, gr) != 0)
			{
				printf("ul_gcd failed\n");
				exit(-1);
			}
		}
		mpz_clear(ga);
		mpz_clear(gb);
		mpz_clear(gr);
		mpz_clear(tst);
		ul_clear(a);
		ul_clear(b);
		ul_clear(r);
	}
	printf("ul_gcd success\n");

	{
		ul a, b, c, tmp;
		mod n;
		mpz_t ga, gb, gc, gn, tst;

		ul_init(a);
		ul_init(b);
		ul_init(c);
		ul_init(tmp);
		mod_init(n);

		mpz_init(ga);
		mpz_init(gb);
		mpz_init(gc);
		mpz_init(gn);
		mpz_init(tst);

		for(i = 0; i < 1000000; i++)
		{
			ul_rand(a);
			ul_rand(b);
			do
				ul_rand(n->n);
			while(ul_cmp_ui(n->n, 0) == 0);
			ul_divrem(tmp, a, a, n->n);
			ul_divrem(tmp, b, b, n->n);

			mpz_set_ul(ga, a);
			mpz_set_ul(gb, b);
			mpz_set_ul(gn, n->n);
			ul_modadd(c, a, b, n);
			mpz_add(gc, ga, gb);
			mpz_mod(gc, gc, gn);
			if(mpz_cmp_ui(gc, 0) < 0)
				mpz_add(gc, gc, gn);
			mpz_set_ul(tst, c);
			if(mpz_cmp(tst, gc) != 0)
			{
				printf("ul_modadd failed\n");
				exit(-1);
			}
		}
		mpz_clear(ga);
		mpz_clear(gb);
		mpz_clear(gc);
		mpz_clear(gn);
		mpz_clear(tst);
		ul_clear(a);
		ul_clear(b);
		ul_clear(c);
		ul_clear(tmp);
		mod_clear(n);
	}
	printf("ul_modadd success\n");

	{
		ul a, b, c, tmp;
		mod n;
		mpz_t ga, gb, gc, gn, tst;

		ul_init(a);
		ul_init(b);
		ul_init(c);
		ul_init(tmp);
		mod_init(n);

		mpz_init(ga);
		mpz_init(gb);
		mpz_init(gc);
		mpz_init(gn);
		mpz_init(tst);

		for(i = 0; i < 1000000; i++)
		{
			ul_rand(a);
			ul_rand(b);
			do
				ul_rand(n->n);
			while(ul_cmp_ui(n->n, 0) == 0);
			ul_divrem(tmp, a, a, n->n);
			ul_divrem(tmp, b, b, n->n);

			mpz_set_ul(ga, a);
			mpz_set_ul(gb, b);
			mpz_set_ul(gn, n->n);
			ul_modsub(c, a, b, n);
			mpz_sub(gc, ga, gb);
			mpz_mod(gc, gc, gn);

			mpz_set_ul(tst, c);
			if(mpz_cmp(tst, gc) != 0)
			{
				printf("ul_modsub failed\n");
				exit(-1);
			}
		}
		mpz_clear(ga);
		mpz_clear(gb);
		mpz_clear(gc);
		mpz_clear(gn);
		mpz_clear(tst);
		ul_clear(a);
		ul_clear(b);
		ul_clear(c);
		ul_clear(tmp);
		mod_clear(n);
	}
	printf("ul_modsub success\n");
#endif
#ifdef BROKEN_TESTS
	{
		ul a, ma, tmp;
		mod n;
		mpz_t ga, gma, gn, gr, tst;

		ul_init(a);
		ul_init(tmp);
		mod_init(n);

		mpz_init(ga);
		mpz_init(gma);
		mpz_init(gn);
		mpz_init_set_ui(gr, 0);
		mpz_init(tst);

		for(i = 0; i < 1000000; i++)
		{
			do
				ul_rand(n->n);
			while(ul_cmp_ui(n->n, 0) == 0);
			ul_setbit(n->n, 0);

#ifdef USE_UL64 || USE_UL128
		mpz_setbit(gr, sizeof(a) * 8);
#else
#error "mpz"
		mpz_setbit(gr, (mpz_sizeinbase(n->n, 2) + 63 & ~63) * 2);
#endif
			mod_set(n, n->n);

			ul_rand(a);
			ul_divrem(tmp, a, a, n->n);

			mpz_set_ul(ga, a);
			mpz_set_ul(gn, n->n);
			ul_to_montgomery(ma, a, n);
			mpz_set_ul(gma, ma);

			mpz_mul(tst, ga, gr);
			mpz_mod(tst, tst, gn);

			if(mpz_cmp(tst, gma) != 0)
			{
				printf("ul_to_montgomery failed\n");
				gmp_printf("N= %Zd\n", gn);
				gmp_printf("A= %Zd\n", ga);
				gmp_printf("MA= %Zd\n", gma);
				gmp_printf("TST= %Zd\n", tst);
				exit(-1);
			}
		}

		ul_clear(a);
		ul_clear(tmp);
		mod_clear(n);

		mpz_clear(ga);
		mpz_clear(gma);
		mpz_clear(gn);
		mpz_clear(gr);
		mpz_clear(tst);
	}
	printf("ul_to_montgomery success\n");

	{
		ul a, b, c, tmp;
		mod n;
		mpz_t ga, gb, gc, gn, tst;

		ul_init(a);
		ul_init(b);
		ul_init(c);
		ul_init(tmp);
		mod_init(n);

		mpz_init(ga);
		mpz_init(gb);
		mpz_init(gc);
		mpz_init(gn);
		mpz_init(tst);

		for(i = 0; i < 1000000; i++)
		{
			ul_rand(a);
			ul_rand(b);
			do
				ul_rand(n->n);
			while(ul_cmp_ui(n->n, 0) == 0);
			ul_setbit(n->n, 0);
			mod_set(n, n->n);
/*
			printf("N= ");
			ul_print(n->n);
			printf("\n");
			printf("RSQ= ");
			ul_print(n->rsq);
			printf("\n");
			printf("NP= %llu\n", n->np);
*/
			ul_divrem(tmp, a, a, n->n);
			ul_divrem(tmp, b, b, n->n);

			mpz_set_ul(ga, a);
			mpz_set_ul(gb, b);
			mpz_set_ul(gn, n->n);

			ul_to_montgomery(a, a, n);
			ul_to_montgomery(b, b, n);
			ul_modmul(c, a, b, n);
			ul_from_montgomery(c, c, n);
			mpz_mul(gc, ga, gb);
			mpz_mod(gc, gc, gn);

			mpz_set_ul(tst, c);
			if(mpz_cmp(tst, gc) != 0)
			{
				printf("ul_modmul failed\n");
				printf("C= ");
				ul_print(c);
				printf("\n");
				gmp_printf("gC= %Zd\n", gc);
				exit(-1);
			}
		}
		mpz_clear(ga);
		mpz_clear(gb);
		mpz_clear(gc);
		mpz_clear(gn);
		mpz_clear(tst);
		ul_clear(a);
		ul_clear(b);
		ul_clear(c);
		ul_clear(tmp);
		mod_clear(n);
	}
	printf("ul_modmul success\n");

	{
		int ret1, ret2;
		ul a, r, n;
		mpz_t ga, gn, tst1, tst2;

		ul_init(a);
		ul_init(r);
		ul_init(n);
		mpz_init(ga);
		mpz_init(gn);
		mpz_init(tst1);
		mpz_init(tst2);

		for(i = 0; i < 1000000; i++)
		{
			do
			{
				ul_rand(n);
			} while(ul_cmp_ui(n, 0) == 0);
			ul_setbit(n, 0);

			ul_rand(a);
			ul_mod(a, a, n);

			mpz_set_ul(ga, a);
			mpz_set_ul(gn, n);
			ret1 = ul_modinv(r, a, n);
			mpz_set_ul(tst1, r);
			ret2 = mpz_invert(tst2, ga, gn);
			if(ret1 != ret2)
			{
				printf("ul_modinv failed, ret mismatch\n");
				exit(-1);
			}
			if(!ret1)
				continue;
			if(mpz_cmp(tst1, tst2) != 0)
			{
				printf("ul_modinv failed\n");
				gmp_printf("A=%Zd\n", ga);
				printf("A2=");
				ul_print(a);
				printf("\n");
				gmp_printf("N=%Zd\n", gn);
				printf("N2=");
				ul_print(n);
				printf("\n");
				gmp_printf("TST1=%Zd\nTST2=%Zd\n", tst1, tst2);
				exit(-1);
			}
		}

		ul_clear(a);
		ul_clear(r);
		ul_clear(n);
		mpz_clear(ga);
		mpz_clear(tst1);
		mpz_clear(tst2);
		mpz_clear(gn);
	}
	printf("ul_modinv success\n");
#endif
	{
		ul a, b, c, tmp;
		mod n;

		ul_init(a);
		ul_init(b);
		ul_init(c);
		ul_init(tmp);
		mod_init(n);

		for(i = 0; i < 1000000; i++)
		{
			ul_rand(a);
			do
				ul_rand(n->n);
			while(ul_cmp_ui(n->n, 0) == 0);
			ul_setbit(n->n, 0);
			mod_set(n, n->n);

			ul_divrem(tmp, a, a, n->n);
			ul_set_ui(b, 2);
			if(!ul_modinv(b, b, n->n))
				continue;

			ul_to_montgomery(a, a, n);
			ul_to_montgomery(b, b, n);

			ul_moddiv2(c, a, n);
			ul_modmul(tmp, a, b, n);
			if(ul_cmp(c, tmp) != 0)
			{
				printf("ul_moddiv2 failed\n");
				exit(-1);
			}
		}
		ul_clear(a);
		ul_clear(b);
		ul_clear(c);
		ul_clear(tmp);
		mod_clear(n);
	}
	printf("ul_moddiv2 success\n");

	{
		ul a, b, c, tmp;
		mod n;

		ul_init(a);
		ul_init(b);
		ul_init(c);
		ul_init(tmp);
		mod_init(n);

		for(i = 0; i < 1000000; i++)
		{
			ul_set_ui(b, 3);
			do
			{
				ul_rand(n->n);
				ul_setbit(n->n, 0);
				ul_mod(tmp, n->n, b);
			}
			while(ul_cmp_ui(tmp, 0) == 0);

			mod_set(n, n->n);
			ul_rand(a);
			ul_divrem(tmp, a, a, n->n);
			if(!ul_modinv(b, b, n->n))
				continue;

			ul_to_montgomery(a, a, n);
			ul_to_montgomery(b, b, n);

			ul_moddiv3(c, a, n);
			ul_modmul(tmp, a, b, n);
			if(ul_cmp(c, tmp) != 0)
			{
				printf("ul_moddiv3 failed\n");
				exit(-1);
			}
		}
		ul_clear(a);
		ul_clear(b);
		ul_clear(c);
		ul_clear(tmp);
		mod_clear(n);
	}
	printf("ul_moddiv3 success\n");
	{
		ul a, b, c, tmp;
		mod n;

		ul_init(a);
		ul_init(b);
		ul_init(c);
		ul_init(tmp);
		mod_init(n);

		for(i = 0; i < 1000000; i++)
		{
			ul_set_ui(b, 5);
			do
			{
				ul_rand(n->n);
				ul_setbit(n->n, 0);
				ul_mod(tmp, n->n, b);
			}
			while(ul_cmp_ui(tmp, 0) == 0);

			mod_set(n, n->n);
			ul_rand(a);
			ul_divrem(tmp, a, a, n->n);
			if(!ul_modinv(b, b, n->n))
				continue;

			ul_to_montgomery(a, a, n);
			ul_to_montgomery(b, b, n);

			ul_moddiv5(c, a, n);
			ul_modmul(tmp, a, b, n);
			if(ul_cmp(c, tmp) != 0)
			{
				printf("ul_moddiv5 failed\n");
				exit(-1);
			}
		}
		ul_clear(a);
		ul_clear(b);
		ul_clear(c);
		ul_clear(tmp);
		mod_clear(n);
	}
	printf("ul_moddiv5 success\n");

	{
		ul a, b, c, tmp;
		mod n;

		ul_init(a);
		ul_init(b);
		ul_init(c);
		ul_init(tmp);
		mod_init(n);

		for(i = 0; i < 1000000; i++)
		{
			ul_set_ui(b, 7);
			do
			{
				ul_rand(n->n);
				ul_setbit(n->n, 0);
				ul_mod(tmp, n->n, b);
			}
			while(ul_cmp_ui(tmp, 0) == 0);

			mod_set(n, n->n);
			ul_rand(a);
			ul_divrem(tmp, a, a, n->n);
			if(!ul_modinv(b, b, n->n))
				continue;

			ul_to_montgomery(a, a, n);
			ul_to_montgomery(b, b, n);

			ul_moddiv7(c, a, n);
			ul_modmul(tmp, a, b, n);
			if(ul_cmp(c, tmp) != 0)
			{
				printf("ul_moddiv7 failed\n");
				exit(-1);
			}
		}
		ul_clear(a);
		ul_clear(b);
		ul_clear(c);
		ul_clear(tmp);
		mod_clear(n);
	}
	printf("ul_moddiv7 success\n");

	{
		ul a, b, c, tmp;
		mod n;

		ul_init(a);
		ul_init(b);
		ul_init(c);
		ul_init(tmp);
		mod_init(n);

		for(i = 0; i < 1000000; i++)
		{
			ul_set_ui(b, 11);
			do
			{
				ul_rand(n->n);
				ul_setbit(n->n, 0);
				ul_mod(tmp, n->n, b);
			}
			while(ul_cmp_ui(tmp, 0) == 0);

			mod_set(n, n->n);
			ul_rand(a);
			ul_divrem(tmp, a, a, n->n);
			if(!ul_modinv(b, b, n->n))
				continue;

			ul_to_montgomery(a, a, n);
			ul_to_montgomery(b, b, n);

			ul_moddiv11(c, a, n);
			ul_modmul(tmp, a, b, n);
			if(ul_cmp(c, tmp) != 0)
			{
				printf("ul_moddiv11 failed\n");
				exit(-1);
			}
		}
		ul_clear(a);
		ul_clear(b);
		ul_clear(c);
		ul_clear(tmp);
		mod_clear(n);
	}
	printf("ul_moddiv11 success\n");

	{
		ul a, b, c, tmp;
		mod n;

		ul_init(a);
		ul_init(b);
		ul_init(c);
		ul_init(tmp);
		mod_init(n);

		for(i = 0; i < 1000000; i++)
		{
			ul_set_ui(b, 13);
			do
			{
				ul_rand(n->n);
				ul_setbit(n->n, 0);
				ul_mod(tmp, n->n, b);
			}
			while(ul_cmp_ui(tmp, 0) == 0);

			mod_set(n, n->n);
			ul_rand(a);
			ul_divrem(tmp, a, a, n->n);
			if(!ul_modinv(b, b, n->n))
				continue;

			ul_to_montgomery(a, a, n);
			ul_to_montgomery(b, b, n);

			ul_moddiv13(c, a, n);
			ul_modmul(tmp, a, b, n);
			if(ul_cmp(c, tmp) != 0)
			{
				printf("ul_moddiv13 failed\n");
				exit(-1);
			}
		}
		ul_clear(a);
		ul_clear(b);
		ul_clear(c);
		ul_clear(tmp);
		mod_clear(n);
	}
	printf("ul_moddiv13 success\n");

	return 0;
}
