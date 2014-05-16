test_ul64_OBJS=test_ul64.o
test_ul64_TARG=test_ul64
test_ul128_OBJS=test_ul128.o
test_ul128_TARG=test_ul128
test_ulmpz_OBJS=test_ulmpz.o
test_ulmpz_TARG=test_ulmpz
test_mpzpoly_OBJS=test_mpzpoly.o
test_mpzpoly_TARG=test_mpzpoly
test_pm1_OBJS=test_pm1.o
test_pm1_TARG=test_pm1
test_pp1_OBJS=test_pp1.o
test_pp1_TARG=test_pp1
test_ecm_OBJS=test_ecm.o
test_ecm_TARG=test_ecm
CFLAGS=-Os -I../include
LDFLAGS=-L../misc -L../ularith -L../cofact -lcofact -lul -lmisc -lgmp -lm

all: $(test_ul64_OBJS) $(test_ul128_OBJS) $(test_ulmpz_OBJS) $(test_mpzpoly_OBJS) $(test_pm1_OBJS) $(test_pp1_OBJS) $(test_ecm_OBJS)
	$(CC) -o $(test_ul64_TARG) $(test_ul64_OBJS) $(LDFLAGS)
	$(CC) -o $(test_ul128_TARG) $(test_ul128_OBJS) $(LDFLAGS)
	$(CC) -o $(test_ulmpz_TARG) $(test_ulmpz_OBJS) $(LDFLAGS)
	$(CC) -o $(test_mpzpoly_TARG) $(test_mpzpoly_OBJS) $(LDFLAGS)
	$(CC) -o $(test_pm1_TARG) $(test_pm1_OBJS) $(LDFLAGS)
	$(CC) -o $(test_pp1_TARG) $(test_pp1_OBJS) $(LDFLAGS)
	$(CC) -o $(test_ecm_TARG) $(test_ecm_OBJS) $(LDFLAGS)
	
clean:
	rm -f $(test_ul64_TARG) $(test_ul64_OBJS)
	rm -f $(test_ul128_TARG) $(test_ul128_OBJS)
	rm -f $(test_ulmpz_TARG) $(test_ulmpz_OBJS)
	rm -f $(test_mpzpoly_TARG) $(test_mpzpoly_OBJS)
	rm -f $(test_pm1_TARG) $(test_pm1_OBJS)
	rm -f $(test_pp1_TARG) $(test_pp1_OBJS)
	rm -f $(test_ecm_TARG) $(test_ecm_OBJS)
