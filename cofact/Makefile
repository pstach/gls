OBJS=cofact_plan.o pm1_ul64.o pm1_ul128.o pm1_mpz.o \
	pp1_ul64.o pp1_ul128.o pp1_ulmpz.o ecm_ul64.o ecm_ul128.o ecm_ulmpz.o \
	prac_bc.o
TARG=libcofact.a
CFLAGS=-Os -I../include

all: $(OBJS)
	@ar q $(TARG) $(OBJS)
	@ranlib $(TARG)
	
clean:
	rm -f $(OBJS) $(TARG)
