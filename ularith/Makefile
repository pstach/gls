OBJS=ul64.o ul128.o ulmpz.o mpzpoly.o
TARG=libul.a
CFLAGS=-Os -I../include

all: $(OBJS)
	@ar q $(TARG) $(OBJS)
	@ranlib $(TARG)
	
clean:
	rm -f $(OBJS) $(TARG)
