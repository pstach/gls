OBJS=util.o gls_config.o fb.o
TARG=libmisc.a
CFLAGS=-Os -I../include

all: $(OBJS)
	@ar q $(TARG) $(OBJS)
	@ranlib $(TARG)
	
clean:
	rm -f $(OBJS) $(TARG)
	
