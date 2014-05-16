OBJS=makefb.o
TARG=makefb
CFLAGS=-Os -I../include
LDFLAGS=-L../misc -L../ularith -lmisc -lul -lgmp -lm

all: $(OBJS)
	$(CC) -o $(TARG) $(OBJS) $(LDFLAGS)

clean:
	rm -f $(TARG) $(OBJS)


