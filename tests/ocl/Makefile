

test_ocl_OBJS=testmp.o config.o ocl_error.o test_ul32.o test_ul64.o test_ul96.o test_ul128.o test_ul160.o test_ul192.o test_ul224.o test_ul256.o
test_ocl_TARG=testmp


CFLAGS=-std=gnu99 -Os -I../../include 
LDFLAGS=-L../../misc -L../../ularith -L../../cofact -lcofact -lul -lmisc -lgmp -lm -lOpenCL

all: $(test_ocl_OBJS)
	$(CC) -o $(test_ocl_TARG) $(test_ocl_OBJS) $(LDFLAGS)

clean:
	rm -f $(test_ocl_TARG) $(test_ocl_OBJS)
