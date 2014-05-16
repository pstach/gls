OBJS=las.o las_norms.o cofact.o \
	ocl_util.o \
	ocl_process_ul32.o \
	ocl_process_ul64.o \
	ocl_process_ul96.o \
	ocl_process_ul128.o \
	ocl_process_ul160.o \
	ocl_process_ul192.o \
	ocl_process_ul224.o \
	ocl_process_ul256.o
TARG=las
CFLAGS=-Os -I../include -g -msse4.1
LDFLAGS=-L../cofact -L../misc -L../ularith -lcofact -lmisc -lul -lgmp -lm -lOpenCL

all: $(OBJS)
	$(CC) -o $(TARG) $(OBJS) $(LDFLAGS)

clean:
	rm -f $(TARG) $(OBJS)


