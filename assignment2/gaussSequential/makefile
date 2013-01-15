TARGET	= gaussSequential.exe
LIBSRCS	= 
LIBOBJS	= main.o

# -fast -fsimple -fns -xarch=generic64 (not supported in linux)
OPT	= -g -fast -xopenmp
PIC	= -fPIC 
DRY = 
FLAGS = -L../heatmap -I../heatmap

CC	= suncc
CFLAGS= $(OPT) $(PIC) $(XOPTS) $(DRY) $(FLAGS)


XLIBS	= -xlic_lib=sunperf -lm -lpngwriter

$(TARGET): $(LIBOBJS)
	$(CC)  $(CFLAGS) -o $@ $(LIBOBJS) $(XLIBS)

clean:
	@/bin/rm -f core core.* $(LIBOBJS)

