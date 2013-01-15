TARGET	= jacobiSequential.exe
LIBSRCS	= 
LIBOBJS	= main.o

# -fast -fsimple -fns -xarch=generic64 (not supported in linux)
OPT	= -g -fast
PIC	= -fPIC
DRY = 
FLAGS = 

CC	= suncc
CFLAGS= $(OPT) $(PIC) $(XOPTS) $(DRY) $(FLAGS)


XLIBS	= -xlic_lib=sunperf

$(TARGET): $(LIBOBJS)
	$(CC)  $(CFLAGS) -o $@ $(LIBOBJS) $(XLIBS)

clean:
	@/bin/rm -f core core.* $(LIBOBJS)

