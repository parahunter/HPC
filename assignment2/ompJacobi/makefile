TARGET	= ompJacobi.exe
LIBSRCS	= 
LIBOBJS	= main.o

# -fast -fsimple -fns -xarch=generic64 (not supported in linux)
OPT	= -g -O3 -xopenmp
PIC	= #-fPIC 
DRY = 
FLAGS =

CC	= g++
CFLAGS= $(OPT) $(PIC) $(XOPTS) $(DRY) $(FLAGS)


XLIBS	=  -lm -lpng # -lpngwriter -lpng

$(TARGET): $(LIBOBJS)
	$(CC)  $(CFLAGS) -o $@ $(LIBOBJS) $(XLIBS)

clean:
	@/bin/rm -f core core.* $(LIBOBJS)


