TARGET	= libmatmult.so
LIBSRCS	= 
LIBOBJS	= lib.o

# -fast -fsimple -fns -xarch=generic64 (not supported in linux)

<<<<<<< HEAD
OPT	= -g -fast #-xrestrict
=======
OPT	= -g
>>>>>>> 123b21f8c626ea358149ac840c309e5cbe6c9a5a
PIC	= -fPIC
DRY = 
FLAGS = 

CC	= suncc
CFLAGS= $(OPT) $(PIC) $(XOPTS) $(DRY) $(FLAGS)

SOFLAGS = -shared 
XLIBS	= -xlic_lib=sunperf

$(TARGET): $(LIBOBJS)
	$(CC)  $(CFLAGS) -o $@ $(SOFLAGS) $(LIBOBJS) $(XLIBS)

clean:
	@/bin/rm -f core core.* $(LIBOBJS) 

