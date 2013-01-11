TARGET	= libmatmult.so
LIBSRCS	= 
LIBOBJS	= lib.o

# -fast -fsimple -fns -xarch=generic64 (not supported in linux)
<<<<<<< HEAD
OPT	= -g #-xtarget=nehalem
=======
OPT	= -g -fast
>>>>>>> 43fdb43008c6d76c2a583ba0652fa9d5e367041c
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

