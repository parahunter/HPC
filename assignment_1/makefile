TARGET	= libmatmult.so
LIBSRCS	= 
LIBOBJS	= lib.o

# -fast -fsimple -fns -xarch=generic64 (not supported in linux)

OPT	= -g -fast -xarch=generic
PIC	= -fPIC

CC	= suncc
CFLAGS= $(OPT) $(PIC) $(XOPTS)

SOFLAGS = -shared 
XLIBS	= -xlic_lib=sunperf

$(TARGET): $(LIBOBJS)
	$(CC)  $(CFLAGS) -o $@ $(SOFLAGS) $(LIBOBJS) $(XLIBS)

clean:
	@/bin/rm -f core core.* $(LIBOBJS) 

