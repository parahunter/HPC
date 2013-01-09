TARGET	= libmatmult.so
LIBSRCS	= 
LIBOBJS	= lib.o

OPT	= -g -fast -fsimple=1
PIC	= -fPIC

CC	= suncc
CFLAGS= $(OPT) $(PIC) $(XOPTS)

SOFLAGS = -shared 
XLIBS	= -xlic_lib=sunperf

$(TARGET): $(LIBOBJS)
	$(CC)  $(CFLAGS) -o $@ $(SOFLAGS) $(LIBOBJS) $(XLIBS)

clean:
	@/bin/rm -f core core.* $(LIBOBJS) 

