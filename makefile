TARGET	= libmatmult.so
LIBSRCS	= 
LIBOBJS	= lib.o

OPT	= -g
PIC	= -fPIC

CC	= suncc
CFLAGS= $(OPT) $(PIC) $(XOPTS)

SOFLAGS = -shared 
XLIBS	= 

$(TARGET): $(LIBOBJS)
	$(CC)  $(CFLAGS) -o $@ $(SOFLAGS) $(LIBOBJS) $(XLIBS)

clean:
	@/bin/rm -f core core.* $(LIBOBJS) 

