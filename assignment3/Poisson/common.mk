# use "nvcc" to compile source files.
CC = nvcc

# the linker is also "nvcc". It might be something else with other compilers.
LD = nvcc

# location of the CUDA Toolkit
#CUDA_PATH ?= /usr/local/cuda
CUDA_PATH ?= /opt/cuda/current

# library directories
LIBDIRS = -L/usr/lib64/atlas 

# libraries to be linked with
LIBS =  -lcublas -lcblas -lgomp

# common includes and paths for CUDA
INCLUDES = -I$(CUDA_PATH)/include -I. -I$(CUDA_PATH)/samples/common/inc

# compiler flags go here.
CFLAGS = -Xcompiler -fopenmp -g -O3 $(INCLUDES) -Dcimg_use_xshm -Dcimg_use_xrandr -Xptxas=-v $(ARCH) # -keep

# linker flags go here. Currently there aren't any, but if we'll switch to
# code optimization, we might add "-s" here to strip debug info and symbols.
LDFLAGS = $(LIBDIRS) $(LIBS)

# list of generated object files.
OBJS = $(PROG).o $(PROG)_kernel.o

# list of dependencies and header files
DEPS = $(PROG).h

# top-level rule, to compile everything.
all: $(PROG) $(DEPS)

# rule to link the program
$(PROG): $(OBJS)
	$(LD) $(LDFLAGS) $(OBJS) -o $(PROG)

# now comes a meta-rule for compiling any "C" source file.

%.o: %.cu
	$(CC) $(CFLAGS) -c $<

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

%.0: %.c
	$(CC) $(CFLAGS) -c $<

# use this command to erase files.
RM = /bin/rm -f

# rule for cleaning re-compilable files.
clean:
	$(RM) $(PROG) $(OBJS) $(CUDA_PROFILE_LOG) cuda_profile_0.log

# rule for running compiled files.
run:
	./$(PROG) 
