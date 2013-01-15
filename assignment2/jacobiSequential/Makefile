#make file exercise 1

#FLAGS1 = -xautopar  -xloopinfo -xreduction # this gives auto parallel version
FLAGS1 = -xopenmp -xloopinfo #openmp
FLAGS2 = -g -fast

#all cores contribute to the clock() so multiple threads increases the clock count even though the wall time goes down

#it seems that the automatic thread optimization is faster
#however it probably does reduction

build : main.o
	suncc -o $@ main.o $(FLAGS1) $(FLAGS2)
	
clean :
	@rm -f *.o core
	@echo "cleaned"

# file dependencies
main.o : main.c
	cc -c main.c $(FLAGS1) $(FLAGS2) 
