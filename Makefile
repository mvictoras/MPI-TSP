MPICC=mpicc

all: tsp

tsp: tsp.c
			$(MPICC) -O3 -lm -o tsp tsp.c

clean:
	    rm -rf tsp
