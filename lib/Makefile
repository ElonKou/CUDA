EXEC= vector
COMMON= -I/usr/local/cuda/include
LDFLAGS= -L/usr/local/cuda/lib64 -lcuda -lcudart
EXECOBJ= vector.o vector_cuda.o
SLIB= libvector.so
SCU= vector_cuda.cu

CC=gcc
CPP=g++
NVCC=nvcc 

%.o: %.c
	$(CC) $(COMMON) -c $< -o $@

%.o: %.cc
	$(CPP) $(COMMON) -c $< -o $@

%.o: %.cu
	$(NVCC) $(COMMON) -c $< -o $@

$(SLIB): $(SCU)
	$(NVCC) $(COMMON) -shared -c $< -o $@

$(EXEC): $(EXECOBJ)
	$(CPP) $(COMMON) $^ -o $@ $(LDFLAGS) 

all:$(EXEC) $(SLIB)

clean:
	rm $(EXECOBJ) $(EXEC) $(SLIB)