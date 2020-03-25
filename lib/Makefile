EXEC= biu
COMMON= -I/usr/local/cuda/include
LDFLAGS= -L/usr/local/cuda/lib64 -lcuda -lcudart
EXECOBJ= biu_vector.o biu_test.o
SLIB= libbiu.so
SCU= biu_vector.cu

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
	sudo chmod a+x $(SLIB)

$(EXEC): $(EXECOBJ)
	$(CPP) $(COMMON) $^ -o $@ $(LDFLAGS) 

all:$(EXEC) $(SLIB)

clean:
	rm $(EXECOBJ) $(EXEC) $(SLIB)