all: main

biulib.o:
	nvcc -c biulib.cu -std=c++11

main: biulib.o
	g++ biulib.cc main.cc biulib.o -lcudart -L/usr/local/cuda/lib64 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -L/usr/local/lib -o main
	rm *.o
