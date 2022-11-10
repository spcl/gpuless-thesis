CC = g++
ARCH=$(CUDA_ARCH)

stream : stream.cu Makefile
	nvcc -std=c++11 -ccbin=$(CC) stream.cu -arch=$(ARCH) -o stream

.PHONY: clean
clean :
	rm -f stream
