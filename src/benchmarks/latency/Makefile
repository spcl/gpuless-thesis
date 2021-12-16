all: latency

.PHONY: latency
latency:
	nvcc -arch=sm_80 -cudart=shared -o latency latency.cu
	c++ -L/opt/cuda/lib64 -I/opt/cuda/include -lcuda -lcudart -o latency-preinit latency-preinit.cpp
