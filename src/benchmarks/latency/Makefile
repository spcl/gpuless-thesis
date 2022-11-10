all: latency

.PHONY: latency
latency:
	nvcc -arch=$(CUDA_ARCH) -cudart=shared -o latency latency.cu
	c++ -L$(CUDA_DIR)/lib64 -I/opt/cuda/include -lcuda -lcudart -o latency-preinit latency-preinit.cpp
