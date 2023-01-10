all: latency

.PHONY: latency
latency:
	nvcc -arch=$(CUDA_ARCH) -cudart=shared -o latency latency.cu
	c++ -L$(CUDA_DIR)/lib64 -I$(CUDA_DIR)/include -lcuda -lcudart -o latency-preinit latency-preinit.cpp

.PHONY: clean
clean:
	rm latency
