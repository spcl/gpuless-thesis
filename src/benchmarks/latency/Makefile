all: latency

.PHONY: latency
latency: latency.cu
	nvcc -arch=sm_80 -cudart=shared -o latency latency.cu
