all: latency

.PHONY: latency
latency: latency.cu
	nvcc -I../../include -cudart=shared -o latency latency.cu
