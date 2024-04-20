include make.config

CC := $(CUDA_DIR)/bin/nvcc -cudart=shared 

INCLUDE := /opt/cuda/include

# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together
a.out:		main.o main_reordered.o
	$(CC)	main.o \
				-I$(INCLUDE) \
				-L$(CUDA_DIR)/lib64  \
				-lm -lcuda -lcudart -o srad_v1
	$(CC)	main_reordered.o \
				-I$(INCLUDE) \
				-L$(CUDA_DIR)/lib64  \
				-lm -lcuda -lcudart -o srad_v1_reordered

# compile main function file into object (binary)
main.o: 	main.cu \
				define.c \
				graphics.c \
				extract_kernel.cu \
				prepare_kernel.cu \
				reduce_kernel.cu \
				srad_kernel.cu \
				srad2_kernel.cu \
				compress_kernel.cu
	$(CC)	main.cu \
				-c -O3 -cudart=shared -arch=$(CUDA_ARCH)

main_reordered.o: 	main_reordered.cu \
				define.c \
				graphics.c \
				extract_kernel.cu \
				prepare_kernel.cu \
				reduce_kernel.cu \
				srad_kernel.cu \
				srad2_kernel.cu \
				compress_kernel.cu
	$(CC) main_reordered.cu \
				-c -O3 -cudart=shared -arch=$(CUDA_ARCH)

# delete all object files
clean:
	rm *.o srad_v1
