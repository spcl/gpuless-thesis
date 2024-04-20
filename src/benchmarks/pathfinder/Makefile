include make.config

CC := $(CUDA_DIR)/bin/nvcc -cudart shared -arch=$(CUDA_ARCH)
INCLUDE := $(CUDA_DIR)/include

SRC = pathfinder.cu

EXE = pathfinder

release:
	$(CC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 
	$(CC) pathfinder_reordered.cu -o pathfinder_reordered -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

enum:
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$$(CUDA_LIB_DIR) 

debug:
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$$(CUDA_LIB_DIR) 

debugenum:
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm -f pathfinder
