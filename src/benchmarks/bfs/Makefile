include make.config

CC := $(CUDA_DIR)/bin/nvcc -cudart shared

INCLUDE := /opt/cuda/include

# bfs.cu is for a single run, bfs-mig.cu is for testing 100 warm runs
SRC = bfs.cu

# bfs should be compiled from bfs.cu, bfs-mig should be compiled from bfs-mig.cu
EXE = bfs 

release: $(SRC)
	$(CC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debug: $(SRC)
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
