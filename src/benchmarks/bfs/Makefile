include make.config

CC := $(CUDA_DIR)/bin/nvcc -cudart shared

INCLUDE := /opt/cuda/include

SRC = bfs.cu

EXE = bfs

all: release release_batched

release_batched:
	$(CC) -DITERATIONS="100" $(SRC) -o $(EXE)_batched -I$(INCLUDE) -L$(CUDA_LIB_DIR)

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
