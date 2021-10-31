CC=nvcc
CCOPTS=-cudart shared -arch=sm_86
TARGETS=manykernel \
		objkernel \
		simple

.PHONY: clean

all: $(TARGETS)

$(TARGETS): %: %.cu
	$(CC) $(CCOPTS) -o $@ $<

clean:
	rm -f $(TARGETS)
