CC=nvcc
# CCOPTS=-cudart shared -arch=sm_86
CCOPTS=-cudart shared \
	   -gencode arch=compute_86,code=sm_86 \
	   -gencode arch=compute_70,code=sm_70

TARGETS=manykernel \
		objkernel \
		simple \
		globals \

.PHONY: clean

all: $(TARGETS) manymodule

$(TARGETS): %: %.cu
	$(CC) $(CCOPTS) -o $@ $<

manymodule: manymodule.cu mod1.cu mod2.cu
	$(CC) $(CCOPTS) -o $@ $^

clean:
	rm -f $(TARGETS) manymodule
