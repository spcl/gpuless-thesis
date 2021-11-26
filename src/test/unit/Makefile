CC=nvcc
CCOPTS=-cudart shared \
	   -arch=all-major \

TARGETS=manykernel \
		objkernel \
		simple \
		globals \
		async \

.PHONY: clean

all: $(TARGETS) manymodule

$(TARGETS): %: %.cu
	$(CC) $(CCOPTS) -o $@ $<

manymodule: manymodule.cu mod1.cu mod2.cu
	$(CC) $(CCOPTS) -o $@ $^

clean:
	rm -f $(TARGETS) manymodule
