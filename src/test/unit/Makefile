CC=nvcc
CCOPTS=-cudart shared \
	   -arch=sm_90 \
	   -lcudnn\

TARGETS=manykernel \
		objkernel \
		simple \
		globals \
		async \
		cudnn \

.PHONY: clean

all: $(TARGETS) manymodule

$(TARGETS): %: %.cu
	$(CC) $(CCOPTS) -o $@ $<

manymodule: manymodule.cu mod1.cu mod2.cu
	$(CC) $(CCOPTS) -o $@ $^

clean:
	rm -f $(TARGETS) manymodule
