CC=nvcc
ODIR=build
CFLAGS=-arch=compute_80 \
	   -code=sm_80,sm_86 \
	   -fatbin \
	   -odir=$(ODIR)
FBIN=$(ODIR)/common.fatbin

all: $(FBIN)

$(ODIR):
	@mkdir -p $(ODIR)

$(ODIR)/%.fatbin: %.cu $(ODIR)
	$(CC) $(CFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(ODIR)/*.fatbin $(ODIR)/*.cubin
