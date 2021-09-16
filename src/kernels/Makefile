CC=nvcc
ODIR=build
CFLAGS=-arch=compute_80 \
	   -code=sm_80 \
	   -cubin \
	   -odir=$(ODIR)
FBIN=$(ODIR)/common.cubin

all: $(FBIN)

$(ODIR):
	@mkdir -p $(ODIR)

$(ODIR)/%.cubin: %.cu $(ODIR)
	$(CC) $(CFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(ODIR)/*.fatbin $(ODIR)/*.cubin
