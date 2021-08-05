CC=nvcc
ODIR=build
CFLAGS=-arch=compute_80 \
	   -code=compute_80 \
	   --fatbin \
	   -odir=$(ODIR)
FBIN=$(ODIR)/vadd.fatbin

all: fatbin
fatbin: $(FBIN)

$(ODIR):
	@mkdir -p $(ODIR)

$(ODIR)/%.fatbin: %.cu $(ODIR)
	$(CC) $(CFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(ODIR)/*.fatbin
