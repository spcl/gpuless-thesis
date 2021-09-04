objects = allocation_protocol_generated.h execution_protocol_generated.h

all: $(objects)

%_generated.h: %.fbs
	flatc --cpp $<

clean:
	rm $(objects)
