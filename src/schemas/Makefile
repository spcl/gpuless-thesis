objects = allocation_protocol_generated.h \
		  execution_protocol_generated.h \
		  cudevice_attributes_generated.h \
		  trace_execution_protocol_generated.h \
		  cuda_runtime_calls_generated.h \

all: $(objects)

%_generated.h: %.fbs
	flatc --cpp $<
	sed -i 's/names\[122\]/names\[128\]/' cudevice_attributes_generated.h # temporary flatbuffer bugfix

clean:
	rm $(objects)
