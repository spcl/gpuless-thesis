# MSC Thesis Project

## Building

```
mkdir src/build
cd src/build
cmake ..
make
```

## `libcudaanalysis`

`libcudaanalysis` can be used to trace CUDA calls for an application. Trace
output will be written to stderr.

Example usage:

```
CUDA_BINARY=/usr/lib/libtorch_cuda.so,/usr/lib/libcudnn_ops_infer.so.8.2.4 \
LD_PRELOAD=/home/luke/ethz/master/thesis/msc-lutobler-gpuless/src/build/libanalysis.so \
./image-recognition
```

