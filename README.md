# MSC Thesis Project

## Building

```
mkdir src/build
cd src/build
cmake ..
make
```


The log output level is set at compile time.
Use the following to set the log level:

```
cmake .. -DSPDLOG_LEVEL_OFF=ON
cmake .. -DSPDLOG_LEVEL_INFO=ON
cmake .. -DSPDLOG_LEVEL_DEBUG=ON
```

Remove `CMakeCache.txt` before changing the log level.

## `libgpuless.so`



## `libcudaanalysis.so`

`libcudaanalysis` can be used to trace CUDA calls for an application. Trace
output will be written to stderr.

Example usage:

```
CUDA_BINARY=/usr/lib/libtorch_cuda.so,/usr/lib/libcudnn_ops_infer.so.8.2.4 \
LD_PRELOAD=/home/luke/ethz/master/thesis/msc-lutobler-gpuless/src/build/libanalysis.so \
./image-recognition
```

