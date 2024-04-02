All results are benchmarked on V100.

All benchmarks are compiled from ```projects/gpuless/benchmark-results/mig-isolation/cuda-stream/stream-mps.cu```, it is slightly modified version based on ```stream.cu``` in the same directory. it adds CHECK_LAST_CUDA_ERROR and the kernel function: raiseError that triggers fatal gpu error during kernel execution.

(CUDA_ARCH in ```projects/gpuless/benchmark-results/mig-isolation/cuda-stream/Makefile``` was set to ```compute_70``` to compile on V100.)

before running benchmarks, remember to set ```CUDA_VISIBLE_DEVICES```, ```CUDA_MPS_PIPE_DIRECTORY```, and ```CUDA_MPS_LOG_DIRECTORY```.

# MPS error containment
To run evaluation of MPS error containment:
```bash
nvidia-cuda-mps-control -d
sh run.sh
```
where variable ```t``` will be the name of folder created to hold results. The program used in run.sh is ```mps-stream```.

To plot the results, use:
```bash
mps-error-plot.py
```
results used for plotting are available in ```bench-results/test11/``` for no error runs,
```bench-results/test12/``` for runs with 4 errors, and
```bench-results/test13/``` for runs with 8 errors
# MPS thread percentage
To run evaluation of MPS thread percentage:
```bash
sh run-bench-percentage.sh mps-stream-percentage
```
percentage-helper.sh and custom-percentage-helper.sh are helper files for the script.

the only difference between ```mps-stream-percentage``` and ```mps-stream``` is that ```mps-stream-percentage``` provides the number of SMs used for the program in the output.

To plot the results, use:
```bash
mps-stream-plot.py
```
results used for plotting are available in ```bench-results/test-mps```. <em>1</em> and <em>2</em> in the names of the results means client 1 and client 2 that are running concurrently. <em>custom</em> means non-uniform partiton supported in Volta architecture and later ones.