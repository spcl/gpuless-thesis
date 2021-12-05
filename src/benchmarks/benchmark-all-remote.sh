#!/usr/bin/env bash

IP=148.187.105.35

echo 'remote TCP execution'
echo

echo 'latency'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/latency \
	remote latency $IP remote

echo 'hotspot'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/hotspot \
	remote hotspot $IP remote

echo 'srad_v1'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/srad_v1 \
	remote srad $IP remote

echo 'pathfinder'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/pathfinder \
	remote pathfinder $IP remote

echo 'bfs'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/bfs \
	remote bfs $IP remote

echo 'resnet50'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/resnet50 \
	remote ~/libtorch/lib/libtorch_cuda.so $IP remote

echo '3d-unet-kits19'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/3d-unet-kits19 \
	remote ~/libtorch/lib/libtorch_cuda.so $IP remote

