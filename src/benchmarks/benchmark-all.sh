#!/usr/bin/env bash

IP=148.187.105.35

echo 'native execution'
echo

echo 'latency'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/latency \
	native latency $IP local

echo 'hotspot'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/hotspot \
	native hotspot $IP local

echo 'srad_v1'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/srad_v1 \
	native srad $IP local

echo 'pathfinder'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/pathfinder \
	native pathfinder $IP local

echo 'bfs'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/bfs \
	native bfs $IP local

echo 'resnet50'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/resnet50 \
	native ~/libtorch/lib/libtorch_cuda.so $IP local

echo '3d-unet-kits19'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/3d-unet-kits19 \
	native ~/libtorch/lib/libtorch_cuda.so $IP local

echo 'local TCP execution'
echo

echo 'latency'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/latency \
	remote latency $IP local

echo 'hotspot'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/hotspot \
	remote hotspot $IP local

echo 'srad_v1'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/srad_v1 \
	remote srad $IP local

echo 'pathfinder'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/pathfinder \
	remote pathfinder $IP local

echo 'bfs'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/bfs \
	remote bfs $IP local

echo 'resnet50'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/resnet50 \
	remote ~/libtorch/lib/libtorch_cuda.so $IP local

echo '3d-unet-kits19'
./benchmark.sh ~/msc-lutobler-gpuless \
	~/msc-lutobler-gpuless/src/benchmarks/3d-unet-kits19 \
	remote ~/libtorch/lib/libtorch_cuda.so $IP local
