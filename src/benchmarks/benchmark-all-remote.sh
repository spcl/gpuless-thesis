#!/usr/bin/env bash

ip='148.187.105.35'
root=${1:-$HOME/msc-lutobler-gpuless}
benchmarks=$root/src/benchmarks

echo
echo 'remote TCP execution'
echo

echo 'latency'
./benchmark.sh $root $benchmarks/latency remote latency $ip remote

echo 'hotspot'
./benchmark.sh $root $benchmarks/hotspot remote hotspot $ip remote

echo 'srad_v1'
./benchmark.sh $root $benchmarks/srad_v1 remote srad $ip remote

echo 'pathfinder'
./benchmark.sh $root $benchmarks/pathfinder remote pathfinder $ip remote

echo 'bfs'
./benchmark.sh $root $benchmarks/bfs remote bfs $ip remote

echo 'resnet50'
./benchmark.sh $root $benchmarks/resnet50 remote ~/libtorch/lib/libtorch_cuda.so $ip remote

echo '3d-unet-kits19'
./benchmark.sh $root $benchmarks/3d-unet-kits19 remote ~/libtorch/lib/libtorch_cuda.so $ip remote

echo 'vgg19'
./benchmark.sh $root $benchmarks/vgg19 remote ~/libtorch/lib/libtorch_cuda.so $ip remote

echo 'resnext50'
./benchmark.sh $root $benchmarks/resnext50 remote ~/libtorch/lib/libtorch_cuda.so $ip remote

echo 'resnext101'
./benchmark.sh $root $benchmarks/resnext101 remote ~/libtorch/lib/libtorch_cuda.so $ip remote

echo 'alexnet'
./benchmark.sh $root $benchmarks/alexnet remote ~/libtorch/lib/libtorch_cuda.so $ip remote

echo 'midas'
./benchmark.sh $root $benchmarks/midas remote ~/libtorch/lib/libtorch_cuda.so $ip remote
