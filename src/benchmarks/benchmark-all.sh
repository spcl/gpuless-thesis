#!/usr/bin/env bash

ip='148.187.105.35'
root=${1:-$HOME/msc-gpuless-lutobler}
benchmarks=$root/src/benchmarks

echo
echo 'native execution'
echo

echo 'latency'
./benchmark.sh $root $benchmarks/latency native latency $ip local

echo 'hotspot'
./benchmark.sh $root $benchmarks/hotspot native hotspot $ip local

echo 'srad_v1'
./benchmark.sh $root $benchmarks/srad_v1 native srad $ip local

echo 'pathfinder'
./benchmark.sh $root $benchmarks/pathfinder native pathfinder $ip local

echo 'bfs'
./benchmark.sh $root $benchmarks/bfs native bfs $ip local

echo 'resnet50'
./benchmark.sh $root $benchmarks/resnet50 native ~/libtorch/lib/libtorch_cuda.so $ip local

echo '3d-unet-kits19'
./benchmark.sh $root $benchmarks/3d-unet-kits19 native ~/libtorch/lib/libtorch_cuda.so $ip local

echo 'vgg19'
./benchmark.sh $root $benchmarks/vgg19 native ~/libtorch/lib/libtorch_cuda.so $ip local

echo 'resnext50'
./benchmark.sh $root $benchmarks/resnext50 native ~/libtorch/lib/libtorch_cuda.so $ip local

echo 'resnext101'
./benchmark.sh $root $benchmarks/resnext101 native ~/libtorch/lib/libtorch_cuda.so $ip local

echo 'alexnet'
./benchmark.sh $root $benchmarks/alexnet native ~/libtorch/lib/libtorch_cuda.so $ip local

echo 'midas'
./benchmark.sh $root $benchmarks/midas native ~/libtorch/lib/libtorch_cuda.so $ip local

echo
echo 'local TCP execution'
echo

echo 'latency'
./benchmark.sh $root $benchmarks/latency remote latency $ip local

echo 'hotspot'
./benchmark.sh $root $benchmarks/hotspot remote hotspot $ip local

echo 'srad_v1'
./benchmark.sh $root $benchmarks/srad_v1 remote srad $ip local

echo 'pathfinder'
./benchmark.sh $root $benchmarks/pathfinder remote pathfinder $ip local

echo 'bfs'
./benchmark.sh $root $benchmarks/bfs remote bfs $ip local

echo 'resnet50'
./benchmark.sh $root $benchmarks/resnet50 remote ~/libtorch/lib/libtorch_cuda.so $ip local

echo '3d-unet-kits19'
./benchmark.sh $root $benchmarks/3d-unet-kits19 remote ~/libtorch/lib/libtorch_cuda.so $ip local

echo 'vgg19'
./benchmark.sh $root $benchmarks/vgg19 remote ~/libtorch/lib/libtorch_cuda.so $ip local

echo 'resnext50'
./benchmark.sh $root $benchmarks/resnext50 remote ~/libtorch/lib/libtorch_cuda.so $ip local

echo 'resnext101'
./benchmark.sh $root $benchmarks/resnext101 remote ~/libtorch/lib/libtorch_cuda.so $ip local

echo 'alexnet'
./benchmark.sh $root $benchmarks/alexnet remote ~/libtorch/lib/libtorch_cuda.so $ip local

echo 'midas'
./benchmark.sh $root $benchmarks/midas remote ~/libtorch/lib/libtorch_cuda.so $ip local
