#!/usr/bin/env bash

ip='127.0.0.1'
root=${1:-$HOME/gpuless}
benchmarks=$root/src/benchmarks

BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_colour () {
	echo -e "${BLUE}$1${NC}"
}
echo
echo 'remote TCP execution'
echo

echo_colour 'latency'
./benchmark.sh $root $benchmarks/latency remote latency $ip remote

echo_colour 'hotspot'
./benchmark.sh $root $benchmarks/hotspot remote hotspot $ip remote

echo_colour 'srad_v1'
./benchmark.sh $root $benchmarks/srad_v1 remote srad $ip remote

echo_colour 'pathfinder'
./benchmark.sh $root $benchmarks/pathfinder remote pathfinder $ip remote

echo_colour 'bfs'
./benchmark.sh $root $benchmarks/bfs remote bfs $ip remote

echo_colour 'resnet50-py'
./benchmark.sh $root $benchmarks/resnet50-py remote ~/libtorch/lib/libtorch_cuda.so $ip remote

echo_colour 'resnet50'
./benchmark.sh $root $benchmarks/resnet50 remote ~/libtorch/lib/libtorch_cuda.so $ip remote

echo_colour '3d-unet-kits19'
./benchmark.sh $root $benchmarks/3d-unet-kits19 remote ~/libtorch/lib/libtorch_cuda.so $ip remote

echo_colour 'vgg19'
./benchmark.sh $root $benchmarks/vgg19 remote ~/libtorch/lib/libtorch_cuda.so $ip remote

echo_colour 'yolop'
./benchmark.sh $root $benchmarks/yolop remote ~/libtorch/lib/libtorch_cuda.so $ip remote

echo_colour 'resnext50'
./benchmark.sh $root $benchmarks/resnext50 remote ~/libtorch/lib/libtorch_cuda.so $ip remote

echo_colour 'resnext101'
./benchmark.sh $root $benchmarks/resnext101 remote ~/libtorch/lib/libtorch_cuda.so $ip remote

echo_colour 'alexnet'
./benchmark.sh $root $benchmarks/alexnet remote ~/libtorch/lib/libtorch_cuda.so $ip remote

echo_colour 'midas'
./benchmark.sh $root $benchmarks/midas remote ~/libtorch/lib/libtorch_cuda.so $ip remote

echo_colour 'myocyte'
./benchmark.sh $root $benchmarks/myocyte remote myocyte $ip remote
