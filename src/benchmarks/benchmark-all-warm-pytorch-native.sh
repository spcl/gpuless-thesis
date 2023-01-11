#!/usr/bin/env bash

ip='127.0.0.1'
root=${1:-$HOME/gpuless}
benchmarks=$root/src/benchmarks
torch=$HOME/libtorch/lib/libtorch_cuda.so
bench_type=native
note="$bench_type-a100-warm-pytorch"

BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_colour () {
        echo -e "${BLUE}$1${NC}"
}

bench=""
#myocyte
#hotspot
#srad_v1
#pathfinder
#bfs
#myocyte"

bench_torch="alexnet
resnext50
resnext101
3d-unet-kits19
vgg19
yolop
resnet50-py"
#midas

for b in $bench
do
	echo_colour $b
	./benchmark-warm-pytorch.sh $root $benchmarks/$b $bench_type $b $ip $note
done

for b in $bench_torch
do
	echo_colour $b
	./benchmark-warm-pytorch.sh $root $benchmarks/$b $bench_type $torch $ip $note
done
