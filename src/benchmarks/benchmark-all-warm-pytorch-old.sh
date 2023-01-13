#!/usr/bin/env bash

ip='148.187.105.35'
root=${1:-$HOME/gpuless}
benchmarks=$root/src/benchmarks
torch=$HOME/libtorch/lib/libtorch_cuda.so
bench_type=remote
note="$bench_type-v100-calls-a100-warm-pytorch-old"

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

bench_torch="midas
alexnet
resnext50
resnext101
3d-unet-kits19
vgg19
resnet50-py"
#yolop

#rm ~/.cache/libgpuless -r

for b in $bench
do
	echo_colour $b
	./benchmark-warm-pytorch-old.sh $root $benchmarks/$b $bench_type $b $ip $note
done

for b in $bench_torch
do
	echo_colour $b
	./benchmark-warm-pytorch-old.sh $root $benchmarks/$b $bench_type $torch $ip $note
done
