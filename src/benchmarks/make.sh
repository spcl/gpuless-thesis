bench_dirs="bfs
dwt2d
gaussian
hotspot
latency
myocyte
pathfinder
srad_v1"

BLUE='\033[0;34m'
NC='\033[0m' # No Color

compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)

CUDA_ARCH="compute_${compute_cap:0:1}${compute_cap:2:1}"

for dir in $bench_dirs
do
	echo -e "${BLUE}$dir${NC}"
	cd $dir
	make clean
	make CUDA_ARCH=$CUDA_ARCH CUDA_DIR=$CUDA_HOME -j8
	cd ..
done

bench_dirs_cmake="resnet50"

for dir in $bench_dirs_cmake
do
	echo -e "${BLUE}$dir${NC}"
	cd $dir
	rm -r build
	mkdir build
	cd build
	cmake ..
	make -j8
	cd ../..
done
