#!/usr/bin/env bash

files=(
    hotspot.log
    dwt2d.log
    srad2.log
    resnet50.log
    bert-squad.log
    3d-unet-kits19.log
    srad_v1.log
    bfs.log
    resnext50.log
    resnext101.log
    vgg19.log
    alexnet.log
    midas.log
    yolop.log
    BERT-SQuAD.log
    gaussian.log
    myocyte.log
    pathfinder.log
)

printf '| Application | Kernel launches | Unique Kernels | cudaMalloc | Copy H2D | Copy D2D | Copy D2H |\n'
/usr/bin/printf '| --- | --- | --- | --- | --- | --- | --- |\n'
for f in ${files[@]}; do
    n_kernel=$(grep 'cuLaunchKernel' "$f" | wc -l)
    n_malloc=$(grep 'cudaMalloc' "$f" | wc -l)
    n_cpyd2h=$(grep 'DeviceToHost' "$f" | wc -l)
    n_cpyh2d=$(grep 'HostToDevice' "$f" | wc -l)
    # n_cpyd2d=$(grep 'DeviceToDevice' "$f" | wc -l)
    n_uniq_kernel=$(grep -E 'cuLaunchKernel' "$f" | sed 's/^\(cuLaunchKernel(.*\)\ \[.*$/\1/' | sort | uniq | wc -l)

    # printf "| $f | $n_kernel | $n_uniq_kernel | $n_malloc | $n_cpyh2d | $n_cpyd2d | $n_cpyd2h |\n"
    # printf "| $f | $n_kernel | $n_uniq_kernel | $n_malloc | $n_cpyh2d | $n_cpyd2h |\n"
    printf "$f & $n_kernel & $n_uniq_kernel & $n_malloc & $n_cpyh2d & $n_cpyd2h \\\\\\\\ \n"
done
