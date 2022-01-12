#!/usr/bin/env bash

file_to_row() {
    fname=${1}
    benchname=${2}
    syncs=$(cut -d' ' -f1 ${fname})
    calls=$(cut -d' ' -f2 ${fname})
    up=$(cut -d' ' -f3 ${fname})
    up=$(echo "scale=2; ${up}/(1000*1000)" | bc)
    down=$(cut -d' ' -f4 ${fname})
    down=$(echo "scale=3; ${down}/(1000*1000)" | bc)
    perc=$(echo "scale=1; 100.0*${syncs}/${calls}" | bc)
    printf '%s & %s & %s & %s\\%% & %s & %s \\\\ \n' ${benchname} ${calls} ${syncs} ${perc} ${up} ${down}
}

file_to_row 'data/synchronize-stats/number-synchronizations-resnet50.out' 'resnet50'
file_to_row 'data/synchronize-stats/number-synchronizations-resnext50.out' 'resnext50'
file_to_row 'data/synchronize-stats/number-synchronizations-resnext101.out' 'resnext101'
file_to_row 'data/synchronize-stats/number-synchronizations-alexnet.out' 'alexnet'
file_to_row 'data/synchronize-stats/number-synchronizations-vgg19.out' 'vgg19'
file_to_row 'data/synchronize-stats/number-synchronizations-yolop.out' 'yolop'
file_to_row 'data/synchronize-stats/number-synchronizations-MiDaS.out' 'MiDaS'
file_to_row 'data/synchronize-stats/number-synchronizations-3d-unet-kits19.out' '3d-unet-kits19'
file_to_row 'data/synchronize-stats/number-synchronizations-BERT-SQuAD.out' 'BERT-SQuAD'
