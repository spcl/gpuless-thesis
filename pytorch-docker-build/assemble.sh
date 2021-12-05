#!/usr/bin/env bash

out_dir=$1
[[ -d ${out_dir} ]] || mkdir ${out_dir}

cp -r data/include ${out_dir}
cp -r data/lib ${out_dir}
# cp -r data/build/lib ${out_dir}
cp -r data/share ${out_dir}
cp -r data/torch ${out_dir}

zip -r ${out_dir}.zip ${out_dir}
