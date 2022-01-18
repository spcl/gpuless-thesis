#!/usr/bin/env bash

exe="$1"

pushd .
cd $exe
while true; do
    python app.py
done
popd
