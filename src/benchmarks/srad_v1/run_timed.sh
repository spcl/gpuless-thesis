#!/bin/sh
./srad_v1 100 0.5 502 458 | tail -n2 | cut -d' ' -f1
