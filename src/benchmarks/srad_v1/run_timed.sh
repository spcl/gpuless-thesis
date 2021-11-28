#!/bin/sh
./srad 100 0.5 502 458 | tail -n1 | cut -d' ' -f1
