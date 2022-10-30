#!/usr/bin/env bash

ltrace -x 'cu*' -f -o out.txt "$@"
grep -oP '(cu|__cu).+?(?=\@)' out.txt > out_cut.txt
cat out_cut.txt ../../trace/api ../../trace/api | sort | uniq --unique
rm out.txt out_cut.txt