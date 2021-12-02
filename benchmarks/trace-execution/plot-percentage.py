import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import sys
import re
import pandas as pd
import numpy

benchmarks = ['latency', 'srad_v1', 'hotspot', 'resnet50']
cols = ['benchmark', 'type', 'perc_diff_to_native']

v = []
for b in benchmarks:
    f_native = open(b + '/native.out')
    f_remote = open(b + '/remote.out')
    f_local = open(b + '/local.out')

    l_native = [l.rstrip() for l in f_native.readlines()]
    l_remote = [l.rstrip() for l in f_remote.readlines()]
    l_local = [l.rstrip() for l in f_local.readlines()]

    native_all = []
    for l in l_native:
        native_all.append(float(l))
    native_median = numpy.median(native_all)

    for l in l_remote:
        perc_diff = (float(l) - native_median) / native_median * 100.0
        v.append([b, 'remote-tcp', perc_diff])

    for l in l_local:
        perc_diff = (float(l) - native_median) / native_median * 100.0
        v.append([b, 'local-tcp', perc_diff])

    f_native.close()
    f_remote.close()
    f_local.close()

bench_data = pd.DataFrame(data=v, columns=cols)

sns.set(rc={"figure.figsize": (8.5, 6.5)})
sns.set_theme(style='whitegrid')

hue_order=['local-tcp', 'remote-tcp']
ax = sns.barplot(x='benchmark',
                 y='perc_diff_to_native',
                 hue='type',
                 hue_order=hue_order,
                 orient='v',
                 data=bench_data,
                 ci=95)
ax.set_ylabel('Difference', rotation='horizontal')
ax.set_xlabel('Benchmark')
ax.yaxis.set_label_coords(-0.06, 1.02)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='upper right')
ax.set_title('Traced remote execution time vs. median native performance\nNVIDIA A100, n=100, 95% confidence')
ax.figure.savefig('trace-bench.pdf')
