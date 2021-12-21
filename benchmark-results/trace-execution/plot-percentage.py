import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import sys
import re
import pandas as pd
import numpy
import re

bechmark_native_median = {}
for fname in os.listdir('data/total'):
    r = r'benchmark-([a-zA-Z0-9_-]*)-native-([a-zA-Z0-9_]*)-.*'
    m = re.search(r, fname)
    if (m):
        f = open('data/total/' + fname)
        lines_float = [float(l.rstrip()) for l in f.readlines()]
        bechmark_native_median[m.group(1)] = numpy.median(lines_float)
        f.close()

v = []
for fname in os.listdir('data/total'):
    f = open('data/total/' + fname)
    lines = [l.rstrip() for l in f.readlines()]

    r = r'benchmark-([a-zA-Z0-9_-]*)-(native|remote)-(remote|local)-.*'
    m = re.search(r, fname)
    benchmark_name = m.group(1)
    benchmark_type = m.group(2)
    benchmark_flag = m.group(3)

    if benchmark_type == 'remote' and benchmark_flag == 'local':
        native_median = bechmark_native_median[benchmark_name]
        for l in lines:
            perc_diff = (float(l) - native_median) / native_median * 100.0
            v.append([benchmark_name, 'Local TCP', perc_diff])
    elif benchmark_type == 'remote' and benchmark_flag == 'remote':
        native_median = bechmark_native_median[benchmark_name]
        for l in lines:
            perc_diff = (float(l) - native_median) / native_median * 100.0
            v.append([benchmark_name, 'Remote TCP', perc_diff])

    f.close()

cols = ['benchmark', 'type', 'perc_diff_to_native']
bench_data = pd.DataFrame(data=v, columns=cols)

hue_order = ['Local TCP', 'Remote TCP']
order = [
        'latency',
        'hotspot',
        'srad_v1',
        'bfs',
        'pathfinder',
        'resnet50',
        'vgg19',
        '3d-unet-kits19',
        'BERT-SQuAD'
        ]

sns.set(rc={"figure.figsize": (8.5, 7.5)})
sns.set_theme(style='whitegrid')
palette = sns.color_palette("muted")
ax = sns.barplot(x='benchmark',
                 y='perc_diff_to_native',
                 hue='type',
                 hue_order=hue_order,
                 order=order,
                 orient='v',
                 data=bench_data,
                 palette=palette,
                 ci=95)
# ax.set_ylabel('Difference \nto native', rotation='horizontal')
ax.set_ylabel('Difference to native')
# ax.set_xlabel('Benchmark')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=25)
# ax.yaxis.set_label_coords(-0.06, 1.02)
# ax.xaxis.set_label_coords(0.5, -0.09)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='upper left')
ax.set_title('Traced remote execution time vs. median native performance\nNVIDIA A100, n=100, 95% confidence')
ax.figure.savefig('trace-percentage.pdf')
