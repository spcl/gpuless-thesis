import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import sys
import re
import pandas as pd
import numpy
import re

benchmark_total_median = {}
benchmark_sync_median = {}

include = [
        # 'latency',
        # 'hotspot',
        # 'srad_v1',
        # 'bfs',
        # 'pathfinder',
        'resnet50',
        'vgg19',
        '3d-unet-kits19',
        'BERT-SQuAD'
        ]


for fname in os.listdir('data/total'):
    r = r'benchmark-([a-zA-Z0-9_-]*)-remote-remote-.*'
    m = re.search(r, fname)
    if m and m.group(1) in include:
        f = open('data/total/' + fname)
        lines_float = [float(l.rstrip()) for l in f.readlines()]
        benchmark_total_median[m.group(1)] = numpy.median(lines_float)
        f.close()

for fname in os.listdir('data/synchronize'):
    r = r'benchmark-([a-zA-Z0-9_-]*)-remote-remote-.*'
    m = re.search(r, fname)
    if m and m.group(1) in include:
        f = open('data/synchronize/' + fname)
        lines_float = [float(l.rstrip()) for l in f.readlines()]
        benchmark_sync_median[m.group(1)] = numpy.median(lines_float)
        f.close()

rows = []
for k, v in benchmark_sync_median.items():
    total = benchmark_total_median[k]
    rows.append([k, float(total) - float(v), float(v)])
cols = ['benchmark', 'time_rest', 'time_sync']
df = pd.DataFrame(data=rows, columns=cols)

sns.set(rc={"figure.figsize": (8.0, 6.5)})
sns.set_theme(style='whitegrid')

fig, ax = plt.subplots()
df.set_index('benchmark').plot(kind='bar', stacked=True, color=['darkblue', 'lightblue'], ax=ax)

ax.legend(['Host time', 'Synchronization time'])
ax.set_ylabel('Time [s]')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=0)
ax.set_title('Execution time components of ML inference benchmarks\nNVIDIA A100, n=20, median execution time')

fig.savefig('trace-time-components.pdf')
