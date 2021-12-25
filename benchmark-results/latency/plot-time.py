import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import sys
import re
import pandas as pd
import numpy

v = []
for fname in os.listdir('data'):
    f = open('data/' + fname)
    lines = [l.rstrip() for l in f.readlines()]

    r = r'benchmark-([a-zA-Z0-9_-]*)-(native|remote|preinit)-(remote|local)-.*'
    m = re.search(r, fname)
    benchmark_name = m.group(1)
    benchmark_type = m.group(2)
    benchmark_flag = m.group(3)

    if benchmark_type == 'native':
        for l in lines:
            v.append([benchmark_name, 'Native', float(l) * 1000.0])
    elif benchmark_type == 'remote' and benchmark_flag == 'local':
        for l in lines:
            v.append([benchmark_name, 'Local TCP', float(l) * 1000.0])
    elif benchmark_type == 'remote' and benchmark_flag == 'remote':
        for l in lines:
            v.append([benchmark_name, 'Remote TCP', float(l) * 1000.0])
    elif benchmark_type == 'preinit' and benchmark_flag == 'local':
        for l in lines:
            v.append([benchmark_name, 'Native Pre-Initialized', float(l) * 1000.0])

    f.close()

cols = ['benchmark', 'type', 'time']
bench_data = pd.DataFrame(data=v, columns=cols)

sns.set(rc={"figure.figsize": (8.0, 5.5)})
sns.set_theme(style='whitegrid')
palette = sns.color_palette("muted")

# hue_order=['Native', 'Native Pre-Initialized', 'Local TCP', 'Remote TCP']
hue_order=['Native', 'Local TCP', 'Remote TCP', 'Native Pre-Initialized']
ax = sns.barplot(x='time',
                 y='benchmark',
                 hue='type',
                 hue_order=hue_order,
                 orient='h',
                 data=bench_data,
                 palette=palette,
                 ci=95)
# ax.set_ylabel('Benchmark', rotation='horizontal')
ax.set_ylabel('')
ax.set_xlabel('Time [ms]')
ax.set_xscale('log')
# ax.yaxis.set_label_coords(-0.06, 1.02)
ax.legend(loc='lower right')
ax.set_title('Latency of minimal CUDA API execution\nNVIDIA A100, n=100, 95% confidence')
ax.figure.savefig('latency.pdf')
