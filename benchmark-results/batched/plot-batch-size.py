import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import ScalarFormatter
import os
import sys
import re
import pandas as pd
import numpy

v = []
cols = ['benchmark', 'Type', 'time', 'size']

def read_bench(fname, name, type_, size):
    f = open(fname)
    lines = [l.rstrip() for l in f.readlines()]

    for line in lines:
        v.append([name, type_, float(line), size])

    f.close()

read_bench('data/resnet50-incremental-1.out', 'resnet50', 'Native', 1)
read_bench('data/resnet50-incremental-2.out', 'resnet50', 'Native', 2)
read_bench('data/resnet50-incremental-4.out', 'resnet50', 'Native', 4)
read_bench('data/resnet50-incremental-8.out', 'resnet50', 'Native', 8)
read_bench('data/resnet50-incremental-16.out', 'resnet50', 'Native', 16)
read_bench('data/resnet50-incremental-32.out', 'resnet50', 'Native', 32)
read_bench('data/resnet50-incremental-64.out', 'resnet50', 'Native', 64)
read_bench('data/resnet50-incremental-128.out', 'resnet50', 'Native', 128)
read_bench('data/resnet50-incremental-256.out', 'resnet50', 'Native', 256)
read_bench('data/resnet50-incremental-512.out', 'resnet50', 'Native', 512)
read_bench('data/resnet50-incremental-1024.out', 'resnet50', 'Native', 1024)

read_bench('data/resnet50-incremental-remote-1.out', 'resnet50', 'Remote', 1)
read_bench('data/resnet50-incremental-remote-2.out', 'resnet50', 'Remote', 2)
read_bench('data/resnet50-incremental-remote-4.out', 'resnet50', 'Remote', 4)
read_bench('data/resnet50-incremental-remote-8.out', 'resnet50', 'Remote', 8)
read_bench('data/resnet50-incremental-remote-16.out', 'resnet50', 'Remote', 16)
read_bench('data/resnet50-incremental-remote-32.out', 'resnet50', 'Remote', 32)
read_bench('data/resnet50-incremental-remote-64.out', 'resnet50', 'Remote', 64)
read_bench('data/resnet50-incremental-remote-128.out', 'resnet50', 'Remote', 128)
read_bench('data/resnet50-incremental-remote-256.out', 'resnet50', 'Remote', 256)
read_bench('data/resnet50-incremental-remote-512.out', 'resnet50', 'Remote', 512)
read_bench('data/resnet50-incremental-remote-1024.out', 'resnet50', 'Remote', 1024)

bench_data = pd.DataFrame(data=v, columns=cols)

# sns.set(rc={"figure.figsize": (9.5, 9.0)})
sns.set_theme(style='whitegrid')
palette = sns.color_palette("muted")

hue_order = ['Native']

# ax = sns.catplot(data=bench_data,
#                   x='size',
#                   y='time',
#                   height=8.0,
#                   palette=palette,
#                   hue='Type',
#                   kind='point')
ax = sns.lineplot(data=bench_data,
                  x='size',
                  y='time',
                  hue='Type'
                  )
ax.set_xlabel('Batch size')
ax.set_ylabel('Time [s]')

# ax_first = ax.fig.get_axes()[0]
# ax_first.set_xlabel('Batch size')
# ax_first.set_ylabel('Time [s]')

ax.set_title('resnet50 inference with different batch sizes\nNVIDIA A100, n=100, 95% confidence')
ax.figure.savefig('resnet50-batched-sizes.pdf')
