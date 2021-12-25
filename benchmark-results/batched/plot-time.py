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
cols = ['benchmark', 'type', 'time']

def read_bench(fname, name, type_):
    f = open(fname)
    lines = [l.rstrip() for l in f.readlines()]

    for line in lines:
        v.append([name, type_, float(line) * 1000.0])

    f.close()

read_bench('data/image-recognition-batched.native.out', 'resnet50', 'Native')
read_bench('data/image-recognition-batched.local.out', 'resnet50', 'Local TCP')
read_bench('data/image-recognition-batched.remote.out', 'resnet50', 'Remote TCP')
read_bench('data/BERT-SQuAD.native.out', 'BERT-SQuaD', 'Native')
read_bench('data/BERT-SQuAD.local.out', 'BERT-SQuaD', 'Local TCP')
read_bench('data/BERT-SQuAD.remote.out', 'BERT-SQuaD', 'Remote TCP')

bench_data = pd.DataFrame(data=v, columns=cols)

sns.set(rc={"figure.figsize": (7.5, 5.0)})
sns.set_theme(style='whitegrid')
palette = sns.color_palette("muted")

hue_order = ['Native', 'Local TCP', 'Remote TCP']
# order = ['resnet50', 'BERT-SQuAD']

# violin plot
ax = sns.violinplot(data=bench_data,
               x='time',
               y='benchmark',
               palette=palette,
               hue='type')
ax.set_ylabel('Benchmark', rotation='horizontal')
ax.set_xlabel('Time [ms]')
ax.yaxis.set_label_coords(-0.06, 1.02)
ax.legend(loc='lower right')
ax.set_title('Runtime of model inference in hot environment\nNVIDIA A100, n=100')
ax.figure.savefig('hot-inference-violin.pdf')
plt.clf()

# barplot
ax = sns.barplot(y='time',
                 x='benchmark',
                 hue='type',
                 hue_order=hue_order,
                 # order=order,
                 # orient='h',
                 data=bench_data,
                 ci=95,
                 palette=palette)
ax.set_ylabel('Time [ms]')
# ax.set_ylabel('Time [ms]', rotation='horizontal')
ax.set_xlabel('Benchmark')
# ax.yaxis.set_label_coords(-0.06, 1.02)
ax.legend(loc='upper left')
ax.set_title('Runtime of model inference in hot environment\nNVIDIA A100, n=100, 95% confidence')
ax.figure.savefig('hot-inference.pdf')

