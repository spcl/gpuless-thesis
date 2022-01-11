import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import ScalarFormatter
import os
import sys
import re
import pandas as pd
import numpy

color1_s0 = '#ca6565'
color1_s1 = '#ffcccc'
color1_s2 = '#F29F9F'
color1_s3 = '#A43A3A'
color1_s4 = '#7C1717'

color2_s0 = '#3d7979'
color2_s1 = '#94b9b9'
color2_s2 = '#5F9191'
color2_s3 = '#236262'
color2_s4 = '#0E4A4A'

color3_s0 = '#9DBC5E'
color3_s1 = '#E3F3C2'
color3_s2 = '#C8E294'
color3_s3 = '#789936'
color3_s4 = '#547416'

def read_bench(v, fname, name):
    f = open(fname)
    lines = [l.rstrip().split(' ') for l in f.readlines()]
    first_time = float(lines[0][1])
    for line in lines:
        t = float(line[1]) - first_time
        s = float(line[0]) / 1024 / 1024
        # print(t)
        # print(s)
        v.append([name, s, t])
    f.close()

def plot(name):
    v = []
    cols = ['benchmark', 'size', 'time']
    read_bench(v, f'data/{name}.out', name)
    bench_data = pd.DataFrame(data=v, columns=cols)

    # histogram
    # sns.set(rc={"figure.figsize": (6.0, 4.5)})
    sns.set_theme(style='whitegrid')
    ax = sns.histplot(x='time', y='size', bins=25, data=bench_data, cbar=True)
    ax.set_ylabel('Data transfer size [MB]')
    ax.set_xlabel('Timestamp [ms]')
    ax.set_title(f'Histogram of synchronization calls ({name})')

    ax.figure.savefig(f'{name}-sync-histogram.pdf')
    plt.clf()

plot('resnet50')
plot('resnext50')
plot('resnext101')
plot('alexnet')
plot('vgg19')
plot('yolop')
plot('3d-unet-kits19')
plot('BERT-SQuAD')
plot('midas')
