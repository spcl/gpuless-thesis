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
    lines = [int(l.rstrip()) for l in f.readlines()]
    for line in lines:
        v.append([name, line])
    f.close()

def plot_hist(name):
    v = []
    cols = ['benchmark', 'n']
    read_bench(v, f'data/{name}.out', name)
    bench_data = pd.DataFrame(data=v, columns=cols)

    # histogram
    # sns.set(rc={"figure.figsize": (6.0, 4.5)})
    sns.set_theme(style='whitegrid')
    ax = sns.histplot(x='n', bins=30, data=bench_data)
    # ax.set_ylabel('Data transfer size [MB]')
    # ax.set_xlabel('Timestamp [ms]')
    ax.set_title(f'Histogram of synchronization trace size ({name})')

    ax.figure.savefig(f'{name}-trace-histogram.pdf')
    plt.clf()

def plot_ecdf(name):
    v = []
    cols = ['benchmark', 'n']
    read_bench(v, f'data/{name}.out', name)
    bench_data = pd.DataFrame(data=v, columns=cols)

    # histogram
    # sns.set(rc={"figure.figsize": (6.0, 4.5)})
    sns.set_theme(style='whitegrid')
    ax = sns.ecdfplot(x='n', data=bench_data)
    # ax.set_ylabel('Data transfer size [MB]')
    ax.set_xlabel('Trace size')
    ax.set_title(f'ECDF of trace size at synchronization ({name})')

    ax.figure.savefig(f'{name}-trace-ecdf.pdf')
    plt.clf()

plot_hist('resnet50')
plot_hist('resnext50')
plot_hist('resnext101')
plot_hist('alexnet')
plot_hist('vgg19')
plot_hist('yolop')
plot_hist('3d-unet-kits19')
plot_hist('BERT-SQuAD')
plot_hist('midas')

plot_ecdf('resnet50')
plot_ecdf('resnext50')
plot_ecdf('resnext101')
plot_ecdf('alexnet')
plot_ecdf('vgg19')
plot_ecdf('yolop')
plot_ecdf('3d-unet-kits19')
plot_ecdf('BERT-SQuAD')
plot_ecdf('midas')
