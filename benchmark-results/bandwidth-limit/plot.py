import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import ScalarFormatter
import os
import sys
import re
import pandas as pd
import numpy


def read_bench(fname, name, bw):
    with open(fname, 'r') as f:
        total_times = [l.rstrip().split(' ')[0] for l in f.readlines()]
        for t in total_times:
            v.append([name, float(t), bw])

def read_bench_mult1000(fname, name, bw):
    with open(fname, 'r') as f:
        total_times = [l.rstrip().split(' ')[0] for l in f.readlines()]
        for t in total_times:
            v.append([name, float(t)*1000.0, bw])

benchmarks = [
        'midas',
        'resnet50',
        'resnext101',
        'yolop',
        '3d-unet-kits19',
        'BERT-SQuAD',
        ]

bwlimits = [
        100000000,
        75000000,
        50000000,
        25000000,
        10000000,
        7500000,
        5000000,
        2500000,
        1000000,
        750000,
        500000,
        250000,
        100000,
        ]

v = []
cols = ['Benchmark', 'time', 'bandwidth']

# cold
for b in benchmarks:
    v.clear()
    for limit in bwlimits:
        fname = f'data/cold/bwlimit-{b}-{str(limit)}.out'
        with open(fname, 'r') as f:
            total_times = [l.rstrip().split(' ')[0] for l in f.readlines()]
            read_bench(fname, b, limit / 1000000.0)
    bench_data = pd.DataFrame(data=v, columns=cols)

    # read native median for horizontal line
    fname_native = f'data/cold/native/benchmark-{b}-native.out'
    with open(fname_native, 'r') as f_native:
        vals = [float(line.rstrip()) for line in f_native.readlines()]
        med = numpy.median(vals)

    sns.set(rc={"figure.figsize": (6.0, 4.5)})
    sns.set_style("ticks", {'axes.grid' : True})
    palette = sns.color_palette("muted")

    ax = sns.lineplot(data=bench_data,
                      x='bandwidth',
                      y='time',
                      ci=95)

    ax.axhline(y=med, xmin=0, xmax=1, linestyle='--', color='orangered')

    ax.set_xscale('log')
    ax.set_ylim([0.0, 100.0])
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.set_xlabel('Network bandwidth [Gbps]')
    ax.set_ylabel('Time [s]')
    ax.set_title(f'{b}')

    r0 = plt.Rectangle((0, 0), 1, 1, fc='blue', edgecolor='none')
    ax.legend(
            handles=[ax.lines[0], ax.lines[1]],
            labels=['GPUless remote execution', 'Native baseline'],
            title='')

    ax.figure.savefig(f'bwlimit-cold-{b}.pdf')
    plt.clf()

# hot
for b in benchmarks:
    v.clear()
    for limit in bwlimits:
        fname = f'data/hot/bwlimit-{b}-{str(limit)}.out'
        with open(fname, 'r') as f:
            total_times = [l.rstrip().split(' ')[0] for l in f.readlines()]
            if b == '3d-unet-kits19':
                read_bench(fname, b, limit / 1000000.0)
            else:
                read_bench_mult1000(fname, b, limit / 1000000.0)
    bench_data = pd.DataFrame(data=v, columns=cols)

    # read native median for horizontal line
    fname_native = f'data/hot/native/benchmark-{b}-native.out'

    if b == '3d-unet-kits19':
        with open(fname_native, 'r') as f_native:
            vals = [float(line.rstrip()) for line in f_native.readlines()]
            med = numpy.median(vals)
    else:
        with open(fname_native, 'r') as f_native:
            vals = [float(line.rstrip())*1000.0 for line in f_native.readlines()]
            med = numpy.median(vals)

    sns.set(rc={"figure.figsize": (6.0, 4.5)})
    sns.set_style("ticks", {'axes.grid' : True})
    palette = sns.color_palette("muted")

    ax = sns.lineplot(data=bench_data,
                      x='bandwidth',
                      y='time',
                      ci=95)

    ax.axhline(y=med, xmin=0, xmax=1, linestyle='--', color='orangered')

    ax.set_xscale('log')
    # ax.set_ylim([0.0, 100.0])
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.set_xlabel('Network bandwidth [Gbps]')
    if b == '3d-unet-kits19':
        ax.set_ylabel('Time [s]')
    else:
        ax.set_ylabel('Time [ms]')
    ax.set_title(f'{b}')

    r0 = plt.Rectangle((0, 0), 1, 1, fc='blue', edgecolor='none')
    ax.legend(
            handles=[ax.lines[0], ax.lines[1]],
            labels=['GPUless remote execution', 'Native baseline'],
            title='')

    ax.figure.savefig(f'bwlimit-hot-{b}.pdf')
    plt.clf()


# # grid plot

# # sns.set_style("ticks", {})
# grid = sns.FacetGrid(data=bench_data, col='Benchmark', col_wrap=2)
# grid.map(sns.lineplot,
#         'bandwidth',
#         'time')

# grid.set_titles(col_template='{col_name}', row_template='')
# grid.set_ylabels('Time [s]')
# grid.set_xlabels('Bandwidth [Gbps]')

# for ax in grid.axes.flat:
#     ax.tick_params(labelbottom=True)
#     ax.set_xscale('log')
#     ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
#     # ax.set_xlabel('Network bandwidth [Gbps]')
#     # ax.set_ylabel('Time [s]')
#     # ax.set_title('Cold-start execution time on bandwidth-limited loopback interface\nNVIDIA RTX 3060, Intel i7-9700KF, n=10')


# # ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
# grid.figure.savefig('bwlimit-all.pdf')

# all in one plot

# # sns.set(rc={"figure.figsize": (9.5, 9.0)})
# # sns.set_theme(style='whitegrid')
# sns.set_style("ticks", {'axes.grid' : True})
# palette = sns.color_palette("muted")

# # ax = sns.catplot(data=bench_data,
# #                   x='bandwidth',
# #                   y='time',
# #                   height=8.0,
# #                   palette=palette,
# #                   hue='Benchmark',
# #                   kind='point')
# ax = sns.lineplot(data=bench_data,
#                   x='bandwidth',
#                   y='time',
#                   hue='Benchmark')
# ax.set_xscale('log')
# ax.xaxis.set_major_formatter(mtick.ScalarFormatter())

# ax.set_xlabel('Network bandwidth [Gbps]')
# ax.set_ylabel('Time [s]')

# ax.set_title('Cold-start execution time on bandwidth-limited loopback interface\nNVIDIA RTX 3060, Intel i7-9700KF, n=10')
# ax.figure.savefig('bwlimit.pdf')
