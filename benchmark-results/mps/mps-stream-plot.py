import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import re
import pandas as pd
all_lines = []
fnames = [
    './bench-results/test-mps/archive/mps-25-1.out',
    './bench-results/test-mps/archive/mps-25-2.out',
    './bench-results/test-mps/archive/mps-50-1.out',
    './bench-results/test-mps/archive/mps-50-2.out',
    './bench-results/test-mps/archive/mps-75-1.out',
    './bench-results/test-mps/archive/mps-75-2.out',
    './bench-results/test-mps/archive/mps-100-1.out', # mps-100-1.out
    './bench-results/test-mps/archive/mps-100-2.out',
    './bench-results/test-mps/archive/mps-custom-25-1.out',
    './bench-results/test-mps/archive/mps-custom-75-1.out',
    './bench-results/test-mps/archive/mps-single.out'
]

config_map = {
    './bench-results/test-mps/archive/mps-25-1.out': 'mps-25-1',
    './bench-results/test-mps/archive/mps-25-2.out': 'mps-25-2',
    './bench-results/test-mps/archive/mps-50-1.out': 'mps-50-1',
    './bench-results/test-mps/archive/mps-50-2.out': 'mps-50-2',
    './bench-results/test-mps/archive/mps-75-1.out': 'mps-75-1',
    './bench-results/test-mps/archive/mps-75-2.out': 'mps-75-2',
    './bench-results/test-mps/archive/mps-100-1.out': 'mps-100-1',
    './bench-results/test-mps/archive/mps-100-2.out': 'mps-100-2',
    './bench-results/test-mps/archive/mps-custom-25-1.out': 'mps-custom-25',
    './bench-results/test-mps/archive/mps-custom-75-1.out': 'mps-custom-75',
    './bench-results/test-mps/archive/mps-single.out': 'mps-single',
}

def action_to_bw(action, time):
    G = 1 << 30
    N = 1 << 26
    bw = 0.0
    if action == 'triad' or action == 'add':
        bw = 3 * 8 * N / time / G
    else:
        bw = 2 * 8 * N / time / G
    return bw

def process(l, config):
    return config+","+l.rstrip()

for fname in fnames:
    with open(fname, 'r') as f:
        lines = f.readlines()
    # print("4000", lines[4000], "4001",lines[4001], "4002",lines[4002])
    all_lines += [process(x, config_map[fname]) for x in lines[1:4001]]
    # break


def config_to_order(config):
    c = config
    if c == "mps-25-1":
        return 25 * 10 + 0
    if c == "mps-25-2":
        return 25 * 10 + 10
    if c == "mps-single":
        return 100
    if c == "mps-50-1":
        return 50 * 10 + 1
    if c == "mps-50-2":
        return 50 * 10 + 10
    if c == "mps-75-1":
        return 75 * 10 + 2
    if c == "mps-75-2":
        return 75 * 10 + 20
    if c == 'mps-100-1':
        return 100 * 10 + 3
    if c == 'mps-100-2':
        return 100 * 10 + 30
    if c == 'mps-custom-25':
        return 250 * 10 + 3
    if c == 'mps-custom-75':
        return 750 * 10 + 3

def bw_plot(f, data, title):
    sns.set(rc={"figure.figsize": (10.0, 6.0)})
    sns.set_theme(style='whitegrid')
    ax = sns.barplot(x='bw',
                     y='config',
                     hue='action',
                     orient='h',
                     ci=95,
                     data=data)
    ax.set_xlabel('Bandwidth [Gb/s]')
    ax.set_ylabel('Partition', rotation='horizontal')
    ax.yaxis.set_label_coords(-0.06, 1.01)
    ax.legend(loc='upper right')
    ax.set_title(title)
    ax.figure.savefig(f'{f}.pdf')
    plt.clf()

cols = ['config', 'action', 'time', 'bw', 'order']

df_m_isolated = pd.DataFrame(columns=cols)

for i in range(0, len(all_lines)):
    vs = all_lines[i].split(',')
    t = float(vs[2])
    vss = [[
        vs[0],
        vs[1],
        t,
        action_to_bw(vs[0], t),
        config_to_order(vs[0])
        ]]
    df_m_isolated = pd.concat([df_m_isolated, pd.DataFrame(vss, columns=cols)], ignore_index=True)

bw_plot('mps', df_m_isolated, 'mps under concurrent load (STREAM benchmark)\nNVIDIA V100, n=1000, size=2^26, 95% confidence')