import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import re
import pandas as pd

fname = sys.argv[1]
with open(fname, 'r') as f:
    lines = f.readlines()
lines = [x.rstrip() for x in lines]

config_order = [
        r".*g isolated",
        r"no partition",
        r".*c isolated",
        r"1c\+3c\+3c.*",
        r"1g\+2g\+4g.*",
        r"3g\+3g.*",
        ]

def config_to_order(config):
    c = -1
    for i in range(0, len(config_order)):
        if re.match(config_order[i], config):
            c = i

    if c == 0:
        return int(config[0]) * 10 + 0
    if c == 1:
        return 100;
    if c == 2:
        return int(config[0]) * 10 + 1
    if c == 3:
        return int(config[-3]) * 10 + 2
    if c == 4:
        return int(config[-3]) * 10 + 3
    if c == 5:
        return int(config[-3]) * 10 + 3

    return -1

def action_to_bw(action, time):
    G = 1 << 30
    N = 1 << 27
    bw = 0.0
    if action == 'triad' or action == 'add':
        bw = 3 * 8 * N / time / G
    else:
        bw = 2 * 8 * N / time / G
    return bw

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
    ax.figure.savefig(f'results/{f}.pdf')
    plt.clf()

cols = ['config', 'action', 'time', 'bw', 'order']

data = pd.DataFrame(columns=cols)
df_g_isolated = pd.DataFrame(columns=cols)
df_c_isolated = pd.DataFrame(columns=cols)
df_g_concurrent = pd.DataFrame(columns=cols)
df_c_concurrent = pd.DataFrame(columns=cols)

for i in range(0, len(lines)):
    vs = lines[i].split(',')
    t = float(vs[2])
    vss = [[
        vs[0],
        vs[1],
        t,
        action_to_bw(vs[1], t),
        config_to_order(vs[0])
        ]]
    if bool(re.match(r".*g isolated", vs[0])):
        df_g_isolated = df_g_isolated.append(pd.DataFrame(vss, columns=cols), ignore_index=True)
        if int(vs[0][0]) <= 4:
            df_g_concurrent = df_g_concurrent.append(pd.DataFrame(vss, columns=cols), ignore_index=True)
        if int(vs[0][0]) == 1:
            df_c_isolated = df_c_isolated.append(pd.DataFrame(vss, columns=cols), ignore_index=True)
        if int(vs[0][0]) == 3:
            df_c_isolated = df_c_isolated.append(pd.DataFrame(vss, columns=cols), ignore_index=True)
    elif bool(re.match(r"no partition", vs[0])):
        df_g_isolated = df_g_isolated.append(pd.DataFrame(vss, columns=cols), ignore_index=True)
        df_c_concurrent = df_c_concurrent.append(pd.DataFrame(vss, columns=cols), ignore_index=True)
        df_c_isolated = df_c_isolated.append(pd.DataFrame(vss, columns=cols), ignore_index=True)
    elif bool(re.match(".*c isolated", vs[0])):
        df_c_isolated = df_c_isolated.append(pd.DataFrame(vss, columns=cols), ignore_index=True)
        if int(vs[0][0]) == 1:
            df_c_concurrent = df_c_concurrent.append(pd.DataFrame(vss, columns=cols), ignore_index=True)
        if int(vs[0][0]) == 3:
            df_c_concurrent = df_c_concurrent.append(pd.DataFrame(vss, columns=cols), ignore_index=True)
    elif bool(re.match(".*(.*g)", vs[0])):
        df_g_concurrent = df_g_concurrent.append(pd.DataFrame(vss, columns=cols), ignore_index=True)
    elif bool(re.match(".*(.*c)", vs[0])):
        df_c_concurrent = df_c_concurrent.append(pd.DataFrame(vss, columns=cols), ignore_index=True)

df_g_isolated = df_g_isolated.sort_values('order')
df_c_isolated = df_c_isolated.sort_values('order')
df_g_concurrent = df_g_concurrent.sort_values('order')
df_c_concurrent = df_c_concurrent.sort_values('order')

bw_plot('gi-isolation', df_g_isolated, 'MIG GPU instances in isolation (STREAM benchmark)\nNVIDIA A100, n=1000, size=2^27, 95% confidence')
bw_plot('gi-concurrent', df_g_concurrent, 'MIG GPU instances under concurrent load (STREAM benchmark)\n1g+2g+4g or 3g+3g configuration vs. isolated performance\nNVIDIA A100, n=1000, size=2^27, 95% confidence')
bw_plot('ci-isolation', df_c_isolated, 'MIG compute instances in isolation (STREAM benchmark)\nComput instances inside 7g GPU instance vs. isolated GPU instances of the same size\nNVIDIA A100, n=1000, size=2^27, 95% confidence')
bw_plot('ci-concurrent', df_c_concurrent, 'MIG compute instances under concurrent load (STREAM benchmark)\n1c+3c+3c configuration vs. compute instances in isolation\nNVIDIA A100, n=1000, size=2^27, 95% confidence')
