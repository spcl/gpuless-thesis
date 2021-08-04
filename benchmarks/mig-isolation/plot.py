import seaborn as sns
import os
import sys
import re
import pandas as pd

fname = sys.argv[1]
with open(fname, 'r') as f:
    lines = f.readlines()
lines = [x.rstrip() for x in lines]

# config_order = [
#         r".*g isolated",
#         r"no partition",
#         r".*c isolated",
#         r"1c+3c+3c.*",
#         r"1g+2g+4g.*",
#         r"3g+3g.*",
#         ]

# def config_to_order(config):
#     for i in range(0, len(config_order)):
#         if bool(re.match(config_order[i], config)):
#             return i

def action_to_bw(action, time):
    G = 1 << 30
    N = 1 << 27
    bw = 0.0
    if action == 'triad' or action == 'add':
        bw = 3 * 8 * N / time / G
    else:
        bw = 2 * 8 * N / time / G
    return bw

cols = ['config', 'action', 'time', 'bw']

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
        ]]
    if bool(re.match(r".*g isolated", vs[0])) or bool(re.match(r"no partition", vs[0])):
        df_g_isolated = df_g_isolated.append(pd.DataFrame(vss, columns=cols), ignore_index=True)
    elif bool(re.match(".*c isolated", vs[0])):
        df_c_isolated = df_c_isolated.append(pd.DataFrame(vss, columns=cols), ignore_index=True)
    elif bool(re.match(".*(.*g)", vs[0])):
        df_g_concurrent = df_g_concurrent.append(pd.DataFrame(vss, columns=cols), ignore_index=True)
    elif bool(re.match(".*(.*c)", vs[0])):
        df_c_concurrent = df_c_concurrent.append(pd.DataFrame(vss, columns=cols), ignore_index=True)

df_g_isolated = df_g_isolated.sort_values('config')
df_c_isolated = df_c_isolated.sort_values('config')
df_g_concurrent = df_g_concurrent.sort_values('config')
df_c_concurrent = df_c_concurrent.sort_values('config')

data = data.append(df_g_isolated, ignore_index=True)
data = data.append(df_c_isolated, ignore_index=True)
data = data.append(df_g_concurrent, ignore_index=True)
data = data.append(df_c_concurrent, ignore_index=True)

# data = data.sort_values('order')

sns.set(rc={"figure.figsize": (10, 10)})
sns.set_theme(style='whitegrid')
ax = sns.barplot(x='bw',
                 y='config',
                 hue='action',
                 orient='h',
                 # ci=95,
                 ci=None,
                 data=data)
ax.set_xlabel('Bandwidth [Gb/s]')
ax.set_ylabel('Partition', rotation='horizontal')
ax.yaxis.set_label_coords(-0.06, 1.01)
ax.legend(loc='upper right')
ax.set_title('Bandwidth of MIG partitions (STREAM benchmark)\nNVIDIA A100, n=1000, size=2^27, 95% confidence')
ax.figure.savefig(f'results/{os.path.basename(fname)}.pdf')
