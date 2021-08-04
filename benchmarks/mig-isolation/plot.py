import seaborn as sns
import os
import sys
import pandas as pd

fname = sys.argv[1]
with open(fname, 'r') as f:
    lines = f.readlines()
lines = [x.rstrip() for x in lines]

cols = ['config', 'action', 'time', 'bw', 'order']
data = pd.DataFrame(columns=cols)

def action_to_bw(action, time):
    G = 1 << 30
    N = 1 << 27
    bw = 0.0
    if action == 'triad' or action == 'add':
        bw = 3 * 8 * N / time / G
    else:
        bw = 2 * 8 * N / time / G
    return bw

for i in range(0, len(lines)):
    vs = lines[i].split(',')
    t = float(vs[2])
    df = pd.DataFrame([[ vs[0], vs[1], t, action_to_bw(vs[1], t) ]], columns=cols)
    data = data.append(df, ignore_index=True)

data = data.sort_values('config')

sns.set(rc={"figure.figsize": (10, 10)})
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
ax.set_title('Bandwidth of MIG partitions (STREAM benchmark)\nNVIDIA A100, n=1000, size=2^27, 95% confidence')
ax.figure.savefig(f'results/{os.path.basename(fname)}.pdf')
