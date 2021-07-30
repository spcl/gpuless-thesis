import seaborn as sns
import os
import sys
import pandas as pd

fname = sys.argv[1]
with open(fname, 'r') as f:
    lines = f.readlines()
lines = [x.rstrip() for x in lines]

cols = ['config', 'action', 'bandwidth']
data = pd.DataFrame(columns=cols)

for i in range(0, len(lines)):
    vs = lines[i].split(',')
    vss = [
            [vs[0], 'copy', float(vs[1])],
            [vs[0], 'scale', float(vs[2])],
            [vs[0], 'add', float(vs[3])],
            [vs[0], 'triad', float(vs[4])],
            ]
    # vss = vs[0:1] + list(map(float, vs[1:]))
    df = pd.DataFrame(vss, columns=cols)
    data = data.append(df, ignore_index=True)

sns.set(rc={"figure.figsize": (9, 6)})
sns.set_theme(style='whitegrid')
ax = sns.barplot(x='bandwidth',
                 y='config',
                 hue='action',
                 orient='h',
                 ci=None,
                 data=data)
ax.set_xlabel('Peak Bandwidth [Gb/s]')
ax.set_ylabel('Partition', rotation='horizontal')
ax.yaxis.set_label_coords(-0.06, 1.01)
ax.legend(loc='upper right')
ax.figure.savefig(f'results/{os.path.basename(fname)}.pdf')
