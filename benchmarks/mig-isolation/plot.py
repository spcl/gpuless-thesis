import seaborn as sns
import os
import sys
import pandas as pd

fname = sys.argv[1]
with open(fname, 'r') as f:
    lines = f.readlines()
lines = [x.rstrip() for x in lines]

cols = ['config', 'action', 'rate', 'avg', 'min', 'max']
data = pd.DataFrame(columns=cols)

for i in range(0, len(lines)):
    vs = lines[i].split(',')
    vss = [ [vs[0], vs[1], float(vs[2]), float(vs[3]), float(vs[4]), float(vs[5])] ]
    df = pd.DataFrame(vss, columns=cols)
    data = data.append(df, ignore_index=True)

sns.set(rc={"figure.figsize": (9, 6)})
sns.set_theme(style='whitegrid')
ax = sns.barplot(x='rate',
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
