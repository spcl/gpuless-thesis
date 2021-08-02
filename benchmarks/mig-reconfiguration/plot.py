import seaborn as sns
import os
import sys
import pandas as pd

fname = sys.argv[1]
with open(fname, 'r') as f:
    lines = f.readlines()
lines = [x.rstrip() for x in lines]

cols = ['profile', 'time', 'action']
mig_reconfig = pd.DataFrame(columns=cols)

for i in range(0, len(lines)):
    vs = list(map(float, lines[i].split(',')))

    if i % 2 == 0:
        for _, v in enumerate(vs):
            df = pd.DataFrame([[i//2 + 1, v * 1000, 'create']], columns=cols)
            mig_reconfig = mig_reconfig.append(df, ignore_index=True)
    else:
        for _, v in enumerate(vs):
            df = pd.DataFrame([[(i-1)//2 + 1, v * 1000, 'destroy']], columns=cols)
            mig_reconfig = mig_reconfig.append(df, ignore_index=True)

sns.set(rc={"figure.figsize": (8.5, 8)})
sns.set_theme(style='whitegrid')
ax = sns.barplot(x='time',
                 y='profile',
                 hue='action',
                 orient='h',
                 capsize=0.15,
                 ci='sd',
                 errwidth=1.5,
                 data=mig_reconfig)
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Profile', rotation='horizontal')
ax.yaxis.set_label_coords(-0.06, 1.01)
ax.legend(loc='upper right')
ax.figure.savefig(f'results/{os.path.basename(fname)}.pdf')
