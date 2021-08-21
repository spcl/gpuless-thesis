import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys
import re
import math
import pandas as pd

cols = ['n_runs', 'method', 'input_size', 'time']
data = pd.DataFrame(columns=cols)

bw = 31.5
bs = 8 / (31.5 * 10**9)

# for i in range(3, 29):
#     rtt = 2 * bs * 2**i * 4 * 1000
#     data = data.append(pd.DataFrame([[1, 'peak bandwith', 2**i * 4, float(rtt)]], columns=cols),
#                        ignore_index=True)

for i in range(3, 29):
    fname = f'bench-tcp-rtt-100-{2**i}-ms.log'
    with open(fname, 'r') as f:
        lines = f.readlines()
    lines = [x.rstrip() for x in lines]
    for l in lines:
        data = data.append(pd.DataFrame([[100, 'remote host', 2**i * 4, float(l)]], columns=cols),
                           ignore_index=True)

for i in range(3, 29):
    fname = f'bench-local-rtt-100-{2**i}-ms.log'
    with open(fname, 'r') as f:
        lines = f.readlines()
    lines = [x.rstrip() for x in lines]
    for l in lines:
        data = data.append(pd.DataFrame([[100, 'local host', 2**i * 4, float(l)]], columns=cols),
                           ignore_index=True)

for i in range(3, 29):
    theory_rtt = 2 * bs * 2**i * 4 * 1000
    fname = f'bench-no-net-rtt-100-{2**i}-ms.log'
    with open(fname, 'r') as f:
        lines = f.readlines()
    lines = [x.rstrip() for x in lines]
    for l in lines:
        data = data.append(pd.DataFrame([[100, 'theoretical speed \n(peak bandwidth + local execution)', 2**i * 4, float(l) + theory_rtt]], columns=cols),
                           ignore_index=True)

sns.set(rc={"figure.figsize": (9.6, 7.2)})
sns.set_theme(style="whitegrid")
ax = sns.lineplot(x="input_size", y="time", hue="method", data=data)

trs = ax.get_xaxis_transform()
ax.axvline(x=2**20, color='green', linestyle='--') # 1 MB
ax.text(2**20 + 2**24, 0.6, '1 MiB', transform=trs, color='green')
ax.axvline(x=2**20*256, color='green', linestyle='--') # 256 MB
ax.text(2**20*256 + 2**24, 0.6, '256 MiB', transform=trs, color='green')
ax.axvline(x=2**30, color='green', linestyle='--') # 1 GB
ax.text(2**30 + 2**24, 0.6, '1 GiB', transform=trs, color='green')

ax.set_xlabel('Data transfer size [bytes]')
ax.set_ylabel('RTT [ms]', rotation='horizontal')
ax.yaxis.set_label_coords(-0.06, 1.01)
ax.set_title('RTT of no-op function execution (with data transfer) over TCP\nNVIDIA A100, n=100')

ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.ticklabel_format(axis='x', style='sci', useMathText=True)

ax.figure.savefig('bench-tcp.pdf')
