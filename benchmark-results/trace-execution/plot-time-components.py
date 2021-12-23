import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import sys
import re
import pandas as pd
import numpy
import re

benchmark_total_median = {}
benchmark_sync_median = {}
benchmark_native_median = {}

benchmark_sync_counter = {}
benchmark_calls_counter = {}

include = [
        # 'latency',
        # 'hotspot',
        # 'srad_v1',
        # 'bfs',
        # 'pathfinder',
        'resnet50',
        'vgg19',
        '3d-unet-kits19',
        'BERT-SQuAD'
        ]

df_stats_values = []
df_stats_cols = ['benchmark', 'sync_count', 'calls_count']
for fname in os.listdir('data/synchronize-stats'):
    r = r'number-synchronizations-([a-zA-Z0-9_-]*)\.out'
    m = re.search(r, fname)
    if m and m.group(1) in include:
        f = open('data/synchronize-stats/' + fname)
        lines = [l.rstrip() for l in f.readlines()]
        l = lines[0].split(' ')
        df_stats_values.append([m.group(1), float(l[0]), float(l[1])])
        # print(m.group(1))
        # print(df_stats_values[-1])
        f.close()
df_stats = pd.DataFrame(data=df_stats_values, columns=df_stats_cols)

sort_map = {
        # 'latency',
        # 'hotspot',
        # 'srad_v1',
        # 'bfs',
        # 'pathfinder',
        'resnet50': 0,
        'vgg19': 1,
        '3d-unet-kits19': 2,
        'BERT-SQuAD': 3
        }
df_stats = df_stats.sort_values(by=['benchmark'], key=lambda x: x.map(sort_map))
print(df_stats)

for fname in os.listdir('data/total'):
    r = r'benchmark-([a-zA-Z0-9_-]*)-remote-remote-.*'
    m = re.search(r, fname)
    if m and m.group(1) in include:
        f = open('data/total/' + fname)
        lines_float = [float(l.rstrip()) for l in f.readlines()]
        benchmark_total_median[m.group(1)] = numpy.median(lines_float)
        f.close()

for fname in os.listdir('data/total'):
    r = r'benchmark-([a-zA-Z0-9_-]*)-native-local-.*'
    m = re.search(r, fname)
    if m and m.group(1) in include:
        f = open('data/total/' + fname)
        lines_float = [float(l.rstrip()) for l in f.readlines()]
        benchmark_native_median[m.group(1)] = numpy.median(lines_float)
        f.close()

for fname in os.listdir('data/synchronize'):
    r = r'benchmark-([a-zA-Z0-9_-]*)-remote-remote-.*'
    m = re.search(r, fname)
    if m and m.group(1) in include:
        f = open('data/synchronize/' + fname)
        lines_float = [float(l.rstrip()) for l in f.readlines()]
        benchmark_sync_median[m.group(1)] = numpy.median(lines_float)
        f.close()

# execution time
rows = []
for k, v in benchmark_sync_median.items():
    total = benchmark_total_median[k]
    rows.append([k, float(total), float(v), 'remote'])
    native_total = benchmark_native_median[k]
    rows.append([k, float(native_total), 0.0, 'native'])

cols = ['benchmark', 'time_total', 'time_sync', 'type']
df = pd.DataFrame(data=rows, columns=cols)

#
# stacked bar plot
#
sns.set(rc={"figure.figsize": (8.0, 6.5)})
sns.set_theme(style='whitegrid')
palette_top = {
        'native': 'lightcoral',
        'remote': 'lightskyblue',
        }
palette_bottom = {
        'native': 'red', # will be 0.0 value, so unused
        'remote': 'sandybrown',
        }

order = [
        # 'latency',
        # 'hotspot',
        # 'srad_v1',
        # 'bfs',
        # 'pathfinder',
        'resnet50',
        'vgg19',
        '3d-unet-kits19',
        'BERT-SQuAD'
        ]

hue_order = ['native', 'remote']
ax_top = sns.barplot(x='benchmark',
        y='time_total',
        hue='type',
        hue_order=hue_order,
        order=order,
        color='indigo',
        orient='v',
        data=df,
        palette=palette_top)

ax = sns.barplot(x='benchmark',
        y='time_sync',
        hue='type',
        hue_order=hue_order,
        order=order,
        color='lightblue',
        orient='v',
        data=df,
        palette=palette_bottom)

ax.set_ylabel('Time [s]')
ax.set_xlabel('')
ax.set_title('Execution time components of ML inference benchmarks\nNVIDIA A100, n=20, median execution time')
ax.get_legend().remove()

#
# line plot
#

# sns.set_theme(style='whitegrid')
sns.set_style('white')
ax_stats = plt.twinx()
sns.pointplot(x='benchmark',
        color='black',
        y='sync_count',
        data=df_stats,
        linestyles=['--'],
        ax=ax_stats)
ax_stats.set_xlabel('')
ax_stats.set_ylabel('#Synchronizations')


#
# custom legend
#
rect_native = plt.Rectangle((0, 0), 1, 1, fc='lightcoral', edgecolor='none')
rect_host = plt.Rectangle((0, 0), 1, 1, fc='lightskyblue', edgecolor='none')
rect_sync = plt.Rectangle((0, 0), 1, 1, fc='sandybrown', edgecolor='none')
line_handle = ax_stats.lines[0]

legend_entries = [
        'Native execution',
        'Host execution',
        'Synchronization',
        '#Synchronizations'
        ]
plt.legend(
        handles=[rect_native, rect_host, rect_sync, line_handle],
        labels=legend_entries,
        title='')

ax.figure.savefig('trace-time-components.pdf')
