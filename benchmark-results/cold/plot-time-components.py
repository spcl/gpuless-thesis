import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import sys
import re
import pandas as pd
import numpy
import re

color1_s0 = '#ca6565'
color1_s1 = '#ffcccc'
# color1_s2 = '#F29F9F'
# color1_s3 = '#A43A3A'
# color1_s4 = '#7C1717'

color2_s0 = '#3d7979'
color2_s1 = '#94b9b9'
# color2_s2 = '#5F9191'
# color2_s3 = '#236262'
# color2_s4 = '#0E4A4A'

# color3_s0 = '#9DBC5E'
# color3_s1 = '#E3F3C2'
# color3_s2 = '#C8E294'
# color3_s3 = '#789936'
# color3_s4 = '#547416'

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
        'resnext50',
        'resnext101',
        'alexnet',
        'vgg19',
        'yolop',
        'MiDaS',
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
        'resnext50': 1,
        'resnext101': 2,
        'alexnet': 3,
        'vgg19': 4,
        'yolop': 5,
        'MiDaS': 6,
        '3d-unet-kits19': 7,
        'BERT-SQuAD': 8
        }
df_stats = df_stats.sort_values(by=['benchmark'], key=lambda x: x.map(sort_map))

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
sns.set(rc={"figure.figsize": (8.5, 7.0)})
# sns.set_theme(style='whitegrid')
sns.set_style("ticks",{'axes.grid' : True})
palette_top = {
        'native': color1_s0,
        'remote': color2_s1,
        }
palette_bottom = {
        'native': 'red', # will be 0.0 value, so unused
        'remote': color2_s0,
        }

order = [
        # 'latency',
        # 'hotspot',
        # 'srad_v1',
        # 'bfs',
        # 'pathfinder',
        'resnet50',
        'resnext50',
        'resnext101',
        'alexnet',
        'vgg19',
        'yolop',
        'MiDaS',
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
ax.tick_params(axis='x', rotation=25)
ax.set_title('Cold execution time components of ML inference benchmarks\nNVIDIA A100, n=20, median execution time')
ax.get_legend().remove()

#
# line plot
#

# sns.set_theme(style='whitegrid')
# sns.set_style('white')
# ax_stats = plt.twinx()
# sns.pointplot(x='benchmark',
#         color='black',
#         y='sync_count',
#         data=df_stats,
#         linestyles=['--'],
#         scale=0.8,
#         ax=ax_stats)
# ax_stats.set_xlabel('')
# ax_stats.set_ylabel('#Synchronizations')

#
# custom legend
#
rect_native = plt.Rectangle((0, 0), 1, 1, fc=color1_s0, edgecolor='none')
rect_host = plt.Rectangle((0, 0), 1, 1, fc=color2_s1, edgecolor='none')
rect_sync = plt.Rectangle((0, 0), 1, 1, fc=color2_s0, edgecolor='none')
# line_handle = ax_stats.lines[0]

legend_entries = [
        'Native execution',
        'Client execution',
        'Waiting for synchronization',
        # '#Synchronizations'
        ]
plt.legend(
        # handles=[rect_native, rect_host, rect_sync, line_handle],
        handles=[rect_native, rect_host, rect_sync],
        labels=legend_entries,
        title='')

ax.figure.savefig('trace-time-components.pdf')
