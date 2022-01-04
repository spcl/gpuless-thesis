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

color3_s0 = '#9DBC5E'
# color3_s1 = '#E3F3C2'
# color3_s2 = '#C8E294'
# color3_s3 = '#789936'
# color3_s4 = '#547416'

v = []
cols = ['benchmark', 'type', 'time']

def read_bench(fname, name, type_):
    f = open(fname)
    lines = [l.rstrip() for l in f.readlines()]
    for line in lines:
        v.append([name, type_, float(line)])
    f.close()

read_bench('data/benchmark-bfs-native-local.out', 'bfs', 'Native')
read_bench('data/benchmark-hotspot-native-local.out', 'hotspot', 'Native')
read_bench('data/benchmark-pathfinder-native-local.out', 'pathfinder', 'Native')
read_bench('data/benchmark-srad_v1-native-local.out', 'srad_v1', 'Native')

read_bench('data/benchmark-bfs-remote-remote.out', 'bfs', 'GPUless remote')
read_bench('data/benchmark-hotspot-remote-remote.out', 'hotspot', 'GPUless remote')
read_bench('data/benchmark-pathfinder-remote-remote.out', 'pathfinder', 'GPUless remote')
read_bench('data/benchmark-srad_v1-remote-remote.out', 'srad_v1', 'GPUless remote')

bench_data = pd.DataFrame(data=v, columns=cols)

# sns.set(rc={"figure.figsize": (8.5, 7.0)})
sns.set_theme(style='whitegrid')
# sns.set_style("ticks")
# sns.set_style("ticks",{'axes.grid' : True})


hue_order = ['Native', 'GPUless remote']
order = ['bfs', 'hotspot', 'pathfinder', 'srad_v1']

palette = {
        'Native': color1_s0,
        'GPUless remote': color2_s0,
        }

ax = sns.barplot(x='benchmark',
                 y='time',
                 hue='type',
                 hue_order=hue_order,
                 order=order,
                 # orient='h',
                 palette=palette,
                 data=bench_data,
                 ci=95)
ax.legend_.set_title(None)
ax.set_ylabel('Time [ms]')
ax.set_xlabel('')
ax.set_title('Runtime of Rodinia benchmarks\nNVIDIA A100, n=100, 95% confidence')
ax.figure.savefig('rodinia.pdf')
