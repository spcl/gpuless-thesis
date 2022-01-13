import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import ScalarFormatter
import os
import sys
import re
import pandas as pd
import numpy

color1_s0 = '#ca6565'
color1_s1 = '#ffcccc'
color1_s2 = '#F29F9F'
color1_s3 = '#A43A3A'
color1_s4 = '#7C1717'

color2_s0 = '#3d7979'
color2_s1 = '#94b9b9'
color2_s2 = '#5F9191'
color2_s3 = '#236262'
color2_s4 = '#0E4A4A'

color3_s0 = '#9DBC5E'
color3_s1 = '#E3F3C2'
color3_s2 = '#C8E294'
color3_s3 = '#789936'
color3_s4 = '#547416'

palette = {
        "AWS Lambda": color2_s0,
        "Native": color1_s0,
        "GPUless remote": color3_s0,
        }

v = []
cols = ['benchmark', 'type', 'time']

def read_bench(fname, name, type_):
    f = open(fname)
    lines = [l.rstrip() for l in f.readlines()]
    for line in lines:
        v.append([name, type_, float(line) * 1000.0])
    f.close()

read_bench('data/aws/alexnet-aws-hot.out', 'alexnet', 'AWS Lambda')
read_bench('data/aws/resnet50-aws-hot.out', 'resnet50', 'AWS Lambda')
read_bench('data/aws/resnext50-aws-hot.out', 'resnext50', 'AWS Lambda')
read_bench('data/aws/resnext101-aws-hot.out', 'resnext101', 'AWS Lambda')
read_bench('data/aws/vgg19-aws-hot.out', 'vgg19', 'AWS Lambda')
read_bench('data/aws/yolop-aws-hot.out', 'yolop', 'AWS Lambda')
read_bench('data/aws/midas-aws-hot.out', 'MiDaS', 'AWS Lambda')
# read_bench('data/aws/3d-unet-kits19-aws-hot.out', '3d-unet-kits19', 'AWS Lambda')
read_bench('data/aws/bert-aws-hot.out', 'BERT-SQuAD', 'AWS Lambda')

read_bench('data/hot/alexnet-native-hot.out', 'alexnet', 'Native')
read_bench('data/hot/resnet50-native-hot.out', 'resnet50', 'Native')
read_bench('data/hot/resnext50-native-hot.out', 'resnext50', 'Native')
read_bench('data/hot/resnext101-native-hot.out', 'resnext101', 'Native')
read_bench('data/hot/vgg19-native-hot.out', 'vgg19', 'Native')
read_bench('data/hot/yolop-native-hot.out', 'yolop', 'Native')
read_bench('data/hot/midas-native-hot.out', 'MiDaS', 'Native')
# read_bench('data/hot/3d-unet-kits19-native-hot.out', '3d-unet-kits19', 'Native')
read_bench('data/hot/BERT-SQuAD-native-hot.out', 'BERT-SQuAD', 'Native')

read_bench('data/hot/alexnet-remote-hot.out', 'alexnet', 'GPUless remote')
read_bench('data/hot/resnet50-remote-hot.out', 'resnet50', 'GPUless remote')
read_bench('data/hot/resnext50-remote-hot.out', 'resnext50', 'GPUless remote')
read_bench('data/hot/resnext101-remote-hot.out', 'resnext101', 'GPUless remote')
read_bench('data/hot/vgg19-remote-hot.out', 'vgg19', 'GPUless remote')
read_bench('data/hot/yolop-remote-hot.out', 'yolop', 'GPUless remote')
read_bench('data/hot/midas-remote-hot.out', 'MiDaS', 'GPUless remote')
# read_bench('data/hot/3d-unet-kits19-remote-hot.out', '3d-unet-kits19', 'GPUless remote')
read_bench('data/hot/BERT-SQuAD-remote-hot.out', 'BERT-SQuAD', 'GPUless remote')

bench_data = pd.DataFrame(data=v, columns=cols)

sns.set(rc={"figure.figsize": (8.5, 7.0)})
# sns.set_theme(style='whitegrid')
# sns.set_style("ticks")
sns.set_style("ticks",{'axes.grid' : True})
# palette = sns.color_palette("muted")

order = [
        'resnet50',
        'resnext50',
        'resnext101',
        'alexnet',
        'vgg19',
        'yolop',
        'MiDaS',
        # '3d-unet-kits19',
        'BERT-SQuAD'
        ]

hue_order = ['Native', 'GPUless remote', 'AWS Lambda']
ax = sns.barplot(x='benchmark',
                 y='time',
                 hue='type',
                 hue_order=hue_order,
                 order=order,
                 # orient='h',
                 data=bench_data,
                 ci=95,
                 palette=palette)
ax.legend_.set_title(None)
ax.set_ylabel('Time [ms]')
ax.set_yscale('log')
ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=25)
ax.set_title('Runtime of ML inference in hot environment\nNVIDIA A100, n=100, 95% confidence')
# plt.grid()
ax.figure.savefig('hot-inference.pdf')
