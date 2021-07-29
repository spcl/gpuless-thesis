import os
import time

def cmd(command):
    o = os.popen(command)
    return o.read().rstrip()

def cmd_prt(command):
    os.system(command)

n_runs = 300
device = 3
home = os.environ['HOME']
out_dir = f'{home}/bench-results'
date = cmd('date --iso-8601=seconds')
out_file = f'{out_dir}/mig-reconfig-bench-{date}.out'

print(f'output: {out_file}')

profiles = [
    '0',
    '5,14,19',
    '5,19,19,19',
    '9,9',
    '9,14,19',
    '9,19,19,19',
    '9,14,14',
    '9,14,19,19',
    '9,19,19,14',
    '9,19,19,19,19',
    '14,14,14,19',
    '14,19,19,14,19',
    '19,19,14,14,19',
    '14,19,19,19,19,19',
    '19,19,14,19,19,19',
    '19,19,19,19,14,19',
    '19,19,19,19,19,14',
    '19,19,19,19,19,19,19',
]

# profiles = [
#     '19,19,19,19,19,19,19',
#     '5,14,19',
#     '5,19,19,19',
#     '19,19,19,19,14,19',
#     '9,19,19,19',
#     '9,14,14',
#     '0',
#     '9,14,19,19',
#     '9,19,19,14',
#     '9,19,19,19,19',
#     '14,14,14,19',
#     '14,19,19,14,19',
#     '9,9',
#     '14,19,19,19,19,19',
#     '19,19,14,14,19',
#     '9,14,19',
#     '19,19,14,19,19,19',
#     '19,19,19,19,19,14',
# ]

times_enable = []
times_disable = []

for i in range(0, n_runs):
    times_enable.append([])
    times_disable.append([])

# enable MIG, clear config
cmd(f'sudo nvidia-smi -i {device} -mig 1')
cmd(f'sudo nvidia-smi mig -i {device} -dgi')

for i in range(0, n_runs):
    print(f'run {i}')

    for j in range(0, len(profiles)):
        s = time.perf_counter()
        cmd_prt(f'sudo nvidia-smi mig -i {device} -cgi {profiles[j]}')
        e = time.perf_counter()
        times_enable[j].append(e-s)

        s = time.perf_counter()
        cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')
        e = time.perf_counter()
        times_disable[j].append(e-s)

# write output
f = open(out_file, 'w')

for j in range(0, len(profiles)):
    line1 = ','.join(map(str, times_enable[j])) + '\n'
    line2 = ','.join(map(str, times_disable[j])) + '\n'
    f.write(line1)
    f.write(line2)

f.close()
cmd(f'sudo nvidia-smi -i {device} -mig 0')
