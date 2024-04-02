#!/usr/bin/env python

import os
import subprocess


def cmd(command):
    o = os.popen(command)
    return o.read().rstrip()


def cmd_prt(command):
    os.system(command)


device = 3
home = os.environ['HOME']
# change to your result directory
out_dir = f'{home}/runs/bench-results'
date = cmd('date --iso-8601=seconds')
# hard coded benchmark directory
benchmark_name = f'hotspot'
path_to_benchmark = f'../{benchmark_name}'
os.chdir(path_to_benchmark)
out_file = f'{out_dir}/A100-{benchmark_name}-bench-{date}.out'
nruns = 100
program = f'./run_timed.sh'
out = []

filter_id1 = ' | head -n1 | sed -rn "s/.*instance ID[ ]+([0-9]*).*/\\1/p"'
filter_ids = ' | grep "created GPU instance" | sed -rn "s/.*instance ID[ ]+([0-9]*).*/\\1/p"'


def get_gpuid(dev):
    cmd = f'nvidia-smi -L | grep "GPU {dev}:" | sed -rn "s/.*GPU-([a-f0-9-]*).*/\\1/p"'
    p = subprocess.Popen(['bash', '-c', cmd], stdout=subprocess.PIPE)
    output, error = p.communicate()
    return output.decode('ascii').rstrip()


def create_partitions(partitions):
    i = cmd(f'sudo nvidia-smi mig -i {device} -cgi {partitions} -C {filter_ids}')
    ids = i.split('\n')
    for k, v in enumerate(ids):
        ids[k] = v.rstrip()
    return ids


def cmd_async(command):
    return subprocess.Popen(['bash', '-c', command], stdout=subprocess.PIPE)


def read_process(p):
    output, error = p.communicate()
    return output.decode('ascii')


# bandwidth no MIG (full GPU)
for _ in range(nruns):
    r = cmd(f'CUDA_VISIBLE_DEVICES={device} {program}')
    out += f'no partition, {r}\n'

# print(r)

gpu_uuid = get_gpuid(device)
print(f'GPU {device}: {gpu_uuid}')

# enable MIG, clear config
cmd(f'sudo nvidia-smi -i {device} -mig 1')
cmd(f'sudo nvidia-smi mig -i {device} -dgi')

#
# using one partition in isolation
#

# bandwidth 1g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 19 -C {filter_id1}')
for _ in range(nruns):
    r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
    out += f'1g isolated, {r}\n'
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 2g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 14 -C {filter_id1}')
for _ in range(nruns):
    r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
    out += f'2g isolated, {r}\n'
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 3g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 9 -C {filter_id1}')
for _ in range(nruns):
    r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
    out += f'3g isolated, {r}\n'
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 4g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 5 -C {filter_id1}')
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
out += f'4g isolated, {r}\n'
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 7g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 0 -C {filter_id1}')
for _ in range(nruns):
    r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
    out += f'7g isolated, {r}\n'
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

#
# using all partitions at the same time
#

# bandwidth 1g+2g+4g
ids = create_partitions('19,14,5')
for _ in range(nruns):
    p1 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[0]}/0 {program}')
    p2 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[1]}/0 {program}')
    p4 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[2]}/0 {program}')
    out += f'1g+2g+4g (1g), {read_process(p1)}'
    out += f'1g+2g+4g (2g), {read_process(p2)}'
    out += f'1g+2g+4g (4g), {read_process(p4)}'
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 3g+3g
ids = create_partitions('9,9')
for _ in range(nruns):
    p3_1 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[0]}/0 {program}')
    p3_2 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[1]}/0 {program}')
    out += f'3g+3g (3g), {read_process(p3_1)}'
    out += f'3g+3g (3g), {read_process(p3_2)}'
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

#
# bandwidth for compute instances
#

# bandwidth 1c
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 0 {filter_id1}')
cmd_prt(f'sudo nvidia-smi mig -i {device} -gi {i} -cci 0')
for _ in range(nruns):
    p1 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
    out += f'1c isolated, {read_process(p1)}'
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 3c
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 0 {filter_id1}')
cmd_prt(f'sudo nvidia-smi mig -i {device} -gi {i} -cci 2')
for _ in range(nruns):
    p1 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
    out += f'3c isolated, {read_process(p1)}'
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 1c+3c+3c
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 0 {filter_id1}')
cmd_prt(f'sudo nvidia-smi mig -i {device} -gi {i} -cci 0,2,2')
for _ in range(nruns):
    p1 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
    p2 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/1 {program}')
    p3 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/2 {program}')
    out += f'1c+3c+3c (1c), {read_process(p1)}'
    out += f'1c+3c+3c (3c), {read_process(p2)}'
    out += f'1c+3c+3c (3c), {read_process(p3)}'
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# disable MIG
cmd_prt(f'sudo nvidia-smi -i {device} -mig 0')

# write output
f = open(out_file, 'w')
for line in out:
    f.write(line)
f.close()
