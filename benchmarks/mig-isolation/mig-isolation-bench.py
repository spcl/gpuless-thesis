import os
import subprocess

def cmd(command):
    o = os.popen(command)
    return o.read().rstrip()

def cmd_prt(command):
    os.system(command)

device = 3
home = os.environ['HOME']
out_dir = f'{home}/runs/bench-results'
date = cmd('date --iso-8601=seconds')
out_file = f'{out_dir}/mig-isolation-bench-{date}.out'
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

def read_stream_output(ident, stream_out):
    lines = stream_out.split('\n')
    ls = []
    for l in lines:
        s = l.split()
        if len(s) < 2:
            continue
        if s[0] is not None and s[0] in ['Copy:', 'Scale:', 'Add:', 'Triad:']:
            print(s)
            ls.append(f'{ident},{s[0][0:-1]},{s[1]},{s[2]},{s[3]},{s[4]}\n')
    return ls

# bandwidth no MIG (full GPU)
r = cmd(f'CUDA_VISIBLE_DEVICES={device} ./cuda-stream/stream')
out += read_stream_output('no partition', r)
print(r)

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
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 ./cuda-stream/stream')
out += read_stream_output('1g isolated', r)
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 2g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 14 -C {filter_id1}')
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 ./cuda-stream/stream')
out += read_stream_output('2g isolated', r)
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 3g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 9 -C {filter_id1}')
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 ./cuda-stream/stream')
out += read_stream_output('3g isolated', r)
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 4g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 5 -C {filter_id1}')
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 ./cuda-stream/stream')
out += read_stream_output('4g isolated', r)
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 7g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 0 -C {filter_id1}')
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 ./cuda-stream/stream')
out += read_stream_output('7g isolated', r)
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

#
# using all partitions at the same time
#

# bandwidth 1g+2g+4g
ids = create_partitions('19,14,5')
# print(f'ids: {ids}')
r1 = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[0]}/0 ./cuda-stream/stream')
r2 = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[1]}/0 ./cuda-stream/stream')
r4 = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[2]}/0 ./cuda-stream/stream')
out += read_stream_output('1g concurrent', r1)
out += read_stream_output('2g concurrent', r2)
out += read_stream_output('4g concurrent', r4)
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 3g+3g+1g
ids = create_partitions('9,9')
# print(f'ids: {ids}')
r3_1 = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[0]}/0 ./cuda-stream/stream')
r3_2 = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[1]}/0 ./cuda-stream/stream')
out += read_stream_output('3g concurrent', r3_1)
out += read_stream_output('3g concurrent', r3_2)
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

#
# bandwidth for compute instances
#

# bandwidth 1c
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 0 {filter_id1}')
cmd_prt(f'sudo nvidia-smi mig -i {device} -gi {i} -cci 0')
r1 = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 ./cuda-stream/stream')
out += read_stream_output('1c isolated', r1)
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 1c+1c+1c+1c+1c+1c+1c
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 0 {filter_id1}')
cmd_prt(f'sudo nvidia-smi mig -i {device} -gi {i} -cci 0,0,0,0,0,0,0')
r1 = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 nsys profile ./cuda-stream/stream')
r2 = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/1 ./cuda-stream/stream')
r3 = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/2 ./cuda-stream/stream')
r4 = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/3 ./cuda-stream/stream')
r5 = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/4 ./cuda-stream/stream')
r6 = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/5 ./cuda-stream/stream')
r7 = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/6 ./cuda-stream/stream')
out += read_stream_output('1c concurrent', r1)
out += read_stream_output('1c concurrent', r2)
out += read_stream_output('1c concurrent', r3)
out += read_stream_output('1c concurrent', r4)
out += read_stream_output('1c concurrent', r5)
out += read_stream_output('1c concurrent', r6)
out += read_stream_output('1c concurrent', r7)
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# disable MIG
cmd_prt(f'sudo nvidia-smi -i {device} -mig 0')

# write output
f = open(out_file, 'w')
for line in out:
    f.write(line)
f.close()
