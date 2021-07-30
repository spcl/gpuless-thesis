import os
import time
import subprocess

def cmd(command):
    o = os.popen(command)
    return o.read().rstrip()

def cmd_prt(command):
    os.system(command)

def get_gpuid(dev):
    cmd = f'nvidia-smi -L | grep "GPU {dev}:" | sed -rn "s/.*GPU-([a-f0-9-]*).*/\\1/p"'
    p = subprocess.Popen(['bash', '-c', cmd], stdout=subprocess.PIPE)
    output, error = p.communicate()
    return output.decode('ascii').rstrip()

def read_stream_output(ident, stream_out):
    lines = stream_out.split('\n')
    r = {}
    for l in lines:
        s = l.split()
        if len(s) < 2:
            continue
        if s[0] is not None and s[0] == 'Copy:':
            r['copy'] = float(s[1])
        if s[0] is not None and s[0] == 'Scale:':
            r['scale'] = float(s[1])
        if s[0] is not None and s[0] == 'Add:':
            r['add'] = float(s[1])
        if s[0] is not None and s[0] == 'Triad:':
            r['triad'] = float(s[1])
    return f'{ident},{r["copy"]},{r["scale"]},{r["add"]},{r["triad"]}\n'

# n_runs = 1
device = 3
home = os.environ['HOME']
out_dir = f'{home}/bench-results'
# out_dir = 'results'
date = cmd('date --iso-8601=seconds')
out_file = f'{out_dir}/mig-isolation-bench-{date}.out'
out = []

filter_id = ' | head -n1 | sed -rn "s/.*instance ID[ ]+([0-9]*).*/\\1/p"'

# bandwidth no MIG (full GPU)
r = cmd('CUDA_VISIBLE_DEVICES=3 ./cuda-stream/stream')
out.append(read_stream_output('full', r))
print(r)

gpu_uuid = get_gpuid(device)
print(f'GPU {device}: {gpu_uuid}')

# enable MIG, clear config
cmd(f'sudo nvidia-smi -i {device} -mig 1')
cmd(f'sudo nvidia-smi mig -i {device} -dgi')

###
### using one partition in isolation
###

# bandwidth 1g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 19 -C {filter_id}')
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 ./cuda-stream/stream')
out.append(read_stream_output('1g isolated', r))
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 2g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 14 -C {filter_id}')
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 ./cuda-stream/stream')
out.append(read_stream_output('2g isolated', r))
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 3g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 9 -C {filter_id}')
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 ./cuda-stream/stream')
out.append(read_stream_output('3g isolated', r))
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 4g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 5 -C {filter_id}')
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 ./cuda-stream/stream')
out.append(read_stream_output('4g isolated', r))
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 7g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 0 -C {filter_id}')
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 ./cuda-stream/stream')
out.append(read_stream_output('7g isolated', r))
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

###
### using all partitions at the same time
###

# bandwidth 1g, other partitions used
# i = cmd_prt(f'sudo nvidia-smi mig -i {device} -cgi 19,14,5 -C {filter_id}')
# r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 ./cuda-stream/stream')
# cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/1/0 ./cuda-stream/stream')
# cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/1/0 ./cuda-stream/stream')
# out.append(read_stream_output('1g full', r))
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# # bandwidth 2g, other partitions used
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# # bandwidth 3g, other partitions used
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# # bandwidth 4g, other partitions used
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# # bandwidth 7g, other partitions used
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# disable MIG
cmd_prt(f'sudo nvidia-smi -i {device} -mig 0')

# write output
f = open(out_file, 'w')
for line in out:
    f.write(line)
f.close()
