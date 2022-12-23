# Find in dumped_ptx.ptx the entry line of the kernel function 
# _ZN2at6native13reduce_kernelILi512ELi1ENS0_8ReduceOpIfNS0_7MeanOpsIffEEjfLi4EEEEEvT1_

kernel_mangle = "_ZN2at6native13reduce_kernelILi512ELi1ENS0_8ReduceOpIfNS0_7MeanOpsIffEEjfLi4EEEEEvT1_"
# kernel_mangle = "_ZN6caffe24math54_GLOBAL__N__beabb26c_14_elementwise_cu_f0da4091_12388115AxpbyCUDAKernelIffEEvlPKT_PKT0_S5_PS6_"
# kernel_mangle = "_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_22index_fill_kernel_implINS0_10OpaqueTypeILi16EEEEEvRNS_14TensorIteratorElllT_EUliE_EEviT1_"
dump_filename = "dumped_ptx.ptx"

lines = []
with open(dump_filename, 'r') as f:
    function_started = False
    for line in f:
        if kernel_mangle in line:
            function_started = True
            lines.append(line)
        elif function_started:
            lines.append(line)
            if '.entry' in line:
                break

# remove empty lines and \n
lines = [line.strip() for line in lines if line.strip()]

# Get parameter list
parameters = {}
end_parameters = None
for idx, line in enumerate(lines[1:]): 
    if ')' in line:
        end_parameters = idx + 1
        break
    assert(line.startswith('.param'))
    line = line.strip().split()[1:] # remove .param
    # remove last comma if line ends with comma
    if line[-1].endswith(','):
        line[-1] = line[-1][:-1]
    
    if line[0] == '.align':
        buffer_align = line[1]
        line = line[2:]
        # get buffer size from squared brackets
        buffer_type = line[0][1:]
        buffer_name = line[1].split('[')[0]
        buffer_size = int(line[1].split('[')[1].split(']')[0])
        parameters[buffer_name] = (buffer_type, buffer_size, buffer_align)
    else:
        # here should do some type checking to optimize
        normal_type = line[0][1:]
        normal_name = line[1]
        parameters[normal_name] = (normal_type, None, None)

parameter_lines = lines[1:end_parameters]
function_lines = lines[end_parameters:]

# For each parameter remember in which register it is stored
table = {}
is_ptr = {}
for param_name, param in parameters.items():
    table[param_name] = {}
    is_ptr[param_name] = {}
    if param[1] is not None:
        buffer_size = int(param[1])
        buffer_align = int(param[2])
        ptrs_offset = range(0, buffer_size, 64//buffer_align)
        for ptr_offset in ptrs_offset:
            table[param_name][ptr_offset] = param_name
            is_ptr[param_name][ptr_offset] = False
    else:
        table[param_name][0] = param_name
        is_ptr[param_name][0] = False

# If the param is a pointer, then it has to go through the cvta instruction to global memory. 
# (well, at least in pytorch).

for line in function_lines:
    if line.startswith('ld.param'):
        if line.startswith('ld.param.u64'):
            operands = line.split(',')
            register = operands[0].split()[-1]
            param_name = operands[1].strip().split()[-1]
            # remove [] and ; from param name
            param_name = param_name.replace('[', '').replace(']', '').replace(';', '')
            split = param_name.split('+')
            param_name = split[0]
            # if entry doesn't exist in table create an empty dict
            if register not in table:
                table[register] = {}
            try:
                if len(split) > 1:
                    index = int(split[1])
                    origin = table[param_name].pop(index)
                    table[register][index] = origin
                else:
                    table[register][0] = table[param_name].pop(0)
            except KeyError:
                print(line)

        if line.startswith('ld.param.v2.u64') or line.startswith('ld.param.v4.u64'):
            operands = line.split('}')
            registers = operands[0].split('{')[1].split(',')
            # strip registers
            registers = [reg.strip() for reg in registers]
            param_name = operands[1].strip().split()[-1]
            # remove [] and ; from param name
            param_name = param_name.replace('[', '').replace(']', '').replace(';', '')
            split = param_name.split('+')
            param_name = split[0]
            index = int(split[1])
            for idx, register in enumerate(registers):
                if register not in table:
                    table[register] = {}
                try:
                    align = int(parameters[param_name][2])
                    local_index = index + idx * (64//align)
                    table[register][local_index] = table[param_name].pop(local_index)
                except KeyError:
                    print(line)

    elif line.startswith('cvta'):
        operands = line.split(',')
        param_name = operands[1].strip().split()[-1]
        # remove [] and ; from param name
        param_name = param_name.replace('[', '').replace(']', '').replace(';', '')
        split = param_name.split('+')
        param_name = split[0]
        origin = None
        try:
            if len(split) > 1:
                index = int(split[1])
                origin = table[param_name][index]
            else:
                index = 0
                origin = table[param_name][0]
            is_ptr[origin][index] = True
        except KeyError:
            print(line)

print("IS_PTR")
print(is_ptr)

# print("TABLE")
# print(table)

