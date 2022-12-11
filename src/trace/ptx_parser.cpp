#include <vector>
#include <map>
#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>

namespace ptx_parser {

struct Parameter {
    std::string name;
    uint64_t offset;
};

struct AlignedParameter {
    std::string name;
    uint64_t offset;
    uint64_t alignment;
};

struct Buffer {
    uint64_t size;
    uint64_t alignment;
};

bool startsWith(const std::string &str, const std::string &prefix) {
    return str.rfind(prefix, 0) == 0;
}

bool endsWith(const std::string &str, const std::string &suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> split_string(std::string str,
                                      const std::string &delimiter) {
    std::vector<std::string> result;

    size_t pos = 0;
    std::string token;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        token = str.substr(0, pos);
        result.push_back(token);
        str.erase(0, pos + delimiter.length());
    }
    result.push_back(str);
    return result;
}



// Input stream for parsing
std::vector<AlignedParameter>
parse_ptr_parameters(std::ifstream &input, const std::string &kernel_mangle) {
    // 1. Find the first line of the kernel
    std::string line;
    while (getline(input, line)) {
        if (line.find(kernel_mangle) != std::string::npos) {
            break;
        }
    }
    // 2. Find and store parameters of the kernel
    std::map<std::string, Buffer> buffers;
    while (getline(input, line)) {
        if (line.find(')') != std::string::npos) {
            break;
        }
        assert(startsWith(line, ".param") && "Expected .param directive");
        auto splitted_line = split_string(line, " ");

        // Remove last comma
        auto last = splitted_line.back();
        if (endsWith(last, ",")) {
            splitted_line.back() = last.substr(0, last.size() - 1);
        }

        if (splitted_line[1] == ".align") {
            uint64_t buffer_align = std::stoi(splitted_line[2]);
            auto buffer = splitted_line[4];
            auto splitted_buffer = split_string(buffer, "[");
            auto buffer_name = splitted_buffer[0];
            // Remove last ']' from the size
            uint64_t buffer_size = std::stoi(
                splitted_buffer[1].substr(0, splitted_buffer[1].size() - 1));
            buffers[buffer_name] = {buffer_size, buffer_align};
        } else {
            buffers[splitted_line[2]] = {0, 0};
        }
    }

    // Create parameters from buffers
    std::vector<Parameter> parameters;
    for (const auto &buffer : buffers) {
        if (buffer.second.size == 0) {
            parameters.push_back({buffer.first, 0});
        } else {
            uint64_t step = 64 / buffer.second.alignment;
            for (uint64_t offset = 0; offset < buffer.second.size;
                 offset += step) {
                parameters.push_back({buffer.first, offset});
            }
        }
    }

    // 3. Create map: register -> offset -> parameter name
    std::map<std::string, std::map<uint64_t, std::string>> table;
    std::map<std::string, std::map<uint64_t, bool>> is_ptr;
    for (const auto &param : parameters) {
        table[param.name][param.offset] = param.name;
        is_ptr[param.name][param.offset] = false;
    }

    // 4. Read instructions, find what parameters are translated with the cvta instruction
    while (getline(input, line)) {
        if (line.find(".entry") != std::string::npos) {
            break;
        }
        if (startsWith(line, "ld.param.u64")) {
            auto operands = split_string(line, ",");
            auto reg = split_string(operands[0], " ").back();
            auto param = split_string(operands[1], " ").back();
            // remove [] and ; from param name
            param = param.substr(1, param.size() - 3);
            auto param_split = split_string(param, "+");
            auto param_name = param_split[0];
            if (param_split.size() == 1) {
                table[reg][0] = table[param_name][0];
            } else {
                auto offset = std::stoi(param_split[1]);
                table[reg][offset] = table[param_name][offset];
            }
        } else if (startsWith(line, "ld.param.v2.u64") ||
                   startsWith(line, "ld.param.v4.u64")) {
            auto operands = split_string(line, "}");
            auto registers =
                split_string(split_string(operands[0], "{")[1], ",");
            auto param = split_string(operands[1], " ").back();
            // remove [] and ; from param name
            param = param.substr(1, param.size() - 3);
            auto param_split = split_string(param, "+");
            auto param_name = param_split[0];
            auto offset = std::stoi(param_split[1]);
            for (size_t i = 0; i < registers.size(); ++i) {
                auto reg = registers[i];
                // remove leading whitespace if any
                if(reg[0] == ' ') {
                    reg = reg.substr(1);
                }
                uint64_t align = buffers[param_name].alignment;
                uint64_t local_offset = offset + i * (64 / align);
                table[reg][local_offset] = table[param_name][local_offset];
            }
        } else if (startsWith(line, "mov.u64") || startsWith(line, "mov.b64")) {
            auto operands = split_string(line, ",");
            auto reg = split_string(operands[0], " ").back();
            auto param = split_string(operands[1], " ").back();
            //  remove if necessary [] and ; from param name
            if (param[0] == '[') {
                param = param.substr(1, param.size() - 3);
            } else {
                param = param.substr(0, param.size() - 1);
            }
            auto param_split = split_string(param, "+");
            auto param_name = param_split[0];
            // Only move if we are keeping track of the parameter
            if (table.find(param_name) == table.end())
                continue;
            if (param_split.size() > 1) {
                auto offset = std::stoi(param_split[1]);
                if (table[param_name].find(offset) == table[param_name].end())
                    continue;
            }
            table[reg] = table[param_name];

        } else if (startsWith(line, "cvta.to.global.u64")) {
            auto operands = split_string(line, ",");
            auto param = split_string(operands[1], " ").back();
            //  remove if necessary [] and ; from param name
            if (param[0] == '[') {
                param = param.substr(1, param.size() - 3);
            } else {
                param = param.substr(0, param.size() - 1);
            }
            auto param_split = split_string(param, "+");
            auto param_name = param_split[0];
            std::string origin;
            uint64_t offset;
            if (param_split.size() == 1) {
                auto localMap = table[param_name];
                //assert(localMap.size() == 1 && "Expected only one offset");
                offset = localMap.begin()->first;
                origin = localMap.begin()->second;
            } else {
                offset = std::stoi(param_split[1]);
                origin = table[param_name][offset];
            }
            is_ptr[origin][offset] = true;
        }
    }

    // Print table
    // for(const auto& reg : table) {
    //     for(const auto& offset : reg.second) {
    //         std::cout << reg.first << " " << offset.first << " " << offset.second << std::endl;
    //     }
    // }

    // 5. Return the parameters that are translated with cvta
    std::vector<AlignedParameter> pointer_parameters;
    for (const auto &param : is_ptr) {
        for (const auto &offset : param.second) {
            if (offset.second) {
                pointer_parameters.push_back({param.first, offset.first,
                                              buffers[param.first].alignment});
            }
        }
    }
    return pointer_parameters;
}

}
/*

int main() {
    // std::string kernel_mangle = "_ZN2at6native13reduce_kernelILi512ELi1ENS0_8ReduceOpIfNS0_7MeanOpsIffEEjfLi4EEEEEvT1_";
    // std::string kernel_mangle = "_ZN6caffe24math54_GLOBAL__N__beabb26c_14_elementwise_cu_f0da4091_12388115AxpbyCUDAKernelIffEEvlPKT_PKT0_S5_PS6_";
    std::string kernel_mangle = "_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_22index_fill_kernel_implINS0_10OpaqueTypeILi16EEEEEvRNS_14TensorIteratorElllT_EUliE_EEviT1_";

    std::ifstream ptx_file("dumped_ptx.ptx");

    std::vector<AlignedParameter> ptr_parameters = parse_ptr_parameters(ptx_file, kernel_mangle);
    for (const auto& param : ptr_parameters) {
        std::cout << param.name << " " << param.offset << " " << param.alignment << "\n";
    }

    return 0;
}*/
