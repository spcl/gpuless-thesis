#include <assert.h>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <vector>

enum PtxParameterType {
    s8 = 0,
    s16 = 1,
    s32 = 2,
    s64 = 3, // signed integers
    u8 = 4,
    u16 = 5,
    u32 = 6,
    u64 = 7, // unsigned integers
    f16 = 8,
    f16x2 = 9,
    f32 = 10,
    f64 = 11, // floating-point
    b8 = 12,
    b16 = 13,
    b32 = 14,
    b64 = 15,     // untyped bits
    pred = 16,    // predicate
    invalid = 17, // invalid type for signaling errors
};

std::map<std::string, PtxParameterType> &getStrToPtxParameterType() {
    static std::map<std::string, PtxParameterType> map_ = {
        {"s8", s8},     {"s16", s16},     {"s32", s32}, {"s64", s64},
        {"u8", u8},     {"u16", u16},     {"u32", u32}, {"u64", u64},
        {"f16", f16},   {"f16x2", f16x2}, {"f32", f32}, {"f64", f64},
        {"b8", b8},     {"b16", b16},     {"b32", b32}, {"b64", b64},
        {"pred", pred},
    };
    return map_;
}

std::map<PtxParameterType, std::string> &getPtxParameterTypeToStr() {
    static std::map<PtxParameterType, std::string> map_ = {
        {s8, "s8"},     {s16, "s16"},     {s32, "s32"}, {s64, "s64"},
        {u8, "u8"},     {u16, "u16"},     {u32, "u32"}, {u64, "u64"},
        {f16, "f16"},   {f16x2, "f16x2"}, {f32, "f32"}, {f64, "f64"},
        {b8, "b8"},     {b16, "b16"},     {b32, "b32"}, {b64, "b64"},
        {pred, "pred"},
    };
    return map_;
}

std::map<PtxParameterType, int> &getPtxParameterTypeToSize() {
    static std::map<PtxParameterType, int> map_ = {
        {s8, 1},  {s16, 2}, {s32, 4}, {s64, 8},   {u8, 1},   {u16, 2},
        {u32, 4}, {u64, 8}, {f16, 2}, {f16x2, 4}, {f32, 4},  {f64, 8},
        {b8, 1},  {b16, 2}, {b32, 4}, {b64, 8},   {pred, 0},
    };
    return map_;
}

PtxParameterType
ptxParameterTypeFromString(const std::string &str) {
    auto it = getStrToPtxParameterType().find(str);
    if (it == getStrToPtxParameterType().end()) {
        return PtxParameterType::invalid;
    }
    return it->second;
}

int byteSizePtxParameterType(PtxParameterType type) {
    auto it = getPtxParameterTypeToSize().find(type);
    if (it == getPtxParameterTypeToSize().end()) {
        return -1;
    }
    return it->second;
}

struct KParamInfo {
    std::string paramName;
    PtxParameterType type;
    bool is_ptr;
    int typeSize;
    int align;
    int size;
    std::vector<int> ptrOffsets;
};

struct NameWithOffset {
    std::string name;
    int offset;
};

bool startsWith(const std::string &str, const std::string &prefix) {
    return str.rfind(prefix, 0) == 0;
}

bool endsWith(const std::string &str, const std::string &suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> split_string(std::string str, const std::string &delimiter) {
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

std::vector<KParamInfo>
parsePtxParameters(const std::string &ptx_data,
                                  const std::smatch &match) {
    const std::string &entry = match[1];
    const size_t str_idx = match.position(2)+1;
    std::istringstream ss(ptx_data.substr(str_idx, ptx_data.size() - str_idx));

    // Extract raw parameters from ptx
    std::vector<KParamInfo> raw_parameters;
    std::vector<NameWithOffset> params;
    std::string line;
    while(getline(ss, line)) {
        if (line.find(')') != std::string::npos) {
            break;
        }
        // NO parameters
        if(!startsWith(line, ".param")) {
            break;
        }
        assert(startsWith(line, ".param") && "Expected .param directive");
        auto splitted_line = split_string(line, " ");

        // Remove last comma
        auto last = splitted_line.back();
        if(endsWith(last, ",")) {
            splitted_line.back() = last.substr(0, last.size() - 1);
        }

        if(splitted_line[1] == ".align") {
            int param_align = std::stoi(splitted_line[2]);
            const std::string &name = splitted_line[4];
            std::vector<std::string> splitted_name = split_string(name, "[");
            const std::string &param_name = splitted_name[0];
            // Remove last ']' from the size
            int param_size = std::stoi(
                splitted_name[1].substr(0, splitted_name[1].size() - 1));

            std::string type_name = splitted_line[3].substr(1, splitted_line[3].size());
            PtxParameterType param_type = ptxParameterTypeFromString(type_name);
            int param_typeSize = byteSizePtxParameterType(param_type);

            KParamInfo param;
            param.size = param_size;
            param.paramName = param_name;
            param.align = param_align;
            param.type = ptxParameterTypeFromString(type_name);
            param.typeSize = param_typeSize;
            raw_parameters.push_back(param);

            for(int offset = 0; offset < param_size; offset += param_align) {
                param.size = param_size - offset;
                params.push_back({param_name, offset});
            }
        } else {
            std::string &name = splitted_line[2];
            std::string typeName = splitted_line[1].substr(1, splitted_line[1].size()-1);

            KParamInfo param;
            param.size = 1;
            param.paramName = name;
            param.align = 0;
            param.type = ptxParameterTypeFromString(typeName);
            param.typeSize = byteSizePtxParameterType(param.type);

            raw_parameters.push_back(param);
            params.push_back({name ,0});
        }
    }

    // Map: register -> offset -> identifier
    std::map<std::string, std::map<uint64_t, std::string>> table;
    std::map<std::string, std::map<uint64_t, bool>> is_ptr;
    for(const auto& param : params) {
        table[param.name][param.offset] = param.name;
        is_ptr[param.name][param.offset] = false;
    }

    // Read through instructions
    while (getline(ss, line)) {
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
            if(param_split.size() == 1) {
                table[reg][0] = table[param_name][0];
            } else {
                int offset = std::stoi(param_split[1]);
                table[reg][offset] = table[param_name][offset];
            }
        } else if (startsWith(line, "ld.param.v2.u64") || startsWith(line, "ld.param.v4.u64")) {
            auto operands = split_string(line, "}");
            auto registers = split_string(split_string(operands[0], "{")[1], ",");
            auto param = split_string(operands[1], " ").back();
            // remove [] and ; from param name
            param = param.substr(1, param.size() - 3);
            auto param_split = split_string(param, "+");
            auto param_name = param_split[0];
            auto offset = std::stoi(param_split[1]);
            for(size_t i = 0; i < registers.size(); ++i) {
                auto reg = registers[i];
                uint64_t local_offset = offset + i * 8;
                table[reg][local_offset] = table[param_name][local_offset];
            }
        } else if (startsWith(line, "mov.u64") || startsWith(line, "mov.b64")) {
            auto operands = split_string(line, ",");
            auto reg = split_string(operands[0], " ").back();
            auto param = split_string(operands[1], " ").back();
            //  remove if necessary [] and ; from param name
            if(param[0] == '[') {
                param = param.substr(1, param.size() - 3);
            } else {
                param = param.substr(0, param.size() - 1);
            }
            auto param_split = split_string(param, "+");
            auto param_name = param_split[0];
            // Only move if we are keeping track of the parameter
            if(table.find(param_name) == table.end()) continue;
            if(param_split.size() > 1) {
                auto offset = std::stoi(param_split[1]);
                if(table[param_name].find(offset) == table[param_name].end()) continue;
            }
            table[reg] = table[param_name];

        } else if (startsWith(line, "cvta.to.global.u64")) {
            auto operands = split_string(line, ",");
            auto param = split_string(operands[1], " ").back();
            //  remove if necessary [] and ; from param name
            if(param[0] == '[') {
                param = param.substr(1, param.size() - 3);
            } else {
                param = param.substr(0, param.size() - 1);
            }
            auto param_split = split_string(param, "+");
            auto param_name = param_split[0];
            std::string origin;
            uint64_t offset;
            if(param_split.size() == 1) {
                auto localMap = table[param_name];
                if(localMap.empty())
                    continue;
                assert(localMap.size() == 1 && "Expected only one offset");
                offset = 0;
                origin = localMap.begin()->second;
            } else {
                offset = std::stoi(param_split[1]);
                origin = table[param_name][offset];
            }
            is_ptr[origin][offset] = true;
        }
    }

    for(auto& p : raw_parameters) {
        for(const auto &is_p : is_ptr[p.paramName]) {
            if(is_p.second) {
                p.ptrOffsets.push_back(is_p.first);
            }
        }
    }

    return raw_parameters;
}


int main() {
    // std::string kernel_mangle = "_ZN2at6native13reduce_kernelILi512ELi1ENS0_8ReduceOpIfNS0_7MeanOpsIffEEjfLi4EEEEEvT1_";
    //std::string kernel_mangle = "_ZN6caffe24math54_GLOBAL__N__beabb26c_14_elementwise_cu_f0da4091_12388115AxpbyCUDAKernelIffEEvlPKT_PKT0_S5_PS6_";
    //std::string kernel_mangle = "_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_22index_fill_kernel_implINS0_10OpaqueTypeILi16EEEEEvRNS_14TensorIteratorElllT_EUliE_EEviT1_";
    std::string kernel_mangle = "_ZN2at6native40_GLOBAL__N__73da4bea_8_Shape_cu_49f7391c19CatArrayBatchedCopyIfjLi3ELi128ELi1EEEvPT_NS1_25CatArrInputTensorMetadataIS3_T0_XT2_EXT3_EEENS1_16TensorSizeStrideIS6_Lj4EEEiS6_";
    /*_ZN2at6native54_GLOBAL__N__56c10e91_21_UpSampleBilinear2d_cu_b1fd23ee34upsample_bilinear2d_nhwc_out_frameIffEEvT0_S3_biiiiiPKT_PS4_i
    std::vector<AlignedParameter> ptr_parameters = parse_ptr_parameters(ptx_file, kernel_mangle);
    for (const auto& param : ptr_parameters) {
        std::cout << param.name << " " << param.offset << " " << param.alignment << "\n";
    }
*/

    // analyze single ptx file
    std::string const &f = "/home/paul/ETH/HS22/DPHPC/gpuless/optimistic-execution-analysis/dumped_ptx.ptx";
    std::ifstream s(f);
    std::stringstream ss;
    ss << s.rdbuf();

    std::ifstream s2(f);

    static std::regex r_func_parameters(R"(.entry.*\s(.*)\(([^\)]*)\))",
                                        std::regex::ECMAScript);
    std::string ptx_data = ss.str();
    std::sregex_iterator i = std::sregex_iterator(
        ptx_data.begin(), ptx_data.end(), r_func_parameters);
    std::map<std::string, std::vector<KParamInfo>> tmp_map;

    for (int cnt = 0; i != std::sregex_iterator(); ++i, ++cnt) {
        std::smatch m = *i;
        const int str_idx = m.position(2);
        const std::string &entry = m[1];
        if(entry != kernel_mangle)
            continue;

        std::vector<KParamInfo> param_infos =
            parsePtxParameters(ptx_data, m);
        tmp_map.emplace(std::make_pair(entry, param_infos));

        for(const auto &par : param_infos) {
            if(par.ptrOffsets.size() > 1) {
                std::cout << "Nice" << std::endl;
            }
        }
    }
    return 0;
}