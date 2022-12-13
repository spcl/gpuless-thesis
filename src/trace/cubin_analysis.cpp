#include <array>
#include <cstring>
#include <cxxabi.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <llvmdemangler/Demangler.h>
#include <regex>
#include <spdlog/spdlog.h>

#include "cubin_analysis.hpp"
#include "cubin_parser/parser_util.h"

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

static std::string exec(const char *cmd) {
    std::array<char, 128> buffer{};
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

bool CubinAnalyzer::isInitialized() { return this->initialized_; }

PtxParameterType
CubinAnalyzer::ptxParameterTypeFromString(const std::string &str) {
    auto it = getStrToPtxParameterType().find(str);
    if (it == getStrToPtxParameterType().end()) {
        return PtxParameterType::invalid;
    }
    return it->second;
}

int CubinAnalyzer::byteSizePtxParameterType(PtxParameterType type) {
    auto it = getPtxParameterTypeToSize().find(type);
    if (it == getPtxParameterTypeToSize().end()) {
        return -1;
    }
    return it->second;
}

/*template <class StringLikeA, class StringLikeB>
static bool iequals(const StringLikeA &a, const StringLikeB &b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                      [](char a, char b) { return tolower(a) == tolower(b); });
}

static bool
deduce_nested_ptr(itanium_demangle::NameWithTemplateArgs *tmp_node) {
    assert(tmp_node != nullptr);

    // If tempplate is an array
    if (iequals(tmp_node->Name->getBaseName(), std::string("array"))) {
        auto template_args = dynamic_cast<itanium_demangle::TemplateArgs *>(
            tmp_node->TemplateArgs);
        if (!template_args)
            throw std::runtime_error("Array type not supported.");
        if (template_args->getParams().size() != 2)
            throw std::runtime_error("Array type not supported.");

        return template_args->getParams()[0]->getKind() ==
               itanium_demangle::Node::KPointerType;
    }

    return false;
}

static std::vector<itanium_demangle::Node *>
expand_mangled_par(itanium_demangle::FunctionEncoding* array) {
    size_t N_unexpanded_par = array->getParams().size();
    std::vector<itanium_demangle::Node *> expanded(N_unexpanded_par);
    unsigned cur_idx = 0;

    for(unsigned i = 0; i < N_unexpanded_par; ++i) {
        if(array->getParams()[i]->getKind() == itanium_demangle::Node::KParameterPackExpansion) {
            auto pack = dynamic_cast<const itanium_demangle::ParameterPack *>(
                            dynamic_cast<itanium_demangle::ParameterPackExpansion *>(array->getParams()[i])
                                ->getChild())->getPack();

            expanded.resize(expanded.size() + pack->size() - 1);
            for(auto p : *pack) {
                expanded[cur_idx] = p;
                ++cur_idx;
            }
        } else {
            expanded[cur_idx] = array->getParams()[i];
            ++cur_idx;
        }
    }
    return expanded;
}

static bool deduce_ptr(itanium_demangle::Node *par) {
    switch (par->getKind()) {
    case itanium_demangle::Node::KNameWithTemplateArgs:
        return deduce_nested_ptr(
            dynamic_cast<itanium_demangle::NameWithTemplateArgs *>(par));
    case itanium_demangle::Node::KPointerType:
        return true;
    default:
        return false;
    }
}*/

struct NameWithOffset {
    std::string name;
    int offset;
};

std::vector<KParamInfo> CubinAnalyzer::parsePtxParameters(const std::string &ptx_data,
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
            int param_align = std::stoi(splitted_line[2].data());
            const std::string_view &name = splitted_line[4];
            std::vector<std::string_view> splitted_name = split_string(name, "[");
            const std::string_view &param_name = splitted_name[0];
            // Remove last ']' from the size
            int param_size = std::stoi(
                splitted_name[1].substr(0, splitted_name[1].size() - 1).data());

            std::string_view type_name = splitted_line[3].substr(1, splitted_line[3].size());
            PtxParameterType param_type = ptxParameterTypeFromString(std::string(type_name));
            int param_typeSize = byteSizePtxParameterType(param_type);

            KParamInfo param(std::string(param_name), ptxParameterTypeFromString(std::string(type_name)), param_typeSize, param_align, param_size, 0);
            raw_parameters.push_back(param);

            for(int offset = 0; offset < param_size; offset += param_align) {
                param.size = param_size - offset;
                params.push_back({std::string(param_name), offset});
            }
        } else {
            std::string_view &name = splitted_line[2];
            std::string_view typeName = splitted_line[1].substr(1, splitted_line[1].size()-1);
            auto type = ptxParameterTypeFromString(std::string(typeName));
            KParamInfo param(std::string(name), type, byteSizePtxParameterType(type), 0, 1, 0);

            raw_parameters.push_back(param);
            params.push_back({std::string(name) ,0});
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
                table[std::string(reg)][0] = table[std::string(param_name)][0];
            } else {
                int offset = std::stoi(param_split[1].data());
                table[std::string(reg)][offset] = table[std::string(param_name)][offset];
            }
        } else if (startsWith(line, "ld.param.v2.u64") || startsWith(line, "ld.param.v4.u64")) {
            auto operands = split_string(line, "}");
            auto registers = split_string(split_string(operands[0], "{")[1], ",");
            auto param = split_string(operands[1], " ").back();
            // remove [] and ; from param name
            param = param.substr(1, param.size() - 3);
            auto param_split = split_string(param, "+");
            auto param_name = param_split[0];
            auto offset = std::stoi(param_split[1].data());
            for(size_t i = 0; i < registers.size(); ++i) {
                auto reg = registers[i];
                uint64_t local_offset = offset + i * 8;
                table[std::string(reg)][local_offset] = table[std::string(param_name)][local_offset];
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
            if(table.find(std::string(param_name)) == table.end()) continue;
            if(param_split.size() > 1) {
                auto offset = std::stoi(param_split[1].data());
                if(table[std::string(param_name)].find(offset) == table[std::string(param_name)].end()) continue;
            }
            table[std::string(reg)] = table[std::string(param_name)];

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
                auto localMap = table[std::string(param_name)];
                if(localMap.empty())
                    continue;
                assert(localMap.size() == 1 && "Expected only one offset");
                offset = 0;
                origin = localMap.begin()->second;
            } else {
                offset = std::stoi(param_split[1].data());
                origin = table[std::string(param_name)][offset];
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

size_t CubinAnalyzer::pathToHash(const std::filesystem::path &path) {
    auto base = path.filename();
    return std::hash<std::string>{}(base.string());
}

bool CubinAnalyzer::isCached(const std::filesystem::path &fname) {
    std::size_t fname_hash = this->pathToHash(fname);
    std::filesystem::path home_dir(std::getenv("HOME"));
    std::filesystem::path cache_dir = home_dir / ".cache" / "libgpuless";
    if (!std::filesystem::is_directory(cache_dir)) {
        std::filesystem::create_directories(cache_dir);
    }
    std::filesystem::path cache_file = cache_dir / std::to_string(fname_hash);
    if (std::filesystem::is_regular_file(cache_file)) {
        return true;
    }
    return false;
}

bool CubinAnalyzer::loadAnalysisFromCache(const std::filesystem::path &fname) {
    std::size_t fname_hash = this->pathToHash(fname);
    std::filesystem::path home_dir(std::getenv("HOME"));
    std::filesystem::path cache_dir = home_dir / ".cache" / "libgpuless";
    std::filesystem::path cache_file = cache_dir / std::to_string(fname_hash);
    if (!std::filesystem::is_regular_file(cache_file)) {
        return false;
    }

    std::map<std::string, std::vector<KParamInfo>> tmp_map;
    std::ifstream in(cache_file);

    while (true) {
        std::string symbol;
        int n_params;
        in >> symbol;
        in >> n_params;

        std::vector<KParamInfo> kparam_infos;
        for (int i = 0; i < n_params; i++) {
            KParamInfo kparam_info;
            uint64_t u64_type;
            in >> kparam_info.paramName;
            in >> u64_type;
            kparam_info.type = PtxParameterType(u64_type);
            in >> kparam_info.typeSize;
            in >> kparam_info.align;
            in >> kparam_info.size;
            int offs_size;
            in >> offs_size;
            std::vector<int> offsets(offs_size);
            for(int l = 0; l < offs_size; ++l)
                in >> offsets[l];
            kparam_info.ptrOffsets = offsets;
            kparam_infos.push_back(kparam_info);
        }

        tmp_map.emplace(symbol, kparam_infos);
        if (in.eof()) {
            break;
        }
    }
    in.close();
    this->kernel_to_kparaminfos.insert(tmp_map.begin(), tmp_map.end());
    return true;
}

void CubinAnalyzer::storeAnalysisToCache(
    const std::filesystem::path &fname,
    const std::map<std::string, std::vector<KParamInfo>> &data) {
    SPDLOG_INFO("Storing analysis to cache: {}", fname.string());
    std::size_t fname_hash = this->pathToHash(fname);
    std::filesystem::path home_dir(std::getenv("HOME"));
    std::filesystem::path cache_dir = home_dir / ".cache" / "libgpuless";
    if (!std::filesystem::is_directory(cache_dir)) {
        std::filesystem::create_directories(cache_dir);
    }
    std::filesystem::path cache_file = cache_dir / std::to_string(fname_hash);

    std::fstream out(cache_file, std::fstream::app);
    for (const auto &d : data) {
        const std::string &symbol = d.first;
        const std::vector<KParamInfo> &kparam_infos = d.second;
        out << symbol << std::endl << kparam_infos.size() << std::endl;
        for (const auto &p : kparam_infos) {
            out << p.paramName << std::endl;
            out << p.type << std::endl;
            out << p.typeSize << std::endl;
            out << p.align << std::endl;
            out << p.size << std::endl;
            out << p.ptrOffsets.size() << std::endl;
            for(auto offs : p.ptrOffsets)
                out << offs << std::endl;
        }
    }
    out.close();
}

bool CubinAnalyzer::analyzePtx(const std::filesystem::path &fname,
                               int major_version, int minor_version) {
    auto tmp = std::filesystem::temp_directory_path() / "libgpuless";
    SPDLOG_INFO("Using tmp directory: {}", tmp.string());
    if (std::filesystem::is_directory(tmp)) {
        std::filesystem::remove_all(tmp);
    }

    std::filesystem::path bin(fname);
    std::filesystem::create_directory(tmp);
    std::filesystem::copy_file(bin, tmp / bin.filename());
    auto tmp_bin = tmp / bin.filename();

    auto tmp_ptx = tmp / "ptx";
    std::filesystem::create_directory(tmp_ptx);

    std::string arch = std::to_string(major_version * 10 + minor_version);
    std::string cmd =
        "cd " + tmp_ptx.string() + "; cuobjdump -xptx all " + tmp_bin.string();
    exec(cmd.c_str());

    for (const auto &d : std::filesystem::directory_iterator(tmp_ptx)) {
        // analyze single ptx file
        std::string const &f = d.path().string();
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
        for (; i != std::sregex_iterator(); ++i) {
            std::smatch m = *i;
            const std::string &entry = m[1];

            std::vector<KParamInfo> param_infos =
                parsePtxParameters(ptx_data, m);
            tmp_map.emplace(std::make_pair(entry, param_infos));
        }
        this->storeAnalysisToCache(std::filesystem::canonical(fname), tmp_map);
        this->kernel_to_kparaminfos.insert(tmp_map.begin(), tmp_map.end());
    }

    return true;
}

bool CubinAnalyzer::analyze(const std::vector<std::string> &cuda_binaries,
                            int major_version, int minor_version) {
    bool ret = false;

    for (const auto &cbin : cuda_binaries) {
        std::filesystem::path cuda_binary(cbin);
        SPDLOG_DEBUG("Analyzing: {}", cuda_binary.string());
        if (!std::filesystem::exists(cuda_binary) ||
            !std::filesystem::is_regular_file(cuda_binary)) {
            SPDLOG_ERROR("Invalid file: {}", cbin);
            return false;
        }

        // check if analysis is cached
        if (this->isCached(cbin)) {
            SPDLOG_INFO("Loading analysis from cache for: {}", cbin);
            ret = this->loadAnalysisFromCache(cbin);
        } else {
            ret = this->analyzePtx(cbin, major_version, minor_version);
        }
    }

    this->initialized_ = true;
    return ret;
}

bool CubinAnalyzer::kernel_parameters(std::string &kernel,
                                      std::vector<KParamInfo> &params) const {
    auto it = this->kernel_to_kparaminfos.find(kernel);
    if (it != this->kernel_to_kparaminfos.end()) {
        params = it->second;
        return true;
    }
    return false;
}

bool CubinAnalyzer::kernel_module(std::string &kernel,
                                  std::vector<uint8_t> &module_data) {
    return true;
}
