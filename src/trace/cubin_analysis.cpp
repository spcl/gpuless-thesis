#include <array>
#include <cstring>
#include <cxxabi.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <spdlog/spdlog.h>
#include <string>
#include <utility>
#include <vector>

#include "cubin_analysis.hpp"
#include "cubin_parser/parser_util.h"
#include "cubin_parser/tree_parser.h"



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

std::string_view range_to_view(std::string::iterator first,
                               std::string::iterator last) {
    if (first != last)
        return {first.operator->(), static_cast<size_t>(last - first)};
    else
        return {nullptr, 0};
}

std::vector<UncollapsedKParamInfo>
CubinAnalyzer::parsePtxParameters(std::string::iterator beg,
                                  std::string::iterator end) {
    // Extract raw parameters from ptx
    std::vector<UncollapsedKParamInfo> raw_parameters;
    std::unordered_map<std::string, std::size_t> par_to_idx;
    std::unordered_set<std::string> param_names;

    std::string line;
    while (getline(beg, end, line)) {
        if (line.find(')') != std::string::npos) {
            break;
        }
        // NO parameters
        if (!startsWith(line, ".param")) {
            break;
        }
        auto splitted_line = splitString(line, " ");

        // Remove last comma
        auto last = splitted_line.back();
        if (endsWith(last, ",")) {
            splitted_line.back() = last.substr(0, last.size() - 1);
        }

        if (splitted_line[1] == ".align") {
            int param_align = std::stoi(splitted_line[2].data());
            const std::string_view &name = splitted_line[4];
            std::vector<std::string_view> splitted_name =
                splitString(name, "[");
            const std::string_view &param_name = splitted_name[0];
            // Remove last ']' from the size
            int param_size = std::stoi(
                splitted_name[1].substr(0, splitted_name[1].size() - 1).data());

            std::string_view type_name =
                splitted_line[3].substr(1, splitted_line[3].size());
            PtxParameterType param_type =
                ptxParameterTypeFromString(std::string(type_name));
            int param_typeSize = byteSizePtxParameterType(param_type);

            UncollapsedKParamInfo param(
                std::string(param_name),
                ptxParameterTypeFromString(std::string(type_name)),
                param_typeSize, param_align, param_size);
            par_to_idx[param.paramName] = raw_parameters.size();
            param_names.insert(param.paramName);
            raw_parameters.emplace_back(std::move(param));
        } else {
            std::string_view &name = splitted_line[2];
            std::string_view typeName =
                splitted_line[1].substr(1, splitted_line[1].size() - 1);
            auto type = ptxParameterTypeFromString(std::string(typeName));
            UncollapsedKParamInfo param(std::string(name), type,
                                        byteSizePtxParameterType(type), 0, 1);

            par_to_idx[param.paramName] = raw_parameters.size();
            param_names.insert(param.paramName);
            raw_parameters.push_back(std::move(param));
        }
    }

    PtxTreeParser::TreeParser parser(std::move(param_names));
    auto trees = parser.parsePtxTrees(range_to_view(beg, end));

    for (auto &tree : trees) {
        auto collapsed = tree.first->eval(nullptr);
        size_t idx = par_to_idx[tree.second];

        if (collapsed) {
            raw_parameters[idx].trees.emplace_back(std::move(collapsed), true);
        } else {
            raw_parameters[idx].trees.emplace_back(tree.first->move_root(),
                                                   false);
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

    std::map<std::string, std::vector<UncollapsedKParamInfo>> tmp_map;
    std::ifstream in(cache_file);

    size_t cnt = 0;

    while (true) {
        std::string symbol;
        int n_params;
        in >> symbol;
        in >> n_params;
        ++cnt;
        if(cnt == 9115) {
            cnt = 2;
        }
        std::vector<UncollapsedKParamInfo> kparam_infos;
        for (int i = 0; i < n_params; i++) {
            UncollapsedKParamInfo kparam_info;
            uint64_t u64_type;
            in >> kparam_info.paramName;
            in >> u64_type;
            kparam_info.type = PtxParameterType(u64_type);
            in >> kparam_info.typeSize;
            in >> kparam_info.align;
            in >> kparam_info.size;
            size_t trees_size;
            in >> trees_size;
            std::vector<std::pair<
                std::unique_ptr<PtxTreeParser::PtxAbstractNode>, bool>>
                trees;
            for (std::size_t k = 0; k < trees_size; ++k) {
                std::unique_ptr<PtxTreeParser::PtxAbstractNode> tree =
                    PtxTreeParser::PtxAbstractNode::unserialize(in);
                bool is_collapsed;
                in >> is_collapsed;
                trees.emplace_back(std::move(tree), is_collapsed);
            }
            kparam_infos.push_back(std::move(kparam_info));
        }

        tmp_map.emplace(symbol, std::move(kparam_infos));
        if (in.eof()) {
            break;
        }
    }
    in.close();
    for (auto &m : tmp_map) {
        this->kernel_to_kparaminfos.emplace(std::move(m));
    }
    return true;
}

void CubinAnalyzer::storeAnalysisToCache(
    const std::filesystem::path &fname,
    const std::map<std::string, std::vector<UncollapsedKParamInfo>> &data) {
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
        const std::vector<UncollapsedKParamInfo> &kparam_infos = d.second;
        out << symbol << '\n' << kparam_infos.size() << '\n';
        for (const auto &p : kparam_infos) {
            out << p.paramName << '\n';
            out << p.type << '\n';
            out << p.typeSize << '\n';
            out << p.align << '\n';
            out << p.size << '\n';
            out << p.trees.size() << '\n';
            for (const auto &tree : p.trees) {
                tree.first->serialize(out);
                out << '\n';
                out << tree.second << '\n';
            }
        }
    }
    out.close();
}

std::map<std::string, std::vector<UncollapsedKParamInfo>>
CubinAnalyzer::analyzeTxt(std::string &ptx_data, int major_version,
                          int minor_version) {
    std::string::iterator it = ptx_data.begin();
    std::string::iterator file_end = ptx_data.end();

    std::map<std::string, std::vector<UncollapsedKParamInfo>> tmp_map;
    std::string line;

    while (getline(it, file_end, line)) {
        if(line.empty())
            continue;
        if (line.find(".entry") == std::string::npos)
            continue;

        std::string entry = std::string(splitString(line, " ").back());
        entry.pop_back();

        if(entry == "_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_15gpu_kernel_implINS0_15CUDAFunctor_addIfEEEEvRNS_18TensorIteratorBaseERKT_EUliE_EEviT1_") {
            std::cout << "Found";
        }
        auto entry_beg = it;
        while (getline(it, file_end, line)) {
            if(line.find(".entry") != std::string::npos) {
                --it;
                while(*(--it) != '\n')
                    ;
                break;
            }
        }
        auto entry_end = it;

        std::vector<UncollapsedKParamInfo> param_infos =
            parsePtxParameters(entry_beg, entry_end);

        tmp_map.emplace(entry, std::move(param_infos));
    }

    return tmp_map;
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
        std::string ptx_data = ss.str();

        auto tmp_map = analyzeTxt(ptx_data, major_version, minor_version);
        this->storeAnalysisToCache(std::filesystem::canonical(fname), tmp_map);
        for (auto &m : tmp_map) {
            this->kernel_to_kparaminfos.emplace(std::move(m));
        }
    }
    size_t cnt = this->kernel_to_kparaminfos.size();
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

std::vector<UncollapsedKParamInfo> *
CubinAnalyzer::kernel_parameters(std::string &kernel) {
    auto it = this->kernel_to_kparaminfos.find(kernel);
    if (it != this->kernel_to_kparaminfos.end()) {
        return &(it->second);
    }
    return nullptr;
}

bool CubinAnalyzer::kernel_module(std::string &kernel,
                                  std::vector<uint8_t> &module_data) {
    return true;
}
int CubinAnalyzer::getline(std::string::iterator &beg,
                           std::string::iterator end, std::string &line) {
    if (beg == end)
        return false;
    char c;
    line.clear();
    while ((beg != end) && ((c = *(beg++)) != '\n')) {
        line.push_back(c);
    }
    return true;
}
