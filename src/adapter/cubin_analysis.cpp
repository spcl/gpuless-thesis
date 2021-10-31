#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>

#include "cubin_analysis.hpp"

static std::string exec(const char *cmd) {
    std::array<char, 128> buffer;
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

// bool CubinAnalyzer::fileIsElf(std::filesystem::path &path) {
//     std::ifstream f(path, std::ios::in | std::ios::binary);

//     // check ELF file magic number
//     char magic_number[4];
//     f.read((char *)&magic_number, 4 * sizeof(char));
//     if ((memcmp(magic_number, ELF_MAGIC, 4 * sizeof(char))) == 0) {
//         return true;
//     }

//     // check if ELF embedded in a fatbin file
//     std::string cmd =
//         "cuobjdump -symbols " + path.string() + " | grep 'STT_FUNC'";
//     std::string r = exec(cmd.c_str());
//     if (r != "") {
//         return true;
//     }

//     return false;
// }

// bool CubinAnalyzer::analyzeSingleElf(const std::filesystem::path &path) {
//     std::string cmd_elf = "cuobjdump -elf " + path.string();
//     std::string cuobjdump_elf = exec(cmd_elf.c_str());

//     std::regex r_nv_info_section("^\\.nv\\.info\\.(.+)$",
//     std::regex::extended); std::regex
//     r_kparam_info("^.*EIATTR_KPARAM_INFO.*$", std::regex::extended);
//     std::regex r_kparam_value(
//         "^.*Ordinal[ \t]+:[ \t]+([0-9x]+).*Size[ \t]+:[ \t]+([0-9x]+).*$",
//         std::regex::extended);

//     std::stringstream ss(cuobjdump_elf);
//     std::string line;
//     while (std::getline(ss, line)) {
//         std::smatch m;
//         if (std::regex_match(line, m, r_nv_info_section)) {
//             std::string symbol = m[1];

//             std::vector<KParamInfo> kparam_infos;
//             while (std::getline(ss, line)) {
//                 if (std::regex_match(line, r_kparam_info)) {
//                     std::string format_line;
//                     std::string value_line;
//                     std::getline(ss, format_line);
//                     std::getline(ss, value_line);

//                     std::smatch sm;
//                     if (std::regex_match(value_line, sm, r_kparam_value)) {
//                         int ordinal = std::stoi(sm[1], nullptr, 16);
//                         int size = std::stoi(sm[2], nullptr, 16);
//                         kparam_infos.emplace_back(ordinal, size);
//                     }
//                 }

//                 if (line == "") {
//                     break;
//                 }
//             }

//             // sort the kernel parameter info by ordinal
//             std::sort(kparam_infos.begin(), kparam_infos.end(),
//                       [](auto &a, auto &b) { return a.ordinal < b.ordinal;
//                       });
//             this->kernel_to_kfunction.insert(std::make_pair(symbol,
//             KFunction{path, kparam_infos}));
//         }
//     }

//     return true;
// }

// bool CubinAnalyzer::analyzeElf(std::string fname, int major_version, int
// minor_version) {
//     auto tmp = std::filesystem::temp_directory_path() / "libcuadapter";
//     if (std::filesystem::is_directory(tmp)) {
//         std::filesystem::remove_all(tmp);
//     }
//     std::filesystem::path bin(fname);
//     std::filesystem::create_directory(tmp);
//     std::filesystem::copy_file(bin, tmp / bin.filename());
//     auto tmp_bin = tmp / bin.filename();

//     std::string arch = std::to_string(major_version * 10 + minor_version);
//     std::string cmd = "cd " + tmp.string() + "; cuobjdump -arch sm_" + arch +
//                       " -xelf all " + tmp_bin.string();
//     exec(cmd.c_str());

//     std::regex r_xelf_fname(".*\\.sm_([0-9]{2})\\.(cubin|fatbin)",
//     std::regex::extended); for (const auto &d :
//     std::filesystem::directory_iterator(tmp)) {
//         std::string const &f = d.path().string();
//         std::smatch m;
//         if (std::regex_match(f, m, r_xelf_fname)) {
//             analyzeSingleElf(d.path());
//         }
//     }

//     // std::filesystem::remove_all(tmp);
//     //     return true;
//     // }

//     // std::string cmd_elf = "cuobjdump -elf " + this->cuda_binary_fname;
//     // std::string cuobjdump_elf = exec(cmd_elf.c_str());

//     // std::regex r_nv_info_section("^\\.nv\\.info\\.(.+)$",
//     std::regex::extended);
//     // std::regex r_kparam_info("^.*EIATTR_KPARAM_INFO.*$",
//     std::regex::extended);
//     // std::regex r_kparam_value(
//     //     "^.*Ordinal[ \t]+:[ \t]+([0-9x]+).*Size[ \t]+:[ \t]+([0-9x]+).*$",
//     //     std::regex::extended);

//     // std::stringstream ss(cuobjdump_elf);
//     // std::string line;
//     // while (std::getline(ss, line)) {
//     //     std::smatch m;
//     //     if (std::regex_match(line, m, r_nv_info_section)) {
//     //         std::string symbol = m[1];

//     //         std::vector<KParamInfo> kparam_infos;
//     //         while (std::getline(ss, line)) {
//     //             if (std::regex_match(line, r_kparam_info)) {
//     //                 std::string format_line;
//     //                 std::string value_line;
//     //                 std::getline(ss, format_line);
//     //                 std::getline(ss, value_line);

//     //                 std::smatch sm;
//     //                 if (std::regex_match(value_line, sm, r_kparam_value))
//     {
//     //                     int ordinal = std::stoi(sm[1], nullptr, 16);
//     //                     int size = std::stoi(sm[2], nullptr, 16);
//     //                     kparam_infos.emplace_back(ordinal, size);
//     //                 }
//     //             }

//     //             if (line == "") {
//     //                 break;
//     //             }
//     //         }

//     //         // sort the kernel parameter info by ordinal
//     //         std::sort(kparam_infos.begin(), kparam_infos.end(),
//     //                   [](auto &a, auto &b) { return a.ordinal < b.ordinal;
//     });

//     //         this->_kernel_parameters.insert(
//     //             std::make_pair(symbol, kparam_infos));
//     //     }
//     // }

//     return false;
// }

// void CubinAnalyzer::parseSymbols(std::stringstream &ss) {
//     static std::regex r_fatbin_ptx("^Fatbin ptx code:",
//     std::regex::extended); static std::regex r_fatbin_elf("^Fatbin elf
//     code:", std::regex::extended);
//
//     static std::regex r_symbol(
//         "^STT[a-zA-Z_]*[ \t]+[a-zA-Z_]+[ \t]+[a-zA-Z_]+[ \t]+(.*)",
//         std::regex::extended);
//
//     int section_ptx = 1;
//     int section_elf = 1;
//     bool in_sec_ptx = false;
//
//     std::string line;
//     while (std::getline(ss, line)) {
//         std::smatch m_section_ptx;
//         if (std::regex_match(line, m_section_ptx, r_fatbin_ptx)) {
//             in_sec_ptx = true;
//             section_ptx++;
//             continue;
//         }
//
//         std::smatch m_section_elf;
//         if (std::regex_match(line, m_section_elf, r_fatbin_elf)) {
//             in_sec_ptx = false;
//             section_elf++;
//             continue;
//         }
//
//         std::smatch m_symbol;
//         if (td::regex_match(line, m_symbol, r_symbol)) {
//             std::string sym = m_symbol[1];
//             if (in_sec_ptx) {
//                 std::cout << sym << std::endl;
//                 this->kernel_to_ptx_section_idx.emplace(
//                     std::make_pair(sym, section_ptx));
//             } else {
//                 this->kernel_to_ptx_section_idx.emplace(
//                     std::make_pair(sym, section_elf));
//             }
//         }
//     }
// }

PtxParameterType
CubinAnalyzer::ptxParameterTypeFromString(const std::string &str) {
    auto it = str_to_ptx_paramter_type.find(str);
    if (it == str_to_ptx_paramter_type.end()) {
        return PtxParameterType::invalid;
    }
    return it->second;
}

int CubinAnalyzer::byteSizePtxParameterType(PtxParameterType type) {
    auto it = ptx_paramter_type_to_size.find(type);
    if (it == ptx_paramter_type_to_size.end()) {
        return -1;
    }
    return it->second;
}

std::vector<KParamInfo>
CubinAnalyzer::parsePtxParameters(const std::string &params) {
    std::vector<KParamInfo> ps;

//    static std::regex r_param(
//        "\\.param\\s*\\.([a-zA-Z0-9]*)\\s*(?:(?:\\.ptr\\s*)?(?:\\.(?:const|"
//        "global|local|shared)\\s*)?\\.align\\s*([0-9]*))?\\s*([a-zA-Z0-9_]*)(?:"
//        "\\[([0-9]*)\\])?",
//        std::regex::ECMAScript);

    static std::regex r_param(
        "\\.param\\s*(?:\\.align\\s*([0-9]*)\\s*)?\\.([a-zA-Z0-9]*)\\s*([a-zA-"
        "Z0-9_]*)(?:\\[([0-9]*)\\])?",
        std::regex::ECMAScript);

    std::sregex_iterator i =
        std::sregex_iterator(params.begin(), params.end(), r_param);
    for (; i != std::sregex_iterator(); ++i) {
        std::smatch m = *i;
        const std::string &align = m[1];
        const std::string &type = m[2];
        const std::string &name = m[3];
        const std::string &size = m[4];

        int ialign = 0;
        if (align != "") {
            ialign = std::stoi(align);
        }

        int isize = 1;
        if (size != "") {
            isize = std::stoi(size);
        }

        PtxParameterType ptxParameterType = ptxParameterTypeFromString(type);
        int typeSize = byteSizePtxParameterType(ptxParameterType);
        ps.push_back(KParamInfo{
            name,
            ptxParameterType,
            typeSize,
            ialign,
            isize,
        });

//        printf("type=%s,align=%s,symbol=%s,size=%s\n", type.c_str(),
//               align.c_str(), name.c_str(), size.c_str());
    }

    return ps;
}

bool CubinAnalyzer::analyzePtx(const std::filesystem::path &fname,
                               int major_version, int minor_version) {
    auto tmp = std::filesystem::temp_directory_path() / "libgpuless";
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
    std::string cmd = "cd " + tmp_ptx.string() + ";" + "cuobjdump -arch=sm_" +
                      arch + " -xptx all " + tmp_bin.string();
    exec(cmd.c_str());

    for (const auto &d : std::filesystem::directory_iterator(tmp_ptx)) {
        // analyze single ptx file
        std::string const &f = d.path().string();
        std::ifstream s(f);
        std::stringstream ss;
        ss << s.rdbuf();

        static std::regex r_func_parameters(".entry.*\\s(.*)\\(([^\\)]*)\\)",
                                            std::regex::ECMAScript);
        std::string ptx_data = ss.str();
        std::sregex_iterator i = std::sregex_iterator(
            ptx_data.begin(), ptx_data.end(), r_func_parameters);
        for (; i != std::sregex_iterator(); ++i) {
            std::smatch m = *i;
            const std::string &entry = m[1];
            const std::string &params = m[2];

            std::vector<KParamInfo> param_infos = parsePtxParameters(params);
            this->kernel_to_kparaminfos.emplace(
                std::make_pair(entry, param_infos));
        }
    }

    return true;
}

// bool CubinAnalyzer::analyzeCode() {
// auto tmp = std::filesystem::temp_directory_path() / "libcuadapter";
// std::filesystem::path bin(this->cuda_binary_fname);
// std::filesystem::create_directory(tmp);
// std::filesystem::copy_file(bin, tmp / bin.filename());
// auto tmp_bin = tmp / bin.filename();

// std::string cmd = "cd " + tmp.string() + "; cuobjdump --extract-elf all " +
//                   tmp_bin.string();
// exec(cmd.c_str());

// std::regex r_xelf_fname(".*\\.sm_([0-9]{2})\\.cubin", std::regex::extended);
// for (const auto &d : std::filesystem::directory_iterator(tmp)) {
//     std::string const &f = d.path().string();
//     std::smatch m;
//     if (std::regex_match(f, m, r_xelf_fname)) {
//         int arch_major_minor = std::stoi(m[1]);
//         std::ifstream i(d.path(), std::ios::binary);
//         std::vector<uint8_t> module_data(std::istreambuf_iterator<char>(i),
//                                          {});
//         this->_arch_modules.emplace(
//             std::make_pair(arch_major_minor, std::move(module_data)));
//     }
// }

// std::filesystem::remove_all(tmp);
//     return true;
// }

// const std::map<std::string, std::vector<KParamInfo>> &
// CubinAnalyzer::kernel_parameters() {
//     return this->_kernel_parameters;
// }

// const std::map<int, std::vector<uint8_t>> &CubinAnalyzer::arch_modules() {
//     return this->_arch_modules;
// }

bool CubinAnalyzer::analyze(std::vector<std::string> cuda_binaries,
                            int major_version, int minor_version) {
    this->cuda_binaries = cuda_binaries;
    bool ret = false;

    for (const auto &cbin : cuda_binaries) {
        std::filesystem::path cuda_binary(cbin);
        if (!std::filesystem::exists(cuda_binary) ||
            !std::filesystem::is_regular_file(cuda_binary)) {
            std::cerr << "invalid file: " << cbin << std::endl;
            return false;
        }

        ret = analyzePtx(cbin, major_version, minor_version);
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
