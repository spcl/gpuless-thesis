#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <regex>

#include "cubin_analysis.hpp"

static std::string exec(const char* cmd) {
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

bool CubinAnalyzer::fileIsElf(std::filesystem::path &path) {
    std::ifstream f(path, std::ios::in | std::ios::binary);

    // check ELF file magic number
    char magic_number[4];
    f.read((char *)&magic_number, 4 * sizeof(char));
    if ((memcmp(magic_number, ELF_MAGIC, 4 * sizeof(char))) == 0) {
        return true;
    }

    // check if ELF embedded in a fatbin file
    std::string cmd =
        "cuobjdump -elf " + path.string() + " | grep '.section .symtab'";
    std::string r = exec(cmd.c_str());
    if (r != "") {
        return true;
    }

    return false;
}

bool CubinAnalyzer::analyzeElf() {
    std::string cmd_elf = "cuobjdump -elf " + this->cuda_binary_fname;
    std::string cuobjdump_elf = exec(cmd_elf.c_str());

    std::regex r_nv_info_section("^\\.nv\\.info\\.(.+)$", std::regex::extended);
    std::regex r_kparam_info("^.*EIATTR_KPARAM_INFO.*$", std::regex::extended);
    std::regex r_kparam_value(
        "^.*Ordinal[ \t]+:[ \t]+([0-9x]+).*Size[ \t]+:[ \t]+([0-9x]+).*$",
        std::regex::extended);

    std::stringstream ss(cuobjdump_elf);
    std::string line;
    while (std::getline(ss, line)) {
        std::smatch m;
        if (std::regex_match(line, m, r_nv_info_section)) {
            std::string symbol = m[1];

            std::vector<KParamInfo> kparam_infos;
            while (std::getline(ss, line)) {
                if (std::regex_match(line, r_kparam_info)) {
                    std::string format_line;
                    std::string value_line;
                    std::getline(ss, format_line);
                    std::getline(ss, value_line);

                    std::smatch sm;
                    if (std::regex_match(value_line, sm, r_kparam_value)) {
                        int ordinal = std::stoi(sm[1], nullptr, 16);
                        int size = std::stoi(sm[2], nullptr, 16);
                        kparam_infos.emplace_back(ordinal, size);
                    }
                }

                if (line == "") {
                    break;
                }
            }

            // sort the kernel parameter info by ordinal
            std::sort(kparam_infos.begin(), kparam_infos.end(),
                      [](auto &a, auto &b) { return a.ordinal < b.ordinal; });

            this->_kernel_parameters.insert(
                std::make_pair(symbol, kparam_infos));
        }
    }

    return false;
}

// TODO
bool CubinAnalyzer::analyzePtx() {
    // cuobjdump -ptx
    return false;
}

bool CubinAnalyzer::analyzeCode() {
    auto tmp = std::filesystem::temp_directory_path() / "libcuadapter";
    std::filesystem::path bin(this->cuda_binary_fname);
    std::filesystem::create_directory(tmp);
    std::filesystem::copy_file(bin, tmp / bin.filename());
    auto tmp_bin = tmp / bin.filename();

    std::string cmd = "cd " + tmp.string() + "; cuobjdump --extract-elf all " +
                      tmp_bin.string();
    exec(cmd.c_str());

    std::regex r_xelf_fname(".*\\.sm_([0-9]{2})\\.cubin", std::regex::extended);
    for (const auto &d : std::filesystem::directory_iterator(tmp)) {
        std::string const &f = d.path().string();
        std::smatch m;
        if (std::regex_match(f, m, r_xelf_fname)) {
            int arch_major_minor = std::stoi(m[1]);
            std::ifstream i(d.path(), std::ios::binary);
            std::vector<uint8_t> module_data(std::istreambuf_iterator<char>(i),
                                             {});
            this->_arch_modules.emplace(
                std::make_pair(arch_major_minor, std::move(module_data)));
        }
    }

    std::filesystem::remove_all(tmp);
    return true;
}

const std::map<std::string, std::vector<KParamInfo>>
    &CubinAnalyzer::kernel_parameters() {
    return this->_kernel_parameters;
}

const std::map<int, std::vector<uint8_t>> &CubinAnalyzer::arch_modules() {
    return this->_arch_modules;
}

bool CubinAnalyzer::analyze(const char *fname) {
    this->cuda_binary_fname = fname;

    std::filesystem::path cuda_binary(this->cuda_binary_fname);
    if (!std::filesystem::exists(cuda_binary) ||
        !std::filesystem::is_regular_file(cuda_binary)) {
        std::cerr << "invalid file: " << cuda_binary_fname << std::endl;
        return false;
    }

    this->file_type = fileIsElf(cuda_binary) ? Elf : Ptx;

    bool ret = false;
    if (this->file_type == Elf) {
        ret = analyzeElf();
        ret = analyzeCode();
    } else {
        ret = analyzePtx();
    }

    return ret;
}
