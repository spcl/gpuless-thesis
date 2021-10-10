#ifndef __CUBIN_ANALYSIS_HPP__
#define __CUBIN_ANALYSIS_HPP__

#include <filesystem>
#include <map>
#include <string>
#include <vector>

const char ELF_MAGIC[4] = {0x7f, 'E', 'L', 'F'};

enum CudaFileType {
    Elf,
    Ptx,
};

struct KParamInfo {
    int ordinal;
    int size;
    KParamInfo(int o, int s) : ordinal(o), size(s) {}
};

struct KFunction {
    std::filesystem::path module_path;
    std::vector<KParamInfo> parameters;
};

class CubinAnalyzer {
  private:
    std::vector<std::string> cuda_binaries;

    CudaFileType file_type;
    std::map<std::string, KFunction> kernel_to_kfunction;

    bool fileIsElf(std::filesystem::path &p);
    bool analyzeElf(std::string fname, int major_version, int minor_version);
    bool analyzeSingleElf(const std::filesystem::path &path);
    bool analyzePtx();

  public:
    CubinAnalyzer() = default;
    bool analyze(std::vector<std::string> cuda_binaries, int major_version,
                 int minor_version);

    bool kernel_parameters(std::string &kernel,
                           std::vector<KParamInfo> &params);
    bool kernel_module(std::string &kernel, std::vector<uint8_t> &module_data);
};

#endif // __CUBIN_ANALYSIS_HPP__
