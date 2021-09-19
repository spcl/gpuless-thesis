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

class CubinAnalyzer {
  private:
    std::string cuda_binary_fname;
    CudaFileType file_type;
    std::map<std::string, std::vector<KParamInfo>> _kernel_parameters;
    std::map<int, std::vector<uint8_t>> _arch_modules;

    bool fileIsElf(std::filesystem::path &p);
    bool analyzeElf();
    bool analyzePtx();
    bool analyzeCode();

  public:
    bool analyze(const char *fname);
    const std::map<std::string, std::vector<KParamInfo>> &kernel_parameters();
    const std::map<int, std::vector<uint8_t>> &arch_modules();
};

#endif // __CUBIN_ANALYSIS_HPP__
