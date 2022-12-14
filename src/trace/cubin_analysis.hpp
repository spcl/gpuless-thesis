#ifndef __CUBIN_ANALYSIS_HPP__
#define __CUBIN_ANALYSIS_HPP__

#include <filesystem>
#include <map>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "cubin_parser/ptx_tree.h"

std::map<std::string, PtxParameterType> &getStrToPtxParameterType();
std::map<PtxParameterType, std::string> &getPtxParameterTypeToStr();
std::map<PtxParameterType, int> &getPtxParameterTypeToSize();

struct UncollapsedKParamInfo {
    std::string paramName;
    PtxParameterType type;
    int typeSize{};
    int align{};
    int size{};
    std::vector<
        std::pair<std::unique_ptr<PtxTreeParser::PtxAbstractNode>, bool>>
        trees;

    UncollapsedKParamInfo() = default;
    UncollapsedKParamInfo(const UncollapsedKParamInfo &) = default;
    UncollapsedKParamInfo(UncollapsedKParamInfo &&) = default;

    UncollapsedKParamInfo(std::string name, PtxParameterType par_type,
                          int typesize, int alignment, int size)
        : paramName(std::move(name)), type(par_type), typeSize(typesize),
          align(alignment), size(size), trees(0){};
};

struct KParamInfo {
    std::string paramName;
    PtxParameterType type;
    int typeSize;
    int align;
    int size;
    std::vector<int> ptrOffsets;

    KParamInfo() = default;

    KParamInfo(std::string name, PtxParameterType par_type, int typesize,
               int alignment, int t_size, size_t vec_size)
        : paramName(std::move(name)), type(par_type), typeSize(typesize),
          align(alignment), size(t_size), ptrOffsets(vec_size){};

    KParamInfo(std::string name, PtxParameterType par_type, int typesize,
               int alignment, int offset)
        : paramName(std::move(name)), type(par_type), typeSize(typesize),
          align(alignment), size(1), ptrOffsets(1) {
        ptrOffsets[0] = offset;
    };
};

class CubinAnalyzer {
  private:
    bool initialized_ = false;
    std::map<std::string, std::vector<UncollapsedKParamInfo>>
        kernel_to_kparaminfos;

    static PtxParameterType ptxParameterTypeFromString(const std::string &str);
    static int byteSizePtxParameterType(PtxParameterType type);

    static int getline(std::string::iterator &beg, std::string::iterator end,
                       std::string &line);

    bool isCached(const std::filesystem::path &fname);
    bool loadAnalysisFromCache(const std::filesystem::path &fname);
    void storeAnalysisToCache(
        const std::filesystem::path &fname,
        const std::map<std::string, std::vector<UncollapsedKParamInfo>> &data);

    static size_t pathToHash(const std::filesystem::path &path);

  public:
    std::map<std::string, std::vector<UncollapsedKParamInfo>>
    analyzeTxt(std::string &ptx_data, int major_version, int minor_version);
    bool analyzePtx(const std::filesystem::path &path, int major_version,
                    int minor_version);
    static std::vector<UncollapsedKParamInfo>
    parsePtxParameters(std::string::iterator beg, std::string::iterator end);

    CubinAnalyzer() = default;
    bool isInitialized();
    bool analyze(const std::vector<std::string> &cuda_binaries,
                 int major_version, int minor_version);

    std::vector<UncollapsedKParamInfo> *kernel_parameters(std::string &kernel);
    bool kernel_module(std::string &kernel, std::vector<uint8_t> &module_data);
};

#endif // __CUBIN_ANALYSIS_HPP__
