#ifndef __CUBIN_ANALYSIS_HPP__
#define __CUBIN_ANALYSIS_HPP__

#include <filesystem>
#include <map>
#include <string>
#include <vector>

enum PtxParameterType {
    s8,
    s16,
    s32,
    s64, // signed integers
    u8,
    u16,
    u32,
    u64, // unsigned integers
    f16,
    f16x2,
    f32,
    f64, // floating-point
    b8,
    b16,
    b32,
    b64,     // untyped bits
    pred,    // predicate
    invalid, // invalid type for signaling errors
};

std::map<std::string, PtxParameterType> &getStrToPtxParameterType();
std::map<PtxParameterType, std::string> &getPtxParameterTypeToStr();
std::map<PtxParameterType, int> &getPtxParameterTypeToSize();

struct KParamInfo {
    std::string paramName;
    PtxParameterType type;
    int typeSize;
    int align;
    int size;
};

class CubinAnalyzer {
  private:
    bool initialized_ = false;
    std::map<std::string, std::vector<KParamInfo>> kernel_to_kparaminfos;

    static PtxParameterType ptxParameterTypeFromString(const std::string &str);
    static int byteSizePtxParameterType(PtxParameterType type);

    bool isCached(const std::filesystem::path &fname);
    bool loadAnalysisFromCache(const std::filesystem::path &fname);
    static void storeAnalysisToCache(
        const std::filesystem::path &fname,
        const std::map<std::string, std::vector<KParamInfo>> &data);

    std::vector<KParamInfo> parsePtxParameters(const std::string &params);
    bool analyzePtx(const std::filesystem::path &path, int major_version,
                    int minor_version);

  public:
    CubinAnalyzer() = default;
    bool isInitialized();
    bool analyze(std::vector<std::string> cuda_binaries, int major_version,
                 int minor_version);

    bool kernel_parameters(std::string &kernel,
                           std::vector<KParamInfo> &params) const;
    bool kernel_module(std::string &kernel, std::vector<uint8_t> &module_data);
};

#endif // __CUBIN_ANALYSIS_HPP__
