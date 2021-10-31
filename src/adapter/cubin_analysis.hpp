#ifndef __CUBIN_ANALYSIS_HPP__
#define __CUBIN_ANALYSIS_HPP__

#include <filesystem>
#include <map>
#include <string>
#include <vector>

//const char ELF_MAGIC[4] = {0x7f, 'E', 'L', 'F'};

enum PtxParameterType {
    s8, s16, s32, s64,    // signed integers
    u8, u16, u32, u64,    // unsigned integers
    f16, f16x2, f32, f64, // floating-point
    b8, b16, b32, b64,    // untyped bits
    pred,                 // predicate
    invalid,              // invalid type for signaling errors
};

static std::map<std::string, PtxParameterType> str_to_ptx_paramter_type = {
    {"s8", s8},   {"s16", s16},   {"s32", s32}, {"s64", s64}, {"u8", u8},
    {"u16", u16}, {"u32", u32},   {"u64", u64}, {"f16", f16}, {"f16x2", f16x2},
    {"f32", f32}, {"f64", f64},   {"b8", b8},   {"b16", b16}, {"b32", b32},
    {"b64", b64}, {"pred", pred},
};

static std::map<PtxParameterType, int> ptx_paramter_type_to_size = {
    {s8, 1},  {s16, 2}, {s32, 4}, {s64, 8},   {u8, 1},   {u16, 2},
    {u32, 4}, {u64, 8}, {f16, 2}, {f16x2, 4}, {f32, 4},  {f64, 8},
    {b8, 1},  {b16, 2}, {b32, 4}, {b64, 8},   {pred, 0},
};

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
    std::vector<std::string> cuda_binaries;
    std::map<std::string, std::vector<KParamInfo>> kernel_to_kparaminfos;

    PtxParameterType ptxParameterTypeFromString(const std::string &str);
    int byteSizePtxParameterType(PtxParameterType type);


    // CudaFileType file_type;
//    std::map<std::string, KFunction> kernel_to_kfunction;
//    std::map<std::string, int> kernel_to_ptx_section_idx;

    // bool fileIsElf(std::filesystem::path &p);
    // bool analyzeElf(std::string fname, int major_version, int minor_version);
    // bool analyzeSingleElf(const std::filesystem::path &path);

//    void parseSymbols(std::stringstream &ss);
    std::vector<KParamInfo> parsePtxParameters(const std::string &params);
    bool analyzePtx(const std::filesystem::path &path,
            int major_version, int minor_version);

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
