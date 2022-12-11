#ifndef GPULESS_TREE_PARSER_H
#define GPULESS_TREE_PARSER_H

#include <array>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace PtxTreeParser {

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

typedef enum {
    Parameter,
    Immediate,
    ImmediateArray,
    SpecialRegister,
    Cvta,
    AddOp,
    MoveOp,
    Register,
    invalidOp,
} PtxNodeKind;

typedef enum { ThreadId, NThreadIds, CTAId, NCTAIds } SpecialRegisterKind;

struct KParamInfo {
    std::string paramName;
    PtxParameterType type;
    int typeSize{};
    int align{};
    int size{};
    std::vector<int> ptrOffsets;

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

struct KLaunchConfig {
    std::array<int, 3> gridDim;
    std::array<int, 3> blockDim;
};

class PtxAbstractNode {
  public:
    PtxAbstractNode() = default;
    virtual ~PtxAbstractNode() = default;
    PtxAbstractNode(const PtxAbstractNode &) = delete;
    PtxAbstractNode &operator=(const PtxAbstractNode &) = delete;

    virtual void print() const = 0;
    virtual std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) = 0;
    virtual void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) = 0;
    virtual PtxNodeKind get_kind() = 0;
};

class PtxImmediate : public PtxAbstractNode {
  public:
    PtxImmediate(int64_t value, PtxParameterType type)
        : _value(value), _type(type) {}

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    PtxNodeKind get_kind() override { return Immediate; }

    [[nodiscard]] int64_t &get_value() { return _value; }
    [[nodiscard]] int64_t get_value() const { return _value; }
    [[nodiscard]] PtxParameterType get_type() const { return _type; }

  private:
    int64_t _value;
    PtxParameterType _type;
};

class PtxImmediateArray : public PtxAbstractNode {
  public:
    PtxImmediateArray(std::vector<int64_t> values, PtxParameterType type)
        : _values(std::move(values)), _type(type) {}

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    PtxNodeKind get_kind() override { return ImmediateArray; }

    [[nodiscard]] std::vector<int64_t> &get_values() { return _values; }
    [[nodiscard]] const std::vector<int64_t> &get_values() const {
        return _values;
    }
    [[nodiscard]] PtxParameterType get_type() const { return _type; }

  private:
    std::vector<int64_t> _values;
    PtxParameterType _type;
};

class PtxSpecialRegister : public PtxAbstractNode {
  public:
    PtxSpecialRegister(SpecialRegisterKind kind, int dim)
        : _kind(kind), _dim(dim) {}
    PtxSpecialRegister(SpecialRegisterKind kind, int dim,
                       std::vector<int64_t> values)
        : _kind(kind), _dim(dim), _values(std::move(values)) {}

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    PtxNodeKind get_kind() override { return SpecialRegister; }

    [[nodiscard]] std::vector<int64_t> &get_values() { return _values; }
    [[nodiscard]] const std::vector<int64_t> &get_values() const {
        return _values;
    }
    [[nodiscard]] std::pair<SpecialRegisterKind, int> get_kind() const {
        return {_kind, _dim};
    }

  private:
    SpecialRegisterKind _kind;
    int _dim;
    std::vector<int64_t> _values;
};

class PtxParameter : public PtxAbstractNode {
  public:
    PtxParameter(std::string name, std::vector<int> offsets, int align,
                 PtxParameterType type)
        : _name(std::move(name)), _offsets(std::move(offsets)), _align(align),
          _type(type) {}
    PtxParameter(std::string name, int offset, int align, PtxParameterType type)
        : _name(std::move(name)), _offsets(1), _align(align), _type(type) {
        _offsets[0] = offset;
    }
    explicit PtxParameter(std::string name, int offset)
        : _name(std::move(name)), _offsets(1), _align(0),
          _type(PtxParameterType::invalid) {
        _offsets[0] = offset;
    }

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    PtxNodeKind get_kind() override { return Parameter; }

    [[nodiscard]] std::string get_name() const { return _name; };
    [[nodiscard]] std::vector<int> &get_offsets() { return _offsets; };
    [[nodiscard]] PtxParameterType get_type() const { return _type; };

  private:
    std::string _name;
    std::vector<int> _offsets;
    int _align;
    PtxParameterType _type = PtxParameterType::invalid;
};

class PtxAddNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxAddNode() = default;
    explicit PtxAddNode(std::unique_ptr<node_type> L,
                        std::unique_ptr<node_type> R)
        : _left(std::move(L)), _right(std::move(R)){};

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    PtxNodeKind get_kind() override { return AddOp; }

  private:
    std::unique_ptr<node_type> _left;
    std::unique_ptr<node_type> _right;
};

class PtxCvtaNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxCvtaNode() = default;
    explicit PtxCvtaNode(std::unique_ptr<node_type> dst)
        : _dst(std::move(dst)){};

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    PtxNodeKind get_kind() override { return Cvta; }

  private:
    std::unique_ptr<node_type> _dst;
};

class PtxTree {
  public:
    using node_type = PtxAbstractNode;

    explicit PtxTree(const std::string &register_name)
        : _root(std::make_unique<PtxCvtaNode>()) {
        _registers_to_leafs[register_name] = std::make_pair(_root.get(), 0);
    }

    void print() const;

    // Gets the node that is currently pointed to in place of the register
    std::pair<node_type *, int>
    find_register_node(const std::string &register_name);

    // Replace name of register
    void replace_register(const std::string &old_register,
                          const std::string &new_register);

    void add_node(std::unique_ptr<node_type> new_node,
                  const std::string &dst_register,
                  const std::vector<std::string> &src_registers);

    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config);
  private:
    std::unique_ptr<node_type> _root;

    // Registers of interest -> leaf node and child index
    std::unordered_map<std::string, std::pair<node_type *, int>>
        _registers_to_leafs;
};

std::map<std::string, PtxNodeKind> &getStrToPtxNodeKind();
std::map<std::string, PtxParameterType> &getStrToPtxParameterType();
std::map<PtxParameterType, std::string> &getPtxParameterTypeToStr();
std::map<PtxParameterType, int> &getPtxParameterTypeToSize();
PtxParameterType ptxParameterTypeFromString(const std::string &str);
PtxNodeKind ptxNodeKindFromString(const std::string &str);
int byteSizePtxParameterType(PtxParameterType type);

std::vector<std::string> split_string(std::string str,
                                      const std::string &delimiter);
bool startsWith(const std::string &str, const std::string &prefix);
bool endsWith(const std::string &str, const std::string &suffix);
bool rgetline(std::string::reverse_iterator &it,
              const std::string::reverse_iterator &end, std::string &line);

std::vector<PtxTree> parsePtxTrees(std::string& ss);

} // namespace PtxTreeParser

#endif // GPULESS_TREE_PARSER_H
