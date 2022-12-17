#ifndef GPULESS_PTX_TREE_H
#define GPULESS_PTX_TREE_H

#include <array>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
namespace PtxTreeParser {
class PtxAbstractNode;
}

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

namespace PtxTreeParser {

enum class PtxNodeKind {
    Parameter,
    Immediate,
    SpecialRegister,
    Cvta,
    AddOp,
    SubOp,
    MulOp,
    AbdOp,
    MadOp,
    SadOp,
    DivOp,
    RemOp,
    AbsOp,
    NegOp,
    MinOp,
    MaxOp,
    ShlOp,
    ShrOp,
    BfiOp,
    AndOp,
    MoveOp,
    Register,
    InvalidOp,
    LdOp
};

enum class SpecialRegisterKind { ThreadId, NThreadIds, CTAId, NCTAIds };

struct KLaunchConfig {
    std::array<unsigned, 3> gridDim;
    std::array<unsigned, 3> blockDim;
    std::vector<std::vector<uint8_t>> *paramBuffers;
    std::vector<UncollapsedKParamInfo> *paramInfos;

    KLaunchConfig(std::array<unsigned, 3> grid, std::array<unsigned, 3> block)
        : gridDim(grid), blockDim(block), paramBuffers(nullptr) {}

    KLaunchConfig(std::array<unsigned, 3> grid, std::array<unsigned, 3> block, std::vector<std::vector<uint8_t>> *buffers, std::vector<UncollapsedKParamInfo> *infos)
        : gridDim(grid), blockDim(block), paramBuffers(buffers), paramInfos(infos) {}
};

enum class PtxOperandKind {
    Register,
    Parameter,
    Immediate,
    SpecialRegisterTid,
    SpecialRegisterNTid,
    SpecialRegisterCtaId,
    SpecialRegisterNCtaId
};

struct PtxOperand {
    PtxOperandKind kind;
    std::string name;
    int64_t value;
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
    [[nodiscard]] virtual PtxNodeKind get_kind() const = 0;

    // Serialization
    virtual void serialize(std::ostream &os) const = 0;
    static std::unique_ptr<PtxAbstractNode> unserialize(std::istream &is);

    typedef std::unique_ptr<PtxAbstractNode> (*Factory)(std::istream &);
    static std::unique_ptr<PtxAbstractNode> null_factory(std::istream &) {
        return nullptr;
    }
};

class PtxImmediate : public PtxAbstractNode {
  public:
    PtxImmediate(int64_t value, PtxParameterType type)
        : _values(1), _type(type) {
        _values[0] = value;
    }

    PtxImmediate(std::vector<int64_t> values, PtxParameterType type)
        : _values(std::move(values)), _type(type) {}

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::Immediate;
    }

    [[nodiscard]] std::vector<int64_t> &get_values() { return _values; }
    [[nodiscard]] const std::vector<int64_t> &get_values() const {
        return _values;
    }
    [[nodiscard]] PtxParameterType get_type() const { return _type; }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::vector<int64_t> _values;
    PtxParameterType _type;
};

class PtxSpecialRegister : public PtxAbstractNode {
  public:
    PtxSpecialRegister(SpecialRegisterKind kind, int dim)
        : _kind(kind), _dim(dim) {}
    PtxSpecialRegister(PtxOperandKind kind, int dim) : _dim(dim) {
        switch (kind) {
        case PtxOperandKind::SpecialRegisterTid:
            _kind = SpecialRegisterKind::ThreadId;
            break;
        case PtxOperandKind::SpecialRegisterNTid:
            _kind = SpecialRegisterKind::NThreadIds;
            break;
        case PtxOperandKind::SpecialRegisterCtaId:
            _kind = SpecialRegisterKind::CTAId;
            break;
        case PtxOperandKind::SpecialRegisterNCtaId:
            _kind = SpecialRegisterKind::NThreadIds;
            break;
        default:
            throw std::runtime_error("Invalid Opearnd for SpecialRegister.");
        }
    }
    PtxSpecialRegister(SpecialRegisterKind kind, int dim,
                       std::vector<int64_t> values)
        : _kind(kind), _dim(dim), _values(std::move(values)) {}

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::SpecialRegister;
    }

    [[nodiscard]] std::vector<int64_t> &get_values() { return _values; }
    [[nodiscard]] const std::vector<int64_t> &get_values() const {
        return _values;
    }
    [[nodiscard]] std::pair<SpecialRegisterKind, int> get_reg_kind() const {
        return {_kind, _dim};
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    SpecialRegisterKind _kind;
    int _dim;
    std::vector<int64_t> _values;
};

class PtxLdOp : public PtxAbstractNode {
  public:
    PtxLdOp(std::unique_ptr<PtxAbstractNode> child, PtxParameterType type)
        : _child(std::move(child)), _type(type) {}

    PtxLdOp(PtxParameterType type) : _child(nullptr), _type(type) {}

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::LdOp;
    }
    std::unique_ptr<PtxAbstractNode>& get_child() {return _child; }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

    [[nodiscard]] PtxParameterType get_type() { return _type; }

  private:
    std::unique_ptr<PtxAbstractNode> _child;
    PtxParameterType _type;
};

class PtxParameter : public PtxAbstractNode {
  public:
    PtxParameter(std::string name, std::vector<int64_t> offsets, int64_t align,
                 PtxParameterType type)
        : _name(std::move(name)), _offsets(std::move(offsets)), _align(align),
          _type(type) {}
    PtxParameter(std::string name, int64_t offset, int64_t align,
                 PtxParameterType type)
        : _name(std::move(name)), _offsets(1), _align(align), _type(type) {
        _offsets[0] = offset;
    }
    explicit PtxParameter(std::string name, int64_t offset)
        : _name(std::move(name)), _offsets(1), _align(0),
          _type(PtxParameterType::s64) {
        _offsets[0] = offset;
    }

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::Parameter;
    }

    [[nodiscard]] std::string get_name() const { return _name; };
    [[nodiscard]] std::vector<int64_t> &get_offsets() { return _offsets; };
    [[nodiscard]] int64_t get_align() { return _align; }
    [[nodiscard]] PtxParameterType get_type() const { return _type; };

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::string _name;
    std::vector<int64_t> _offsets;
    int64_t _align;
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
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::AddOp;
    }
    friend class PtxMadNode;
    friend class PtxSadNode;

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<node_type> _left;
    std::unique_ptr<node_type> _right;
};

class PtxSubNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxSubNode() = default;
    explicit PtxSubNode(std::unique_ptr<node_type> L,
                        std::unique_ptr<node_type> R)
        : _left(std::move(L)), _right(std::move(R)){};

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::SubOp;
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<node_type> _left;
    std::unique_ptr<node_type> _right;
};

class PtxMulNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxMulNode() = default;
    explicit PtxMulNode(std::unique_ptr<node_type> L,
                        std::unique_ptr<node_type> R)
        : _left(std::move(L)), _right(std::move(R)){};

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::MulOp;
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<node_type> _left;
    std::unique_ptr<node_type> _right;
};

class PtxMadNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxMadNode() = default;
    explicit PtxMadNode(std::unique_ptr<node_type> C,
                        std::unique_ptr<node_type> L,
                        std::unique_ptr<node_type> R) {
        _child = std::make_unique<PtxAddNode>(
            std::move(C),
            std::make_unique<PtxMulNode>(std::move(L), std::move(R)));
    };
    explicit PtxMadNode(std::unique_ptr<PtxAddNode> child)
        : _child(std::move(child)) {}

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::MadOp;
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<PtxAddNode> _child;
};

class PtxAbdNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxAbdNode() = default;
    explicit PtxAbdNode(std::unique_ptr<node_type> L,
                        std::unique_ptr<node_type> R)
        : _left(std::move(L)), _right(std::move(R)){};

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::AbdOp;
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<node_type> _left;
    std::unique_ptr<node_type> _right;
};

class PtxSadNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxSadNode() = default;
    explicit PtxSadNode(std::unique_ptr<node_type> C,
                        std::unique_ptr<node_type> L,
                        std::unique_ptr<node_type> R) {
        _child = std::make_unique<PtxAddNode>(
            std::move(C),
            std::make_unique<PtxAbdNode>(std::move(L), std::move(R)));
    };
    explicit PtxSadNode(std::unique_ptr<PtxAddNode> child)
        : _child(std::move(child)) {}

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::SadOp;
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<PtxAddNode> _child;
};

class PtxDivNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxDivNode() = default;
    explicit PtxDivNode(std::unique_ptr<node_type> L,
                        std::unique_ptr<node_type> R)
        : _left(std::move(L)), _right(std::move(R)){};

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::DivOp;
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<node_type> _left;
    std::unique_ptr<node_type> _right;
};

class PtxRemNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxRemNode() = default;
    explicit PtxRemNode(std::unique_ptr<node_type> L,
                        std::unique_ptr<node_type> R)
        : _left(std::move(L)), _right(std::move(R)){};

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::RemOp;
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<node_type> _left;
    std::unique_ptr<node_type> _right;
};

class PtxAbsNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxAbsNode() = default;
    explicit PtxAbsNode(std::unique_ptr<node_type> C) : _child(std::move(C)){};

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::AbsOp;
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<node_type> _child;
};

class PtxNegNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxNegNode() = default;
    explicit PtxNegNode(std::unique_ptr<node_type> C) : _child(std::move(C)){};

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::NegOp;
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<node_type> _child;
};

class PtxMinNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxMinNode() = default;
    explicit PtxMinNode(std::unique_ptr<node_type> L,
                        std::unique_ptr<node_type> R)
        : _left(std::move(L)), _right(std::move(R)){};

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::MinOp;
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<node_type> _left;
    std::unique_ptr<node_type> _right;
};

class PtxMaxNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxMaxNode() = default;
    explicit PtxMaxNode(std::unique_ptr<node_type> L,
                        std::unique_ptr<node_type> R)
        : _left(std::move(L)), _right(std::move(R)){};

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::MaxOp;
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<node_type> _left;
    std::unique_ptr<node_type> _right;
};

class PtxShlNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxShlNode() = default;
    explicit PtxShlNode(std::unique_ptr<node_type> L,
                        std::unique_ptr<node_type> R)
        : _left(std::move(L)), _right(std::move(R)){};

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::ShlOp;
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<node_type> _left;
    std::unique_ptr<node_type> _right;
};

class PtxShrNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxShrNode() = default;
    explicit PtxShrNode(std::unique_ptr<node_type> L,
                        std::unique_ptr<node_type> R)
        : _left(std::move(L)), _right(std::move(R)){};

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::ShrOp;
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<node_type> _left;
    std::unique_ptr<node_type> _right;
};

class PtxBfiNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxBfiNode() = default;
    explicit PtxBfiNode(std::unique_ptr<node_type> L,
                        std::unique_ptr<node_type> R,
                        std::unique_ptr<node_type> pos,
                        std::unique_ptr<node_type> length)
        : _left(std::move(L)), _right(std::move(R)),
          _pos(std::move(pos)), _length(std::move(length)){};

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::BfiOp;
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<node_type> _left;
    std::unique_ptr<node_type> _right;
    std::unique_ptr<node_type> _pos;
    std::unique_ptr<node_type> _length;
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
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::Cvta;
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<node_type> _dst;
};

class PtxAndNode : public PtxAbstractNode {
  public:
    using node_type = PtxAbstractNode;
    PtxAndNode() = default;
    explicit PtxAndNode(std::unique_ptr<node_type> L,
                        std::unique_ptr<node_type> R)
        : _left(std::move(L)), _right(std::move(R)){};

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    [[nodiscard]] PtxNodeKind get_kind() const override {
        return PtxNodeKind::AndOp;
    }

    void serialize(std::ostream &os) const override;
    static std::unique_ptr<PtxAbstractNode> create(std::istream &is);

  private:
    std::unique_ptr<node_type> _left;
    std::unique_ptr<node_type> _right;
};

class PtxTree {
  public:
    using node_type = PtxAbstractNode;

    explicit PtxTree(const std::string &register_name)
        : _root(std::make_unique<PtxCvtaNode>()) {
        _registers_to_leafs[register_name] = {std::make_pair(_root.get(), 0)};
    }

    explicit PtxTree(std::unique_ptr<node_type> root)
        : _root(std::move(root)) {}

    void print() const;

    // Gets the node that is currently pointed to in place of the register
    std::vector<std::pair<node_type *, int>>
    find_register_node(const std::string &register_name);

    // Replace name of register
    void replace_register(const std::string &old_register,
                          const std::string &new_register);

    void add_node(PtxNodeKind new_node,
                  const std::vector<PtxOperand> &operands);
    void add_node(std::unique_ptr<PtxParameter> new_node,
                  const std::vector<PtxOperand> &operands);
    void add_node(std::unique_ptr<PtxImmediate> new_node,
                  const std::vector<PtxOperand> &operands);
    void add_node(std::unique_ptr<PtxLdOp> new_node,
                  const std::vector<PtxOperand> &operands);
    void add_node(std::unique_ptr<PtxSpecialRegister> new_node,
                  const std::vector<PtxOperand> &operands);

    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config);

    void serialize(std::ostream &os) { _root->serialize(os); }
    static std::unique_ptr<PtxTree> unserialize(std::istream &is) {
        return std::make_unique<PtxTree>(PtxAbstractNode::unserialize(is));
    }

    std::unique_ptr<PtxAbstractNode> move_root() {
        std::unique_ptr<PtxAbstractNode> root = nullptr;
        _root.swap(root);
        return root;
    }

  private:
    std::unique_ptr<node_type> _root;

    // Registers of interest -> leaf node and child index
    std::unordered_map<std::string, std::vector<std::pair<node_type *, int>>>
        _registers_to_leafs;
};

/*
 * Serialization
 */
std::string stringFromNodeKind(PtxNodeKind kind);
std::string stringFromSpecialRegister(SpecialRegisterKind kind);

std::map<std::string, PtxNodeKind> &getStrToPtxNodeKind();
std::map<PtxParameterType, std::string> &getPtxParameterTypeToStr();
PtxNodeKind ptxNodeKindFromString(const std::string &str);

std::unique_ptr<PtxTree::node_type> produceNode(PtxNodeKind kind);

} // namespace PtxTreeParser

#endif // GPULESS_PTX_TREE_H
