#ifndef GPULESS_PTX_TREE_H
#define GPULESS_PTX_TREE_H

#include <array>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../cubin_analysis.hpp"

namespace PtxTreeParser {

enum class PtxNodeKind {
    Parameter,
    Immediate,
    ImmediateArray,
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
    MoveOp,
    LdOp,
    Register,
    invalidOp,
};

enum class SpecialRegisterKind { ThreadId, NThreadIds, CTAId, NCTAIds };

struct KLaunchConfig {
    std::array<int, 3> gridDim;
    std::array<int, 3> blockDim;
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
    int64_t offset;
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
    PtxNodeKind get_kind() override { return PtxNodeKind::Immediate; }

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
    PtxNodeKind get_kind() override { return PtxNodeKind::ImmediateArray; }

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
    PtxSpecialRegister(PtxOperandKind kind, int dim)
        : _dim(dim) {
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
    PtxNodeKind get_kind() override { return PtxNodeKind::SpecialRegister; }

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
    PtxNodeKind get_kind() override { return PtxNodeKind::Parameter; }

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
    PtxNodeKind get_kind() override { return PtxNodeKind::AddOp; }
    friend class PtxMadNode;
    friend class PtxSadNode;
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
    PtxNodeKind get_kind() override { return PtxNodeKind::AddOp; }

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
    PtxNodeKind get_kind() override { return PtxNodeKind::AddOp; }

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
        _child = std::make_unique<PtxAddNode>(std::move(C), std::make_unique<PtxMulNode>(std::move(L), std::move(R)));
    };

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    PtxNodeKind get_kind() override { return PtxNodeKind::AddOp; }

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
    PtxNodeKind get_kind() override { return PtxNodeKind::AbdOp; }

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
        _child = std::make_unique<PtxAddNode>(std::move(C), std::make_unique<PtxAbdNode>(std::move(L), std::move(R)));
    };

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    PtxNodeKind get_kind() override { return PtxNodeKind::AddOp; }

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
        PtxNodeKind get_kind() override { return PtxNodeKind::AddOp; }

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
        PtxNodeKind get_kind() override { return PtxNodeKind::AddOp; }

    private:
        std::unique_ptr<node_type> _left;
        std::unique_ptr<node_type> _right;
    };

class PtxAbsNode : public PtxAbstractNode {
    public:
        using node_type = PtxAbstractNode;
        PtxAbsNode() = default;
        explicit PtxAbsNode(std::unique_ptr<node_type> C)
                : _child(std::move(C)) {};

        void print() const override;
        std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
        void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
        PtxNodeKind get_kind() override { return PtxNodeKind::AddOp; }

    private:
        std::unique_ptr<node_type> _child;
    };

class PtxNegNode : public PtxAbstractNode {
public:
    using node_type = PtxAbstractNode;
    PtxNegNode() = default;
    explicit PtxNegNode(std::unique_ptr<node_type> C)
            : _child(std::move(C)) {};

    void print() const override;
    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config) override;
    void set_child(std::unique_ptr<PtxAbstractNode> child, int idx) override;
    PtxNodeKind get_kind() override { return PtxNodeKind::AddOp; }

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
        PtxNodeKind get_kind() override { return PtxNodeKind::AddOp; }

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
        PtxNodeKind get_kind() override { return PtxNodeKind::AddOp; }

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
        PtxNodeKind get_kind() override { return PtxNodeKind::AddOp; }

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
        PtxNodeKind get_kind() override { return PtxNodeKind::AddOp; }

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
    PtxNodeKind get_kind() override { return PtxNodeKind::Cvta; }

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
                  const std::vector<PtxOperand> &operands);

    std::unique_ptr<PtxAbstractNode> eval(KLaunchConfig *config);

  private:
    std::unique_ptr<node_type> _root;

    // Registers of interest -> leaf node and child index
    std::unordered_map<std::string, std::pair<node_type *, int>>
        _registers_to_leafs;
};

/*
 * Serialization
 */
std::string stringFromNodeKind(PtxNodeKind kind);
std::string stringFromSpecialRegister(SpecialRegisterKind kind);

std::map<std::string, PtxNodeKind> &getStrToPtxNodeKind();
std::map<std::string, PtxParameterType> &getStrToPtxParameterType();
std::map<PtxParameterType, std::string> &getPtxParameterTypeToStr();
std::map<PtxParameterType, int> &getPtxParameterTypeToSize();
PtxParameterType ptxParameterTypeFromString(const std::string &str);
PtxNodeKind ptxNodeKindFromString(const std::string &str);
int byteSizePtxParameterType(PtxParameterType type);

} // namespace PtxTreeParser

#endif // GPULESS_PTX_TREE_H
