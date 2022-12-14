#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <regex>
#include <sstream>

#include "ptx_tree.h"

namespace PtxTreeParser {
/*
 * PTX NODE HELPER
 */
template <class F, class L, class R>
std::unique_ptr<PtxAbstractNode> BinaryEval(KLaunchConfig *config, L &l, R &r,
                                            F op) {
    std::unique_ptr<PtxAbstractNode> left_eval(l->eval(config));
    std::unique_ptr<PtxAbstractNode> right_eval(r->eval(config));

    if (!left_eval && !right_eval) {
        // Could not collapse anything, no evaluation possible
        return nullptr;
    }
    if (!left_eval && right_eval) {
        // Could only collapse rigt node, no evaluation possible
        r.swap(right_eval);
        return nullptr;
    }
    if (left_eval && !right_eval) {
        // Could only collapse left node, no evaluation possible
        l.swap(left_eval);
        return nullptr;
    }

    PtxNodeKind left_kind = left_eval->get_kind();
    PtxNodeKind right_kind = right_eval->get_kind();

    // One node is immediate and one is parameter
    auto par_imm = [&](PtxParameter &par, const PtxImmediate &imm, bool inv) {
        std::vector<int64_t> &offsets(par.get_offsets());
        const std::vector<int64_t> &values(imm.get_values());

        size_t n_offsets = offsets.size();
        size_t n_imm = values.size();

        if (n_imm > 1) {
            for (size_t i = 0; i < n_offsets; ++i) {
                for (size_t k = 1; k < n_imm; ++k) {
                    offsets.push_back(inv ? op(values[k], offsets[i])
                                          : op(offsets[i], values[k]));
                }
            }
        }

        for (size_t i = 0; i < n_offsets; ++i) {
            offsets[i] = inv ? op(imm.get_values()[0], offsets[i])
                             : op(offsets[i], imm.get_values()[0]);
        }
    };
    if (left_kind == PtxNodeKind::Parameter &&
        right_kind == PtxNodeKind::Immediate) {
        par_imm(static_cast<PtxParameter &>(*left_eval),
                static_cast<PtxImmediate &>(*right_eval), false);
        return left_eval;
    }
    if (right_kind == PtxNodeKind::Parameter &&
        left_kind == PtxNodeKind::Immediate) {
        par_imm(static_cast<PtxParameter &>(*right_eval),
                static_cast<PtxImmediate &>(*left_eval), true);
        return right_eval;
    }

    // Both are immediate
    if (left_kind == PtxNodeKind::Immediate &&
        right_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxImmediate &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));

        size_t n_l = left.get_values().size();
        size_t n_r = right.get_values().size();

        auto &l_values(left.get_values());
        auto &r_values(right.get_values());

        for (size_t i = 0; i < n_l; ++i) {
            for (size_t j = 1; j < n_r; ++j) {
                l_values.push_back(op(l_values[i], r_values[j]));
            }
        }

        for (size_t i = 0; i < n_l; ++i)
            l_values[i] = op(l_values[i], r_values[0]);

        return left_eval;
    }

    // One is immediate and one is special register
    auto spec_imm = [&](const PtxSpecialRegister &reg, PtxImmediate &imm,
                        bool inv) {
        const std::vector<int64_t> &reg_values(reg.get_values());
        std::vector<int64_t> &imm_values(imm.get_values());

        size_t n_reg = reg_values.size();
        size_t n_imm = imm_values.size();

        for (size_t i = 0; i < n_imm; ++i) {
            for (size_t j = 1; j < n_reg; ++j) {
                imm_values.push_back(inv ? op(imm_values[i], reg_values[j])
                                         : op(reg_values[j], imm_values[i]));
            }
        }

        for (size_t i = 0; i < n_imm; ++i)
            imm_values[i] = inv ? op(imm_values[i], reg_values[0])
                                : op(reg_values[0], imm_values[i]);
    };
    if (left_kind == PtxNodeKind::SpecialRegister &&
        right_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxSpecialRegister &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));
        spec_imm(left, right, false);
        return right_eval;
    }
    if (right_kind == PtxNodeKind::SpecialRegister &&
        left_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxImmediate &>(*left_eval));
        auto &right(static_cast<PtxSpecialRegister &>(*right_eval));
        spec_imm(right, left, true);
        return left_eval;
    }

    throw std::runtime_error("Binary Operation on these nodes not supported.");
}

template <typename E>
constexpr typename std::underlying_type<E>::type to_underlying(E e) noexcept {
    return static_cast<typename std::underlying_type<E>::type>(e);
}

template<class L, class R>
inline void serializeBinOp(const L &l,const R &r, std::ostream& os, PtxNodeKind kind) {
    os << to_underlying(kind) << " ";
    if(l)
        l->serialize(os);
    else
        os << -1 << " ";

    if(r)
        r->serialize(os);
    else
        os << -1 << " ";
}

template<class T>
inline std::unique_ptr<T> createBinOp(std::istream& is) {
    std::unique_ptr<PtxAbstractNode> left, right;
    left = PtxAbstractNode::unserialize(is);
    right = PtxAbstractNode::unserialize(is);
    return std::make_unique<T>(std::move(left), std::move(right));
}

template<class Derived>
inline std::unique_ptr<Derived> castToDerived(std::unique_ptr<PtxAbstractNode> ptr) {
    return std::unique_ptr<Derived>(static_cast<Derived*>(ptr.release()));
}

/*
 *  PTX NODES IMPLEMENTATIONS
 */

// PtxParameter implementation
void PtxParameter::print() const {
    std::cout << "PtxParameter: Name " << _name << ", Type " << _type
              << ", Offsets ";
    if (this->_offsets.empty()) {
        std::cout << "empty.\n";
    } else {
        for (auto &val : this->_offsets) {
            std::cout << val << '\t';
        }
        std::cout << ".\n";
    }
}
std::unique_ptr<PtxAbstractNode> PtxParameter::eval(KLaunchConfig *config) {
    return std::make_unique<PtxParameter>(_name, _offsets, _align, _type);
}
void PtxParameter::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    throw std::runtime_error("PtxParameter has no children.");
}
void PtxParameter::serialize(std::ostream &os) const {
    os << to_underlying(PtxNodeKind::Parameter) << " ";
    os << _name << " ";
    os << _offsets.size() << " ";
    for (auto offset : _offsets)
        os << offset << " ";
    os << _align << " ";
    os << to_underlying(_type) << " ";
}
std::unique_ptr<PtxAbstractNode> PtxParameter::create(std::istream &is) {
    std::string name;
    int64_t align;
    size_t size;
    int type_id;

    is >> name;
    is >> size;
    std::vector<int64_t> offsets(size);
    for (size_t i = 0; i < size; ++i)
        is >> offsets[i];
    is >> align;
    is >> type_id;

    return std::make_unique<PtxParameter>(name, offsets, align,
                                          PtxParameterType(type_id));
}

// PtxAddNode implementation
void PtxAddNode::print() const {
    std::cout << "PtxAddNode: Left ";
    if (_left) {
        std::cout << "non-empty: \n";
        _left->print();
    } else {
        std::cout << "empty. \n";
    }

    std::cout << "PtxAddNode: Right ";
    if (_right) {
        std::cout << "non-empty: \n";
        _right->print();
    } else {
        std::cout << "empty. \n";
    }
}
std::unique_ptr<PtxAbstractNode> PtxAddNode::eval(KLaunchConfig *config) {
    return BinaryEval(config, _left, _right, std::plus<int64_t>{});
}
void PtxAddNode::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    if (idx == 0) {
        _left.swap(child);
        return;
    }
    if (idx == 1) {
        _right.swap(child);
        return;
    }

    throw std::runtime_error("Invalid index.");
}
void PtxAddNode::serialize(std::ostream &os) const {
    serializeBinOp(_left, _right, os, PtxNodeKind::AddOp);
}
std::unique_ptr<PtxAbstractNode> PtxAddNode::create(std::istream &is) {
    return createBinOp<PtxAddNode>(is);
}

// PtxSubNode implementation
void PtxSubNode::print() const {
    std::cout << "PtxSubNode: Left ";
    if (_left) {
        std::cout << "non-empty: \n";
        _left->print();
    } else {
        std::cout << "empty. \n";
    }

    std::cout << "PtxSubNode: Right ";
    if (_right) {
        std::cout << "non-empty: \n";
        _right->print();
    } else {
        std::cout << "empty. \n";
    }
}
std::unique_ptr<PtxAbstractNode> PtxSubNode::eval(KLaunchConfig *config) {
    return BinaryEval(config, _left, _right, std::minus<int64_t>{});
}
void PtxSubNode::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    if (idx == 0) {
        _left.swap(child);
        return;
    }
    if (idx == 1) {
        _right.swap(child);
        return;
    }

    throw std::runtime_error("Invalid index.");
}
void PtxSubNode::serialize(std::ostream &os) const {
    serializeBinOp(_left, _right, os, PtxNodeKind::SubOp);
}
std::unique_ptr<PtxAbstractNode> PtxSubNode::create(std::istream &is) {
    return createBinOp<PtxSubNode>(is);
}

// PtxMulNode implementation
void PtxMulNode::print() const {
    std::cout << "PtxMulNode: Left ";
    if (_left) {
        std::cout << "non-empty: \n";
        _left->print();
    } else {
        std::cout << "empty. \n";
    }

    std::cout << "PtxMulNode: Right ";
    if (_right) {
        std::cout << "non-empty: \n";
        _right->print();
    } else {
        std::cout << "empty. \n";
    }
}
std::unique_ptr<PtxAbstractNode> PtxMulNode::eval(KLaunchConfig *config) {
    return BinaryEval(config, _left, _right, std::multiplies<int64_t>{});
}
void PtxMulNode::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    if (idx == 0) {
        _left.swap(child);
        return;
    }
    if (idx == 1) {
        _right.swap(child);
        return;
    }

    throw std::runtime_error("Invalid index.");
}
void PtxMulNode::serialize(std::ostream &os) const {
    serializeBinOp(_left, _right, os, PtxNodeKind::MulOp);
}
std::unique_ptr<PtxAbstractNode> PtxMulNode::create(std::istream &is) {
    return createBinOp<PtxMulNode>(is);
}

// PtxMadNode implementation
void PtxMadNode::print() const {
    std::cout << "PtxMadNode: Child ";
    if (_child) {
        std::cout << "non-empty: \n";
        _child->print();
    } else {
        std::cout << "empty. \n";
    }
}
std::unique_ptr<PtxAbstractNode> PtxMadNode::eval(KLaunchConfig *config) {
    return _child->eval(config);
}
void PtxMadNode::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {

    if (idx == 0) {
        _child->set_child(std::move(child), 0);
        return;
    }
    if (idx == 1) {
        _child->_right->set_child(std::move(child), 0);
        return;
    }
    if (idx == 2) {
        _child->_right->set_child(std::move(child), 1);
        return;
    }

    throw std::runtime_error("Invalid index.");
}
void PtxMadNode::serialize(std::ostream &os) const {
    os << to_underlying(PtxNodeKind::MadOp) << " ";
    _child->serialize(os);
}
std::unique_ptr<PtxAbstractNode> PtxMadNode::create(std::istream &is) {
    int type;
    is >> type;
    if(type != to_underlying(PtxNodeKind::AddOp))
        throw std::runtime_error("Child of Mad needs to be Add");
    return std::make_unique<PtxMadNode>(castToDerived<PtxAddNode>(PtxAddNode::create(is)));
}

// PtxSadNode implementation
void PtxSadNode::print() const {
    std::cout << "PtxSadNode: Child ";
    if (_child) {
        std::cout << "non-empty: \n";
        _child->print();
    } else {
        std::cout << "empty. \n";
    }
}
std::unique_ptr<PtxAbstractNode> PtxSadNode::eval(KLaunchConfig *config) {
    return _child->eval(config);
}
void PtxSadNode::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {

    if (idx == 0) {
        _child->set_child(std::move(child), 0);
        return;
    }
    if (idx == 1) {
        _child->_right->set_child(std::move(child), 0);
        return;
    }
    if (idx == 2) {
        _child->_right->set_child(std::move(child), 1);
        return;
    }

    throw std::runtime_error("Invalid index.");
}
void PtxSadNode::serialize(std::ostream &os) const {
    os << to_underlying(PtxNodeKind::SadOp) << " ";
    _child->serialize(os);
}
std::unique_ptr<PtxAbstractNode> PtxSadNode::create(std::istream &is) {
    int type;
    is >> type;
    if(type != to_underlying(PtxNodeKind::AddOp))
        throw std::runtime_error("Child of Mad needs to be Add");
    return std::make_unique<PtxSadNode>(castToDerived<PtxAddNode>(PtxAddNode::create(is)));
}

// PtxDivNode implementation
void PtxDivNode::print() const {
    std::cout << "PtxDivNode: Left ";
    if (_left) {
        std::cout << "non-empty: \n";
        _left->print();
    } else {
        std::cout << "empty. \n";
    }

    std::cout << "PtxDivNode: Right ";
    if (_right) {
        std::cout << "non-empty: \n";
        _right->print();
    } else {
        std::cout << "empty. \n";
    }
}
std::unique_ptr<PtxAbstractNode> PtxDivNode::eval(KLaunchConfig *config) {
    return BinaryEval(config, _left, _right, std::divides<int64_t>{});
}
void PtxDivNode::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    if (idx == 0) {
        _left.swap(child);
        return;
    }
    if (idx == 1) {
        _right.swap(child);
        return;
    }

    throw std::runtime_error("Invalid index.");
}
void PtxDivNode::serialize(std::ostream &os) const {
    serializeBinOp(_left, _right, os, PtxNodeKind::DivOp);
}
std::unique_ptr<PtxAbstractNode> PtxDivNode::create(std::istream &is) {
    return createBinOp<PtxDivNode>(is);
}

// PtxRemNode implementation
void PtxRemNode::print() const {
    std::cout << "PtxRemNode: Left ";
    if (_left) {
        std::cout << "non-empty: \n";
        _left->print();
    } else {
        std::cout << "empty. \n";
    }

    std::cout << "PtxRemNode: Right ";
    if (_right) {
        std::cout << "non-empty: \n";
        _right->print();
    } else {
        std::cout << "empty. \n";
    }
}
std::unique_ptr<PtxAbstractNode> PtxRemNode::eval(KLaunchConfig *config) {
    return BinaryEval(config, _left, _right, std::modulus<int64_t>{});
}
void PtxRemNode::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    if (idx == 0) {
        _left.swap(child);
        return;
    }
    if (idx == 1) {
        _right.swap(child);
        return;
    }

    throw std::runtime_error("Invalid index.");
}
void PtxRemNode::serialize(std::ostream &os) const {
    serializeBinOp(_left, _right, os, PtxNodeKind::RemOp);
}
std::unique_ptr<PtxAbstractNode> PtxRemNode::create(std::istream &is) {
    return createBinOp<PtxRemNode>(is);
}

// PtxAbdNode implementation
void PtxAbdNode::print() const {
    std::cout << "PtxAbdNode: Left ";
    if (_left) {
        std::cout << "non-empty: \n";
        _left->print();
    } else {
        std::cout << "empty. \n";
    }

    std::cout << "PtxAbdNode: Right ";
    if (_right) {
        std::cout << "non-empty: \n";
        _right->print();
    } else {
        std::cout << "empty. \n";
    }
}
std::unique_ptr<PtxAbstractNode> PtxAbdNode::eval(KLaunchConfig *config) {
    return BinaryEval(config, _left, _right,
                      [](int64_t l, int64_t r) { return std::abs(l - r); });
}
void PtxAbdNode::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    if (idx == 0) {
        _left.swap(child);
        return;
    }
    if (idx == 1) {
        _right.swap(child);
        return;
    }

    throw std::runtime_error("Invalid index.");
}
void PtxAbdNode::serialize(std::ostream &os) const {
    serializeBinOp(_left, _right, os, PtxNodeKind::AbdOp);
}
std::unique_ptr<PtxAbstractNode> PtxAbdNode::create(std::istream &is) {
    return createBinOp<PtxAbdNode>(is);
}

// PtxAbsNode implementation
void PtxAbsNode::print() const {
    std::cout << "PtxAbsNode: Child ";
    if (_child) {
        std::cout << "non-empty: \n";
        _child->print();
    } else {
        std::cout << "empty. \n";
    }
}
std::unique_ptr<PtxAbstractNode> PtxAbsNode::eval(KLaunchConfig *config) {
    std::unique_ptr<PtxAbstractNode> child_eval(_child->eval(config));

    if (!child_eval) {
        // Could not collapse anything, no evaluation possible
        return nullptr;
    }

    PtxNodeKind child_kind = child_eval->get_kind();

    // Child is immediate
    if (child_kind == PtxNodeKind::Immediate) {
        auto &child(static_cast<PtxImmediate &>(*child_eval));
        for (auto &value : child.get_values()) {
            value = std::abs(value);
        }

        return child_eval;
    }

    throw std::runtime_error("Absolute of this node not supported.");
}
void PtxAbsNode::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    if (idx == 0) {
        _child.swap(child);
        return;
    }
    throw std::runtime_error("Invalid index.");
}
void PtxAbsNode::serialize(std::ostream &os) const {
    os << to_underlying(PtxNodeKind::AbsOp) << " ";
    _child->serialize(os);
}
std::unique_ptr<PtxAbstractNode> PtxAbsNode::create(std::istream &is) {
    std::unique_ptr<PtxAbstractNode> child = PtxAbstractNode::unserialize(is);
    return std::make_unique<PtxAbsNode>(std::move(child));
}

// PtxNegNode implementation
void PtxNegNode::print() const {
    std::cout << "PtxNegNode: Child ";
    if (_child) {
        std::cout << "non-empty: \n";
        _child->print();
    } else {
        std::cout << "empty. \n";
    }
}
std::unique_ptr<PtxAbstractNode> PtxNegNode::eval(KLaunchConfig *config) {
    std::unique_ptr<PtxAbstractNode> child_eval(_child->eval(config));

    if (!child_eval) {
        // Could not collapse anything, no evaluation possible
        return nullptr;
    }

    PtxNodeKind child_kind = child_eval->get_kind();

    // Child is immediate
    if (child_kind == PtxNodeKind::Immediate) {
        auto &child(static_cast<PtxImmediate &>(*child_eval));
        for (auto &value : child.get_values()) {
            value = -value;
        }
        return child_eval;
    }

    throw std::runtime_error("Negation of this node not supported.");
}
void PtxNegNode::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    if (idx == 0) {
        _child.swap(child);
        return;
    }
    throw std::runtime_error("Invalid index.");
}
void PtxNegNode::serialize(std::ostream &os) const {
    os << to_underlying(PtxNodeKind::NegOp) << " ";
    _child->serialize(os);
}
std::unique_ptr<PtxAbstractNode> PtxNegNode::create(std::istream &is) {
    std::unique_ptr<PtxAbstractNode> child = PtxAbstractNode::unserialize(is);
    return std::make_unique<PtxNegNode>(std::move(child));
}

// PtxMinNode implementation
void PtxMinNode::print() const {
    std::cout << "PtxMinNode: Left ";
    if (_left) {
        std::cout << "non-empty: \n";
        _left->print();
    } else {
        std::cout << "empty. \n";
    }

    std::cout << "PtxMinNode: Right ";
    if (_right) {
        std::cout << "non-empty: \n";
        _right->print();
    } else {
        std::cout << "empty. \n";
    }
}
std::unique_ptr<PtxAbstractNode> PtxMinNode::eval(KLaunchConfig *config) {
    return BinaryEval(config, _left, _right,
                      [](int64_t a, int64_t b) { return std::min(a, b); });
}
void PtxMinNode::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    if (idx == 0) {
        _left.swap(child);
        return;
    }
    if (idx == 1) {
        _right.swap(child);
        return;
    }

    throw std::runtime_error("Invalid index.");
}
void PtxMinNode::serialize(std::ostream &os) const {
    serializeBinOp(_left, _right, os, PtxNodeKind::MinOp);
}
std::unique_ptr<PtxAbstractNode> PtxMinNode::create(std::istream &is) {
    return createBinOp<PtxMinNode>(is);
}

// PtxMaxNode implementation
void PtxMaxNode::print() const {
    std::cout << "PtxMaxNode: Left ";
    if (_left) {
        std::cout << "non-empty: \n";
        _left->print();
    } else {
        std::cout << "empty. \n";
    }

    std::cout << "PtxMaxNode: Right ";
    if (_right) {
        std::cout << "non-empty: \n";
        _right->print();
    } else {
        std::cout << "empty. \n";
    }
}
std::unique_ptr<PtxAbstractNode> PtxMaxNode::eval(KLaunchConfig *config) {
    return BinaryEval(config, _left, _right,
                      [](int64_t a, int64_t b) { return std::max(a, b); });
}
void PtxMaxNode::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    if (idx == 0) {
        _left.swap(child);
        return;
    }
    if (idx == 1) {
        _right.swap(child);
        return;
    }

    throw std::runtime_error("Invalid index.");
}
void PtxMaxNode::serialize(std::ostream &os) const {
    serializeBinOp(_left, _right, os, PtxNodeKind::MaxOp);
}
std::unique_ptr<PtxAbstractNode> PtxMaxNode::create(std::istream &is) {
    return createBinOp<PtxMaxNode>(is);
}

// PtxShlNode implementation
void PtxShlNode::print() const {
    std::cout << "PtxShlNode: Left ";
    if (_left) {
        std::cout << "non-empty: \n";
        _left->print();
    } else {
        std::cout << "empty. \n";
    }

    std::cout << "PtxShlNode: Right ";
    if (_right) {
        std::cout << "non-empty: \n";
        _right->print();
    } else {
        std::cout << "empty. \n";
    }
}
std::unique_ptr<PtxAbstractNode> PtxShlNode::eval(KLaunchConfig *config) {
    return BinaryEval(config, _left, _right,
                      [](int64_t a, int64_t b) { return a << b; });
}
void PtxShlNode::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    if (idx == 0) {
        _left.swap(child);
        return;
    }
    if (idx == 1) {
        _right.swap(child);
        return;
    }

    throw std::runtime_error("Invalid index.");
}
void PtxShlNode::serialize(std::ostream &os) const {
    serializeBinOp(_left, _right, os, PtxNodeKind::ShlOp);
}
std::unique_ptr<PtxAbstractNode> PtxShlNode::create(std::istream &is) {
    return createBinOp<PtxShlNode>(is);
}

// PtxShrNode implementation
void PtxShrNode::print() const {
    std::cout << "PtxShrNode: Left ";
    if (_left) {
        std::cout << "non-empty: \n";
        _left->print();
    } else {
        std::cout << "empty. \n";
    }

    std::cout << "PtxShrNode: Right ";
    if (_right) {
        std::cout << "non-empty: \n";
        _right->print();
    } else {
        std::cout << "empty. \n";
    }
}
std::unique_ptr<PtxAbstractNode> PtxShrNode::eval(KLaunchConfig *config) {
    return BinaryEval(config, _left, _right,
                      [](int64_t a, int64_t b) { return a >> b; });
}
void PtxShrNode::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    if (idx == 0) {
        _left.swap(child);
        return;
    }
    if (idx == 1) {
        _right.swap(child);
        return;
    }

    throw std::runtime_error("Invalid index.");
}
void PtxShrNode::serialize(std::ostream &os) const {
    serializeBinOp(_left, _right, os, PtxNodeKind::ShrOp);
}
std::unique_ptr<PtxAbstractNode> PtxShrNode::create(std::istream &is) {
    return createBinOp<PtxShrNode>(is);
}

// PtxImmediate implementation
std::unique_ptr<PtxAbstractNode> PtxImmediate::eval(KLaunchConfig *config) {
    return std::make_unique<PtxImmediate>(this->_values, this->_type);
}
void PtxImmediate::print() const {
    std::cout << "PtxImmediate: Values ";
    for (auto value : _values) {
        std::cout << value << "\t";
    }
    std::cout << ".\n";
}
void PtxImmediate::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    throw std::runtime_error("Immediate has no children.");
}
void PtxImmediate::serialize(std::ostream &os) const {
    os << to_underlying(PtxNodeKind::Immediate) << " ";
    os << _values.size() << " ";
    for (auto value : _values) {
        os << value << " ";
    }
    os << to_underlying(_type) << " ";
}
std::unique_ptr<PtxAbstractNode> PtxImmediate::create(std::istream &is) {
    size_t size;
    is >> size;
    std::vector<int64_t> values(size);
    for (size_t i = 0; i < size; ++i) {
        is >> values[i];
    }
    int type_id;
    is >> type_id;

    return std::make_unique<PtxImmediate>(values, PtxParameterType(type_id));
}

// PtxSpecialRegister implementation
void PtxSpecialRegister::print() const {
    std::cout << "PtxSpecialRegister: Kind " << stringFromSpecialRegister(_kind)
              << ", dim " << _dim << "Values ";
    if (_values.empty()) {
        std::cout << "empty. \n";
    } else {
        for (auto value : _values) {
            std::cout <<  value << '\t';
        }
        std::cout << ".\n";
    }
}
std::unique_ptr<PtxAbstractNode>
PtxSpecialRegister::eval(KLaunchConfig *config) {
    if (!config) { // Cannot evaluate, as value not yet known
        return nullptr;
    }
    if (!_values.empty()) { // Value already filled
        return std::make_unique<PtxSpecialRegister>(_kind, _dim, _values);
    }

    std::vector<int64_t> values(1);
    if (_kind == SpecialRegisterKind::NThreadIds) {
        values[0] = config->blockDim[_dim];
    }
    if (_kind == SpecialRegisterKind::NCTAIds) {
        values[0] = config->gridDim[_dim];
    }
    if (_kind == SpecialRegisterKind::ThreadId) {
        values.resize(config->blockDim[_dim]);
        std::iota(values.begin(), values.end(), 0);
    }
    if (_kind == SpecialRegisterKind::CTAId) {
        values.resize(config->gridDim[_dim]);
        std::iota(values.begin(), values.end(), 0);
    }

    return std::make_unique<PtxSpecialRegister>(_kind, _dim, values);
}
void PtxSpecialRegister::set_child(std::unique_ptr<PtxAbstractNode> child,
                                   int idx) {
    throw std::runtime_error("PtxSpecialRegister has no children.");
}
void PtxSpecialRegister::serialize(std::ostream &os) const {
    os << to_underlying(PtxNodeKind::SpecialRegister) << " ";
    os << to_underlying(_kind) << " ";
    os << _dim << " ";
    os << _values.size() << " ";
    for (auto value : _values)
        os << value << " ";
}
std::unique_ptr<PtxAbstractNode>
PtxSpecialRegister::create(std::istream &is) {
    int kind_id, dim;
    is >> kind_id;
    is >> dim;
    size_t size;
    is >> size;
    std::vector<int64_t> values(size);
    for (size_t i = 0; i < size; ++i)
        is >> values[i];

    return std::make_unique<PtxSpecialRegister>(SpecialRegisterKind(kind_id),
                                                dim, values);
}

// PtxCvtaNode implementation
void PtxCvtaNode::print() const {
    std::cout << "PtxCvtaNode. \n";
    if (_dst)
        _dst->print();
}
std::unique_ptr<PtxAbstractNode> PtxCvtaNode::eval(KLaunchConfig *config) {
    if(_dst)
        return _dst->eval(config);
}
void PtxCvtaNode::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    if (idx != 0)
        throw std::runtime_error("Invalid index.");
    _dst.swap(child);
}
void PtxCvtaNode::serialize(std::ostream &os) const {
    os << to_underlying(PtxNodeKind::Cvta) << " ";
    _dst->serialize(os);
}
std::unique_ptr<PtxAbstractNode> PtxCvtaNode::create(std::istream &is) {
    std::unique_ptr<PtxAbstractNode> child = PtxAbstractNode::unserialize(is);
    return std::make_unique<PtxCvtaNode>(std::move(child));
}

/*
 * PTX TREE IMPLEMENTATION
 */

void PtxTree::print() const { if(_root) _root->print(); }

std::pair<PtxTree::node_type *, int>
PtxTree::find_register_node(const std::string &register_name) {
    if (auto ret = _registers_to_leafs.find(register_name);
        ret != _registers_to_leafs.end()) {
        return ret->second;
    } else {
        // Register is not of interest
        return {nullptr, 0};
    }
}

void PtxTree::replace_register(const std::string &old_register,
                               const std::string &new_register) {
    if (auto ret = _registers_to_leafs.find(old_register);
        ret != _registers_to_leafs.end()) {
        _registers_to_leafs[new_register] = ret->second;
        _registers_to_leafs.erase(ret);
    }
}
void PtxTree::add_node(std::unique_ptr<PtxTree::node_type> new_node,
                       const std::vector<PtxOperand> &operands) {
    if (operands[0].kind != PtxOperandKind::Register)
        throw std::runtime_error("Destination must be register!");

    if (auto [par_node, idx] = find_register_node(operands[0].name); par_node) {
        for (uint64_t i = 0; i < operands.size() - 1; ++i) {
            auto &operand = operands[i + 1];

            switch (operand.kind) {
            case PtxOperandKind::Register:
                _registers_to_leafs[operand.name] = {new_node.get(), i};
                break;
            case PtxOperandKind::Parameter:
                new_node->set_child(
                    std::make_unique<PtxParameter>(operand.name, operand.value),
                    static_cast<int>(i));
                break;
            case PtxOperandKind::Immediate:
                new_node->set_child(
                    std::make_unique<PtxImmediate>(operand.value, s64),
                    static_cast<int>(i));
                break;
            default:
                new_node->set_child(std::make_unique<PtxSpecialRegister>(
                                        operand.kind, operand.value),
                                    static_cast<int>(i));
                break;
            }
        }

        par_node->set_child(std::move(new_node), idx);
    }
}
std::unique_ptr<PtxAbstractNode> PtxTree::eval(KLaunchConfig *config) {
    return _root->eval(config);
}

std::unique_ptr<PtxTree::node_type> produceNode(PtxNodeKind kind) {
    switch (kind) {
    case PtxNodeKind::AddOp:
        return std::make_unique<PtxAddNode>(nullptr, nullptr);
    case PtxNodeKind::SubOp:
        return std::make_unique<PtxSubNode>(nullptr, nullptr);
    case PtxNodeKind::MulOp:
        return std::make_unique<PtxMulNode>(nullptr, nullptr);
    case PtxNodeKind::AbdOp:
        return std::make_unique<PtxAbdNode>(nullptr, nullptr);
    case PtxNodeKind::MadOp:
        return std::make_unique<PtxMadNode>(nullptr, nullptr, nullptr);
    case PtxNodeKind::SadOp:
        return std::make_unique<PtxSadNode>(nullptr, nullptr, nullptr);
    case PtxNodeKind::DivOp:
        return std::make_unique<PtxDivNode>(nullptr, nullptr);
    case PtxNodeKind::RemOp:
        return std::make_unique<PtxRemNode>(nullptr, nullptr);
    case PtxNodeKind::AbsOp:
        return std::make_unique<PtxAbsNode>(nullptr);
    case PtxNodeKind::NegOp:
        return std::make_unique<PtxNegNode>(nullptr);
    case PtxNodeKind::MinOp:
        return std::make_unique<PtxMinNode>(nullptr, nullptr);
    case PtxNodeKind::MaxOp:
        return std::make_unique<PtxMaxNode>(nullptr, nullptr);
    case PtxNodeKind::ShlOp:
        return std::make_unique<PtxShlNode>(nullptr, nullptr);
    case PtxNodeKind::ShrOp:
        return std::make_unique<PtxShrNode>(nullptr, nullptr);
    default:
        throw std::runtime_error("Invalid Operation.");
    }
}

/*
 * TYPE SERIALIZATION
 */

inline std::string stringFromNodeKind(PtxNodeKind kind) {
    switch (kind) {
    case PtxNodeKind::Parameter:
        return "Parameter";
    case PtxNodeKind::Immediate:
        return "Immediate";
    case PtxNodeKind::SpecialRegister:
        return "SpecialRegister";
    case PtxNodeKind::Cvta:
        return "Cvta";
    case PtxNodeKind::AddOp:
        return "AddOp";
    case PtxNodeKind::SubOp:
        return "SubOp";
    case PtxNodeKind::MulOp:
        return "MulOp";
    case PtxNodeKind::DivOp:
        return "DivOp";
    case PtxNodeKind::RemOp:
        return "RemOp";
    case PtxNodeKind::AbsOp:
        return "AbsOp";
    case PtxNodeKind::NegOp:
        return "NegOp";
    case PtxNodeKind::MinOp:
        return "MinOp";
    case PtxNodeKind::MaxOp:
        return "MaxOp";
    case PtxNodeKind::ShrOp:
        return "ShrOp";
    case PtxNodeKind::ShlOp:
        return "ShlOp";
    case PtxNodeKind::MadOp:
        return "MadOp";
    case PtxNodeKind::SadOp:
        return "SadOp";
    case PtxNodeKind::LdOp:
        return "LdOp";
    case PtxNodeKind::AbdOp:
        return "AbdOp";
    case PtxNodeKind::MoveOp:
        return "MoveOp";
    case PtxNodeKind::Register:
        return "Register";
    case PtxNodeKind::InvalidOp:
        return "InvalidOp";
    }
}

std::map<std::string, PtxNodeKind> &nodeKindFromString() {
    static std::map<std::string, PtxNodeKind> map_ = {
        {"Parameter", PtxNodeKind::Parameter},
        {"Immediate", PtxNodeKind::Immediate},
        {"SpecialRegister", PtxNodeKind::SpecialRegister},
        {"Cvta", PtxNodeKind::Cvta},
        {"AddOp", PtxNodeKind::AddOp},
        {"SubOp", PtxNodeKind::SubOp},
        {"MulOp", PtxNodeKind::MulOp},
        {"DivOp", PtxNodeKind::DivOp},
        {"RemOp", PtxNodeKind::RemOp},
        {"AbsOp", PtxNodeKind::AbsOp},
        {"NegOp", PtxNodeKind::NegOp},
        {"MinOp", PtxNodeKind::MinOp},
        {"MaxOp", PtxNodeKind::MaxOp},
        {"ShrOp", PtxNodeKind::ShrOp},
        {"ShlOp", PtxNodeKind::ShlOp},
        {"MadOp", PtxNodeKind::MadOp},
        {"SadOp", PtxNodeKind::SadOp},
        {"LdOp", PtxNodeKind::LdOp},
        {"AbdOp", PtxNodeKind::AbdOp},
        {"MoveOp", PtxNodeKind::MoveOp},
        {"Register", PtxNodeKind::Register},
        {"InvalidOp", PtxNodeKind::InvalidOp}};
    return map_;
}

std::string stringFromSpecialRegister(SpecialRegisterKind kind) {
    switch (kind) {
    case SpecialRegisterKind::ThreadId:
        return "ThreadId";
    case SpecialRegisterKind::NThreadIds:
        return "NThreadIds";
    case SpecialRegisterKind::CTAId:
        return "CTAId";
    case SpecialRegisterKind::NCTAIds:
        return "NCTAIDs";
    }
}

std::unique_ptr<PtxAbstractNode>
PtxAbstractNode::unserialize(std::istream &is) {
    static std::map<PtxNodeKind, Factory> fac_map = {
        {PtxNodeKind::Parameter, PtxParameter::create},
        {PtxNodeKind::Immediate, PtxImmediate::create},
        {PtxNodeKind::SpecialRegister, PtxSpecialRegister::create},
        {PtxNodeKind::Cvta, PtxCvtaNode::create},
        {PtxNodeKind::AddOp, PtxAddNode::create},
        {PtxNodeKind::SubOp, PtxSubNode::create},
        {PtxNodeKind::MulOp, PtxMulNode::create},
        {PtxNodeKind::AbdOp, PtxAbdNode::create},
        {PtxNodeKind::MadOp, PtxMadNode::create},
        {PtxNodeKind::SadOp, PtxSadNode::create},
        {PtxNodeKind::DivOp, PtxDivNode::create},
        {PtxNodeKind::RemOp, PtxRemNode::create},
        {PtxNodeKind::AbsOp, PtxAbsNode::create},
        {PtxNodeKind::NegOp, PtxNegNode::create},
        {PtxNodeKind::MinOp, PtxMinNode::create},
        {PtxNodeKind::MaxOp, PtxMaxNode::create},
        {PtxNodeKind::ShlOp, PtxShlNode::create},
        {PtxNodeKind::ShrOp, PtxShrNode::create},
        {PtxNodeKind::InvalidOp, null_factory}
    };

    int type_id;
    is >> type_id;
    if(type_id >= 0)
        return fac_map[PtxNodeKind(type_id)](is);
    else
        return fac_map[PtxNodeKind::InvalidOp](is);
}
} // namespace PtxTreeParser