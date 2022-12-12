#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <regex>
#include <sstream>

#include "ptx_tree.h"

namespace PtxTreeParser {
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
            std::cout << std::hex << val << '\t';
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
    std::unique_ptr<PtxAbstractNode> left_eval(_left->eval(config));
    std::unique_ptr<PtxAbstractNode> right_eval(_right->eval(config));

    if (!left_eval && !right_eval) {
        // Could not collapse anything, no evaluation possible
        return nullptr;
    }
    if (!left_eval && right_eval) {
        // Could only collapse rigt node, no evaluation possible
        _right.swap(right_eval);
        return nullptr;
    }
    if (left_eval && !right_eval) {
        // Could only collapse left node, no evaluation possible
        _left.swap(left_eval);
        return nullptr;
    }

    PtxNodeKind left_kind = left_eval->get_kind();
    PtxNodeKind right_kind = right_eval->get_kind();

    // One node is immediate and one is parameter
    auto add_par_imm = [](PtxParameter &par, const PtxImmediate &imm) {
        std::vector<int> &offsets(par.get_offsets());
        int imm_value = static_cast<int>(imm.get_value());

        for (auto &offset : offsets) {
            offset += imm_value;
        }
    };
    if (left_kind == PtxNodeKind::Parameter &&
        right_kind == PtxNodeKind::Immediate) {
        add_par_imm(static_cast<PtxParameter &>(*left_eval),
                    static_cast<PtxImmediate &>(*right_eval));
        return left_eval;
    }
    if (right_kind == PtxNodeKind::Parameter &&
        left_kind == PtxNodeKind::Immediate) {
        add_par_imm(static_cast<PtxParameter &>(*right_eval),
                    static_cast<PtxImmediate &>(*left_eval));
        return right_eval;
    }

    // Both are immediate
    if (left_kind == PtxNodeKind::Immediate &&
        right_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxImmediate &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));

        left.get_value() += right.get_value();
        return left_eval;
    }

    // One is immediate and one is special register
    auto add_spec_imm = [](PtxSpecialRegister &reg, PtxImmediate &imm) {
        std::vector<int64_t> &values(reg.get_values());
        int64_t imm_value = imm.get_value();
        for (auto &value : values) {
            value += imm_value;
        }

        return values;
    };
    if (left_kind == PtxNodeKind::SpecialRegister &&
        right_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxSpecialRegister &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));
        return std::make_unique<PtxImmediateArray>(add_spec_imm(left, right),
                                                   right.get_type());
    }
    if (right_kind == PtxNodeKind::SpecialRegister &&
        left_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxImmediate &>(*left_eval));
        auto &right(static_cast<PtxSpecialRegister &>(*right_eval));
        return std::make_unique<PtxImmediateArray>(add_spec_imm(right, left),
                                                   left.get_type());
    }

    throw std::runtime_error("Addition of these nodes not supported.");
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
    std::unique_ptr<PtxAbstractNode> left_eval(_left->eval(config));
    std::unique_ptr<PtxAbstractNode> right_eval(_right->eval(config));

    if (!left_eval && !right_eval) {
        // Could not collapse anything, no evaluation possible
        return nullptr;
    }
    if (!left_eval && right_eval) {
        // Could only collapse rigt node, no evaluation possible
        _right.swap(right_eval);
        return nullptr;
    }
    if (left_eval && !right_eval) {
        // Could only collapse left node, no evaluation possible
        _left.swap(left_eval);
        return nullptr;
    }

    PtxNodeKind left_kind = left_eval->get_kind();
    PtxNodeKind right_kind = right_eval->get_kind();

    // left is param, right immediate
    auto sub_par_imm = [](PtxParameter &par, const PtxImmediate &imm) {
        std::vector<int> &offsets(par.get_offsets());
        int imm_value = static_cast<int>(imm.get_value());

        for (auto &offset : offsets) {
            offset -= imm_value;
        }
    };
    if (left_kind == PtxNodeKind::Parameter &&
        right_kind == PtxNodeKind::Immediate) {
        sub_par_imm(static_cast<PtxParameter &>(*left_eval),
                    static_cast<PtxImmediate &>(*right_eval));
        return left_eval;
    }

    // Both are immediate
    if (left_kind == PtxNodeKind::Immediate &&
        right_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxImmediate &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));

        left.get_value() -= right.get_value();
        return left_eval;
    }

    // One is immediate and one is special register
    auto sub_spec_imm = [](PtxSpecialRegister &reg, PtxImmediate &imm) {
        std::vector<int64_t> &values(reg.get_values());
        int64_t imm_value = imm.get_value();
        for (auto &value : values) {
            value -= imm_value;
        }

        return values;
    };
    if (left_kind == PtxNodeKind::SpecialRegister &&
        right_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxSpecialRegister &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));
        return std::make_unique<PtxImmediateArray>(sub_spec_imm(left, right),
                                                   right.get_type());
    }

    throw std::runtime_error("Subtraction of these nodes not supported.");
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
    std::unique_ptr<PtxAbstractNode> left_eval(_left->eval(config));
    std::unique_ptr<PtxAbstractNode> right_eval(_right->eval(config));

    if (!left_eval && !right_eval) {
        // Could not collapse anything, no evaluation possible
        return nullptr;
    }
    if (!left_eval && right_eval) {
        // Could only collapse rigt node, no evaluation possible
        _right.swap(right_eval);
        return nullptr;
    }
    if (left_eval && !right_eval) {
        // Could only collapse left node, no evaluation possible
        _left.swap(left_eval);
        return nullptr;
    }

    PtxNodeKind left_kind = left_eval->get_kind();
    PtxNodeKind right_kind = right_eval->get_kind();

    // Both are immediate
    if (left_kind == PtxNodeKind::Immediate &&
        right_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxImmediate &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));

        left.get_value() *= right.get_value();
        return left_eval;
    }

    throw std::runtime_error("Multiplication of these nodes not supported.");
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
    std::unique_ptr<PtxAbstractNode> left_eval(_left->eval(config));
    std::unique_ptr<PtxAbstractNode> right_eval(_right->eval(config));

    if (!left_eval && !right_eval) {
        // Could not collapse anything, no evaluation possible
        return nullptr;
    }
    if (!left_eval && right_eval) {
        // Could only collapse rigt node, no evaluation possible
        _right.swap(right_eval);
        return nullptr;
    }
    if (left_eval && !right_eval) {
        // Could only collapse left node, no evaluation possible
        _left.swap(left_eval);
        return nullptr;
    }

    PtxNodeKind left_kind = left_eval->get_kind();
    PtxNodeKind right_kind = right_eval->get_kind();

    // Both are immediate
    if (left_kind == PtxNodeKind::Immediate &&
        right_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxImmediate &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));

        left.get_value() /= right.get_value();
        return left_eval;
    }

    throw std::runtime_error("Division of these nodes not supported.");
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
    std::unique_ptr<PtxAbstractNode> left_eval(_left->eval(config));
    std::unique_ptr<PtxAbstractNode> right_eval(_right->eval(config));

    if (!left_eval && !right_eval) {
        // Could not collapse anything, no evaluation possible
        return nullptr;
    }
    if (!left_eval && right_eval) {
        // Could only collapse rigt node, no evaluation possible
        _right.swap(right_eval);
        return nullptr;
    }
    if (left_eval && !right_eval) {
        // Could only collapse left node, no evaluation possible
        _left.swap(left_eval);
        return nullptr;
    }

    PtxNodeKind left_kind = left_eval->get_kind();
    PtxNodeKind right_kind = right_eval->get_kind();


    auto rem_par_imm = [](PtxParameter &par, const PtxImmediate &imm) {
        std::vector<int> &offsets(par.get_offsets());
        int imm_value = static_cast<int>(imm.get_value());

        for (auto &offset : offsets) {
            offset %= imm_value;
        }
    };
    if (left_kind == PtxNodeKind::Parameter &&
        right_kind == PtxNodeKind::Immediate) {
        rem_par_imm(static_cast<PtxParameter &>(*left_eval),
                    static_cast<PtxImmediate &>(*right_eval));
        return left_eval;
    }

    // Both are immediate
    if (left_kind == PtxNodeKind::Immediate &&
        right_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxImmediate &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));

        left.get_value() %= right.get_value();
        return left_eval;
    }

    // One is immediate and one is special register
    auto rem_spec_imm = [](PtxSpecialRegister &reg, PtxImmediate &imm) {
        std::vector<int64_t> &values(reg.get_values());
        int64_t imm_value = imm.get_value();
        for (auto &value : values) {
            value %= imm_value;
        }

        return values;
    };
    if (left_kind == PtxNodeKind::SpecialRegister &&
        right_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxSpecialRegister &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));
        return std::make_unique<PtxImmediateArray>(rem_spec_imm(left, right),
                                                   right.get_type());
    }



    throw std::runtime_error("Remainder of these nodes not supported.");
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
    std::unique_ptr<PtxAbstractNode> left_eval(_left->eval(config));
    std::unique_ptr<PtxAbstractNode> right_eval(_right->eval(config));

    if (!left_eval && !right_eval) {
        // Could not collapse anything, no evaluation possible
        return nullptr;
    }
    if (!left_eval && right_eval) {
        // Could only collapse rigt node, no evaluation possible
        _right.swap(right_eval);
        return nullptr;
    }
    if (left_eval && !right_eval) {
        // Could only collapse left node, no evaluation possible
        _left.swap(left_eval);
        return nullptr;
    }

    PtxNodeKind left_kind = left_eval->get_kind();
    PtxNodeKind right_kind = right_eval->get_kind();

    // left is param, right immediate
    auto abd_par_imm = [](PtxParameter &par, const PtxImmediate &imm) {
        std::vector<int> &offsets(par.get_offsets());
        int imm_value = static_cast<int>(imm.get_value());

        for (auto &offset : offsets) {
            offset -= imm_value;
            if (offset < 0) offset = -offset;
        }
    };
    if (left_kind == PtxNodeKind::Parameter &&
        right_kind == PtxNodeKind::Immediate) {
        abd_par_imm(static_cast<PtxParameter &>(*left_eval),
                    static_cast<PtxImmediate &>(*right_eval));
        return left_eval;
    }
    if (left_kind == PtxNodeKind::Immediate &&
        right_kind == PtxNodeKind::Parameter) {
        abd_par_imm(static_cast<PtxParameter &>(*right_eval),
                    static_cast<PtxImmediate &>(*left_eval));
        return right_eval;
    }

    // Both are immediate
    if (left_kind == PtxNodeKind::Immediate &&
        right_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxImmediate &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));

        left.get_value() -= right.get_value();
        if (left.get_value() < 0) left.get_value() = -left.get_value();
        return left_eval;
    }

    if (left_kind == PtxNodeKind::Parameter &&
        right_kind == PtxNodeKind::Parameter) {
        auto &left(static_cast<PtxParameter &>(*left_eval));
        auto &right(static_cast<PtxParameter &>(*right_eval));

        std::vector<int> &offsets_left(left.get_offsets());
        std::vector<int> &offsets_right(right.get_offsets());
        if (offsets_left.size() != offsets_right.size()) {
            throw std::runtime_error("Not the same amount of offsets to calculate the absolute difference");
        }

        for (unsigned i = 0; i < offsets_left.size(); ++i) {
            offsets_left[i] -= offsets_right[i];
            if (offsets_left[i] < 0) {
                offsets_left[i] = -offsets_left[i];
            }
        }

        return left_eval;
    }

    // One is immediate and one is special register
    auto abd_spec_imm = [](PtxSpecialRegister &reg, PtxImmediate &imm) {
        std::vector<int64_t> &values(reg.get_values());
        int64_t imm_value = imm.get_value();
        for (auto &value : values) {
            value -= imm_value;
            if (value < 0) value = -value;
        }

        return values;
    };
    if (left_kind == PtxNodeKind::SpecialRegister &&
        right_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxSpecialRegister &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));
        return std::make_unique<PtxImmediateArray>(abd_spec_imm(left, right),
                                                   right.get_type());
    }

    if (left_kind == PtxNodeKind::Immediate &&
        right_kind == PtxNodeKind::SpecialRegister) {
        auto &left(static_cast<PtxImmediate &>(*left_eval));
        auto &right(static_cast<PtxSpecialRegister &>(*right_eval));
        return std::make_unique<PtxImmediateArray>(abd_spec_imm(right, left),
                                                   left.get_type());
    }

    if (left_kind == PtxNodeKind::SpecialRegister &&
        right_kind == PtxNodeKind::SpecialRegister) {
        auto &left(static_cast<PtxSpecialRegister &>(*left_eval));
        auto &right(static_cast<PtxSpecialRegister &>(*right_eval));
        std::vector<int64_t> &values_left(left.get_values());
        std::vector<int64_t> &values_right(right.get_values());
        if (values_left.size() != values_right.size()) {
            throw std::runtime_error("Not the same amount of offsets to calculate the absolute difference");
        }

        for (unsigned i = 0; i < values_left.size(); ++i) {
            values_left[i] -= values_right[i];
            if (values_left[i] < 0) {
                values_left[i] = -values_left[i];
            }
        }

        return std::make_unique<PtxImmediateArray>(values_left,
                                                   PtxParameterType::s64);
    }

    throw std::runtime_error("Absolute difference of these nodes not supported.");
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
        if (child.get_value() < 0)  {
            child.get_value() = -child.get_value();
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
        child.get_value() = -child.get_value();
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
    std::unique_ptr<PtxAbstractNode> left_eval(_left->eval(config));
    std::unique_ptr<PtxAbstractNode> right_eval(_right->eval(config));

    if (!left_eval && !right_eval) {
        // Could not collapse anything, no evaluation possible
        return nullptr;
    }
    if (!left_eval && right_eval) {
        // Could only collapse rigt node, no evaluation possible
        _right.swap(right_eval);
        return nullptr;
    }
    if (left_eval && !right_eval) {
        // Could only collapse left node, no evaluation possible
        _left.swap(left_eval);
        return nullptr;
    }

    PtxNodeKind left_kind = left_eval->get_kind();
    PtxNodeKind right_kind = right_eval->get_kind();


    // Both are immediate
    if (left_kind == PtxNodeKind::Immediate &&
        right_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxImmediate &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));

        left.get_value() = std::min(left.get_value(), right.get_value());
        return left_eval;
    }

    throw std::runtime_error("Minimum of these nodes not supported.");
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
    std::unique_ptr<PtxAbstractNode> left_eval(_left->eval(config));
    std::unique_ptr<PtxAbstractNode> right_eval(_right->eval(config));

    if (!left_eval && !right_eval) {
        // Could not collapse anything, no evaluation possible
        return nullptr;
    }
    if (!left_eval && right_eval) {
        // Could only collapse rigt node, no evaluation possible
        _right.swap(right_eval);
        return nullptr;
    }
    if (left_eval && !right_eval) {
        // Could only collapse left node, no evaluation possible
        _left.swap(left_eval);
        return nullptr;
    }

    PtxNodeKind left_kind = left_eval->get_kind();
    PtxNodeKind right_kind = right_eval->get_kind();


    // Both are immediate
    if (left_kind == PtxNodeKind::Immediate &&
        right_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxImmediate &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));

        left.get_value() = std::max(left.get_value(), right.get_value());
        return left_eval;
    }

    throw std::runtime_error("Maximum of these nodes not supported.");
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
    std::unique_ptr<PtxAbstractNode> left_eval(_left->eval(config));
    std::unique_ptr<PtxAbstractNode> right_eval(_right->eval(config));

    if (!left_eval && !right_eval) {
        // Could not collapse anything, no evaluation possible
        return nullptr;
    }
    if (!left_eval && right_eval) {
        // Could only collapse rigt node, no evaluation possible
        _right.swap(right_eval);
        return nullptr;
    }
    if (left_eval && !right_eval) {
        // Could only collapse left node, no evaluation possible
        _left.swap(left_eval);
        return nullptr;
    }

    PtxNodeKind left_kind = left_eval->get_kind();
    PtxNodeKind right_kind = right_eval->get_kind();

    // Both are immediate
    if (left_kind == PtxNodeKind::Immediate &&
        right_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxImmediate &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));

        left.get_value() <<= right.get_value();
        return left_eval;
    }

    throw std::runtime_error("Shift-left of these nodes not supported.");
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
    std::unique_ptr<PtxAbstractNode> left_eval(_left->eval(config));
    std::unique_ptr<PtxAbstractNode> right_eval(_right->eval(config));

    if (!left_eval && !right_eval) {
        // Could not collapse anything, no evaluation possible
        return nullptr;
    }
    if (!left_eval && right_eval) {
        // Could only collapse rigt node, no evaluation possible
        _right.swap(right_eval);
        return nullptr;
    }
    if (left_eval && !right_eval) {
        // Could only collapse left node, no evaluation possible
        _left.swap(left_eval);
        return nullptr;
    }

    PtxNodeKind left_kind = left_eval->get_kind();
    PtxNodeKind right_kind = right_eval->get_kind();

    // Both are immediate
    if (left_kind == PtxNodeKind::Immediate &&
        right_kind == PtxNodeKind::Immediate) {
        auto &left(static_cast<PtxImmediate &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));

        left.get_value() >>= right.get_value();
        return left_eval;
    }

    throw std::runtime_error("Shift-right of these nodes not supported.");
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


// PtxImmediate implementation
std::unique_ptr<PtxAbstractNode> PtxImmediate::eval(KLaunchConfig *config) {
    return std::make_unique<PtxImmediate>(this->_value, this->_type);
}
void PtxImmediate::print() const {
    std::cout << "PtxImmediate: Value " << _value << ".\n";
}
void PtxImmediate::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    throw std::runtime_error("Immediate has no children.");
}

// PtxImmediateArray implementation
void PtxImmediateArray::print() const {
    std::cout << "PtxImmediateArray:"
              << "Values ";
    for (auto value : this->_values) {
        std::cout << std::hex << value << '\t';
    }
    std::cout << ".\n";
}
std::unique_ptr<PtxAbstractNode>
PtxImmediateArray::eval(KLaunchConfig *config) {
    return std::make_unique<PtxImmediateArray>(this->_values, this->_type);
}
void PtxImmediateArray::set_child(std::unique_ptr<PtxAbstractNode> child,
                                  int idx) {
    throw std::runtime_error("PtxImmediateArray has no children.");
}

// PtxSpecialRegister implementation
void PtxSpecialRegister::print() const {
    std::cout << "PtxSpecialRegister: Kind " << stringFromSpecialRegister(_kind)
              << ", dim " << _dim << "Values ";
    if (_values.empty()) {
        std::cout << "empty. \n";
    } else {
        for (auto value : _values) {
            std::cout << std::hex << value << '\t';
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

// PtxCvtaNode implementation
void PtxCvtaNode::print() const {
    std::cout << "PtxCvtaNode. \n";
    if(_dst) _dst->print();
}
std::unique_ptr<PtxAbstractNode> PtxCvtaNode::eval(KLaunchConfig *config) {
    return _dst->eval(config);
}
void PtxCvtaNode::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    if (idx != 0)
        throw std::runtime_error("Invalid index.");
    _dst.swap(child);
}

/*
 * PTX TREE IMPLEMENTATION
 */

void PtxTree::print() const { _root->print(); }

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
    if(operands[0].kind != PtxOperandKind::Register)
        throw std::runtime_error("Destination must be register!");

    if (auto [par_node, idx] = find_register_node(operands[0].name); par_node) {
        for (uint64_t i = 0; i < operands.size()-1; ++i) {
            auto& operand = operands[i+1];

            switch (operand.kind) {
            case PtxOperandKind::Register:
                _registers_to_leafs[operand.name] = {new_node.get(), i};
                break;
            case PtxOperandKind::Parameter:
                new_node->set_child(std::make_unique<PtxParameter>(operand.name, operand.offset), static_cast<int>(i));
                break;
            case PtxOperandKind::Immediate:
                new_node->set_child(std::make_unique<PtxImmediate>(operand.offset, s64), static_cast<int>(i));
                break;
            default:
                new_node->set_child(std::make_unique<PtxSpecialRegister>(operand.kind, operand.offset), static_cast<int>(i));
                break;
            }
        }

        par_node->set_child(std::move(new_node), idx);
    }
}
std::unique_ptr<PtxAbstractNode> PtxTree::eval(KLaunchConfig *config) {
    return _root->eval(config);
}

/*
 * TYPE SERIALIZATION
 */

std::string stringFromNodeKind(PtxNodeKind kind) {
    switch (kind) {
    case PtxNodeKind::Parameter:
        return "Parameter";
    case PtxNodeKind::Immediate:
        return "Immediate";
    case PtxNodeKind::ImmediateArray:
        return "ImmediateArray";
    case PtxNodeKind::SpecialRegister:
        return "SpecialRegister";
    case PtxNodeKind::Cvta:
        return "Cvta";
    case PtxNodeKind::AddOp:
        return "AddOp";
    case PtxNodeKind::MoveOp:
        return "MoveOp";
    case PtxNodeKind::Register:
        return "Register";
    case PtxNodeKind::invalidOp:
        return "InvalidOp";
    }
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

std::map<std::string, PtxNodeKind> &getStrToPtxNodeKind() {
    static std::map<std::string, PtxNodeKind> map_ = {
        {"add", PtxNodeKind::AddOp},
        {"mov", PtxNodeKind::MoveOp},
    };
    return map_;
}

} // namespace PtxTreeParser