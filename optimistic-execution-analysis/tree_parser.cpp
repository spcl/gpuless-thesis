#include "tree_parser.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <regex>
#include <sstream>

namespace PtxTreeParser {

std::map<std::string, PtxNodeKind> &getStrToPtxNodeKind() {
    static std::map<std::string, PtxNodeKind> map_ = {
        {"add", AddOp},
        {"mov", MoveOp},
    };
    return map_;
}

std::map<std::string, PtxParameterType> &getStrToPtxParameterType() {
    static std::map<std::string, PtxParameterType> map_ = {
        {"s8", s8},     {"s16", s16},     {"s32", s32}, {"s64", s64},
        {"u8", u8},     {"u16", u16},     {"u32", u32}, {"u64", u64},
        {"f16", f16},   {"f16x2", f16x2}, {"f32", f32}, {"f64", f64},
        {"b8", b8},     {"b16", b16},     {"b32", b32}, {"b64", b64},
        {"pred", pred},
    };
    return map_;
}

std::map<PtxParameterType, std::string> &getPtxParameterTypeToStr() {
    static std::map<PtxParameterType, std::string> map_ = {
        {s8, "s8"},     {s16, "s16"},     {s32, "s32"}, {s64, "s64"},
        {u8, "u8"},     {u16, "u16"},     {u32, "u32"}, {u64, "u64"},
        {f16, "f16"},   {f16x2, "f16x2"}, {f32, "f32"}, {f64, "f64"},
        {b8, "b8"},     {b16, "b16"},     {b32, "b32"}, {b64, "b64"},
        {pred, "pred"},
    };
    return map_;
}

std::map<PtxParameterType, int> &getPtxParameterTypeToSize() {
    static std::map<PtxParameterType, int> map_ = {
        {s8, 1},  {s16, 2}, {s32, 4}, {s64, 8},   {u8, 1},   {u16, 2},
        {u32, 4}, {u64, 8}, {f16, 2}, {f16x2, 4}, {f32, 4},  {f64, 8},
        {b8, 1},  {b16, 2}, {b32, 4}, {b64, 8},   {pred, 0},
    };
    return map_;
}

PtxParameterType ptxParameterTypeFromString(const std::string &str) {
    auto it = getStrToPtxParameterType().find(str);
    if (it == getStrToPtxParameterType().end()) {
        return PtxParameterType::invalid;
    }
    return it->second;
}

PtxNodeKind ptxOpCodeFromString(const std::string &str) {
    auto it = getStrToPtxNodeKind().find(str);
    if (it == getStrToPtxNodeKind().end()) {
        return PtxNodeKind ::invalidOp;
    }
    return it->second;
}

int byteSizePtxParameterType(PtxParameterType type) {
    auto it = getPtxParameterTypeToSize().find(type);
    if (it == getPtxParameterTypeToSize().end()) {
        return -1;
    }
    return it->second;
}

std::vector<std::string> split_string(std::string str,
                                      const std::string &delimiter) {
    std::vector<std::string> result;

    size_t pos = 0;
    std::string token;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        token = str.substr(0, pos);
        result.push_back(token);
        str.erase(0, pos + delimiter.length());
    }
    result.push_back(str);
    return result;
}

bool startsWith(const std::string &str, const std::string &prefix) {
    return str.rfind(prefix, 0) == 0;
}

bool endsWith(const std::string &str, const std::string &suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool rgetline(std::string::reverse_iterator &it,
              const std::string::reverse_iterator &end, std::string &line) {
    if (it == end)
        return false;
    std::stringstream ss;
    char t;
    while (it != end && (t = *it++) != '\n') {
        ss << t;
    }
    line = ss.str();
    std::reverse(line.begin(), line.end());
    return true;
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
    if (left_kind == Parameter && right_kind == Immediate) {
        add_par_imm(static_cast<PtxParameter &>(*left_eval),
                    static_cast<PtxImmediate &>(*right_eval));
        return left_eval;
    }
    if (right_kind == Parameter && left_kind == Immediate) {
        add_par_imm(static_cast<PtxParameter &>(*right_eval),
                    static_cast<PtxImmediate &>(*left_eval));
        return right_eval;
    }

    // Both are immediate
    if (left_kind == Immediate && right_kind == Immediate) {
        auto &left(static_cast<PtxImmediate &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));

        left.get_value() += right.get_value();
        return left_eval;
    }

    // One is immediate and one is special register
    auto add_spec_imm = [](PtxSpecialRegister &reg, PtxImmediate &imm) {
        std::vector<int64_t> &values(reg.get_values());
        uint64_t imm_value = imm.get_value();
        for (auto &value : values) {
            value += imm_value;
        }

        return values;
    };
    if (left_kind == SpecialRegister && right_kind == Immediate) {
        auto &left(static_cast<PtxSpecialRegister &>(*left_eval));
        auto &right(static_cast<PtxImmediate &>(*right_eval));
        return std::make_unique<PtxImmediateArray>(add_spec_imm(left, right),
                                                   right.get_type());
    }
    if (right_kind == SpecialRegister && left_kind == Immediate) {
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

// PtxImmediate implementation
std::unique_ptr<PtxAbstractNode> PtxImmediate::eval(KLaunchConfig *config) {
    return std::make_unique<PtxImmediate>(this->_value, this->_type);
}
void PtxImmediate::print() const {
    std::cout << "PtxImmediate: Value " << _value << ", Type "
              << getPtxParameterTypeToSize()[_type] << ".\n";
}
void PtxImmediate::set_child(std::unique_ptr<PtxAbstractNode> child, int idx) {
    throw std::runtime_error("Immediate has no children.");
}

// PtxImmediateArray implementation
void PtxImmediateArray::print() const {
    std::cout << "PtxImmediateArray: Type "
              << getPtxParameterTypeToSize()[_type] << ", Values ";
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
    std::cout << "PtxSpecialRegister: Kind " << _kind << ", dim " << _dim
              << "Values ";
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
    if (_kind == NThreadIds) {
        values[0] = config->blockDim[_dim];
    }
    if (_kind == NCTAIds) {
        values[0] = config->gridDim[_dim];
    }
    if (_kind == ThreadId) {
        values.resize(config->blockDim[_dim]);
        std::iota(values.begin(), values.end(), 0);
    }
    if (_kind == CTAId) {
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
    _dst->print();
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
void PtxTree::add_node(std::unique_ptr<node_type> new_node,
                       const std::string &dst_register,
                       const std::vector<std::string> &src_registers) {
    if (auto [par_node, idx] = find_register_node(dst_register); par_node) {
        for (uint64_t i = 0; i < src_registers.size(); ++i) {
            _registers_to_leafs[src_registers[i]] = {new_node.get(), i};
        }
        par_node->set_child(std::move(new_node), idx);
    }
}
std::unique_ptr<PtxAbstractNode> PtxTree::eval(KLaunchConfig *config) {
    return _root->eval(config);
}

std::tuple<std::string, PtxNodeKind, int64_t> strip_argument(std::string arg,
                                                             bool strip_last) {
    if (strip_last)
        arg.pop_back();
    int64_t offset = 0;

    if (arg[0] == '[') {
        auto splitted_line = split_string(arg.substr(1, arg.size() - 2), "+");

        arg = splitted_line[0];
        if (splitted_line.size() > 1) {
            offset = std::stoll(splitted_line[1]);
        }

        return {arg, Parameter, offset};
    }

    if (arg[0] == '%') {
        if (arg[1] == 'r')
            return {arg, Register, 0};

        if (arg.substr(1, 4) == "ntid" || arg.substr(1, 6) == "nctaid" ||
            arg.substr(1, 3) == "tid" || arg.substr(1, 4) == "ntid") {
            return {arg, SpecialRegister, 0};
        }

        return {arg, Register, 0};
        arg.pop_back();
    }

    throw std::runtime_error("Invlaid argument");
}

std::vector<PtxTree> parsePtxTrees(std::string &ss) {
    std::vector<PtxTree> trees;
    std::string line;

    auto it = ss.rbegin();
    const auto end_it = ss.rend();
    while (rgetline(it, end_it, line)) {
        std::cout << line << std::endl;
        auto splitted_line = split_string(line, " ");
        if (startsWith(line, "cvta.to.global.u64")) {
            // Create a new tree and add it to the vector
            auto [name, kind, offset] =
                strip_argument(splitted_line.back(), true);

            if (kind == Register) {
                trees.emplace_back(name);
            }
            if (kind == Parameter) {
                trees.emplace_back("_t");
                trees.back().add_node(std::make_unique<PtxParameter>(name, 0),
                                      "_t", {});
            }
        } else if (startsWith(line, "mov")) {
            auto [dst_name, dst_kind, dst_offset] =
                strip_argument(splitted_line[1], true);
            auto [src_name, src_kind, src_offset] =
                strip_argument(splitted_line[2], true);

            if (dst_kind != Register)
                throw std::runtime_error("Destination must be register.");

            if (src_kind == Register) {
                for (auto &tree : trees) {
                    tree.replace_register(dst_name, src_name);
                }
            } else if (src_kind == Parameter) {
                for (auto &tree : trees) {
                    tree.add_node(
                        std::make_unique<PtxParameter>(src_name, src_offset),
                        dst_name, {});
                }
            } else {
                throw std::runtime_error("Invalid move instruction source.");
            }
        }
    }

    return trees;
}

std::vector<KParamInfo> parsePtxParameters(const std::string &ptx_data,
                                           const std::smatch &match) {
    const std::string &entry = match[1];
    const size_t str_idx = match.position(2)+1;
    std::string kernel_lines = ptx_data.substr(str_idx, match.length(2)-2);
    std::vector<PtxTree> trees = parsePtxTrees(kernel_lines);

    // Do fancy stuff with trees

    // Extract raw parameters from ptx
    // std::vector<KParamInfo> raw_parameters;
    // std::vector<NameWithOffset> params;
    // std::string line;
    // while(getline(ss, line)) {
    //     if (line.find(')') != std::string::npos) {
    //         break;
    //     }
    //     // NO parameters
    //     if(!startsWith(line, ".param")) {
    //         break;
    //     }
    //     assert(startsWith(line, ".param") && "Expected .param directive");
    //     auto splitted_line = split_string(line, " ");

    //     // Remove last comma)
    // }
    return {};
}

bool analyzePtx(const std::string &fname) {
    // analyze single ptx file
    std::ifstream s(fname);
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();

    static std::regex r_func_parameters(R"(.entry.*\s(.*)\(([^\)]*)\))",
                                        std::regex::ECMAScript);
    std::sregex_iterator i = std::sregex_iterator(
        ptx_data.begin(), ptx_data.end(), r_func_parameters);
    std::map<std::string, std::vector<KParamInfo>> tmp_map;
    for (; i != std::sregex_iterator(); ++i) {
        std::smatch m = *i;

        std::vector<KParamInfo> param_infos =
            parsePtxParameters(ptx_data, m);

        const std::string &entry = m[1];
        tmp_map.emplace(std::make_pair(entry, param_infos));
    }

    return true;
}

} // namespace PtxTreeParser