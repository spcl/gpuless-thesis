#include <map>
#include <unordered_map>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <regex>
#include <utility>
#include <assert.h>


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
    AddOp,
    MoveOp,
    invalidOp,
} PtxOpCode;

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
        : paramName(std::move(name)), type(par_type), typeSize(typesize), align(alignment),
          size(t_size), ptrOffsets(vec_size){};
};

std::map<std::string, PtxOpCode> &getStrToPtxOpCode() {
    static std::map<std::string, PtxOpCode> map_ = {
        {"add", AddOp}, {"mov", MoveOp},
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

PtxOpCode ptxOpCodeFromString(const std::string &str) {
    auto it = getStrToPtxOpCode().find(str);
    if (it == getStrToPtxOpCode().end()) {
        return PtxOpCode::invalidOp;
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

std::vector<std::string> split_string(std::string str, const std::string &delimiter) {
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

bool rgetline(std::string::reverse_iterator &it, const std::string::reverse_iterator &end,
              std::string &line) {
    if(it == end) return false;
    std::stringstream ss;
    char t;
    while (it != end && (t = *it++) != '\n') {
        ss << t;
    }
    line = ss.str();
    std::reverse(line.begin(), line.end());
    return true;
}

class PtxAbstractOperand {
public:
    // Make this struct abstract
    virtual std::string get_name() = 0;
    virtual uint64_t get_offset() {
        return 0;
    }
};

class PtxRegister: public PtxAbstractOperand {
private:
    std::string name;
    uint64_t offset;
    std::string name_with_offset;
public:
    PtxRegister(std::string _name_with_offset) : name_with_offset(_name_with_offset) {
        auto split = split_string(name, "+");
        name = split[0];
        offset = split.size() == 1 ? 0 : std::stoull(split[1]);
    };
    PtxRegister(std::string _name, uint64_t _offset) : name(_name), offset(_offset) {
        name_with_offset = name + "+" + std::to_string(offset);
    };
    std::string get_name() override {
        return name;
    }
    uint64_t get_offset() override {
        return offset;
    }
    std::string get_name_with_offset() {
        return name_with_offset;
    }
};

class PtxParameter : public PtxAbstractOperand {
private:
    std::string name;
    uint64_t offset;
    std::string name_with_offset;
    PtxParameterType type = PtxParameterType::invalid;
public:
    PtxParameter(std::string _name, PtxParameterType _type, uint64_t _offset) : name(_name), type(_type), offset(_offset) {};
    PtxParameter(std::string _name, uint64_t _offset) : name(_name), offset(_offset) {};
    PtxParameter(std::string _name_with_offset) : name_with_offset(_name_with_offset) {
        auto split = split_string(_name_with_offset, "+");
        name = split[0];
        offset = split.size() == 1 ? 0 : std::stoull(split[1]);
    };
    std::string get_name() override {
        return name;
    }
    uint64_t get_offset() {
        return offset;
    }
    PtxParameterType get_type() {
        assert(type != PtxParameterType::invalid && "Type of parameter not known");
        return type;
    }
};


class PtxAbstractNode {
public:
    virtual void print() = 0;
    virtual PtxAbstractNode* eval() = 0;
    virtual std::vector<PtxAbstractNode*> getChildren() = 0;
    virtual ~PtxAbstractNode() = default;
};

class PtxAbstractValueNode : public PtxAbstractNode {
public:
    PtxAbstractNode* eval() override {
        return this;
    }
    std::vector<PtxAbstractNode*> getChildren() override {
        return std::vector<PtxAbstractNode*>();
    }
    virtual ~PtxAbstractValueNode() = default;
};

class PtxParamValueNode : public PtxAbstractValueNode {
private:
    PtxParameter param;
public:
    PtxParamValueNode(PtxParameter _param) : param(_param) {};
    void print() override {
        // std::cout << "PtxParamValueNode(" << param.get_name() << ", " << getPtxParameterTypeToStr()[param.get_type()] << ", " << param.get_offset() << ")\n";
        std::cout << "PtxParamValueNode(" << param.get_name() << ", " << param.get_offset() << ")\n";
    }
    ~PtxParamValueNode() {}

    PtxParameter getParam() {
        return param;
    }
};

class PtxAddNode : public PtxAbstractNode {
private:
    PtxAbstractNode* L;
    PtxAbstractNode* R;
public:
    PtxAddNode(PtxAbstractNode* _L, PtxAbstractNode* _R) : L(_L), R(_R) {}
    void print() override {
        std::cout << "AddNode" << std::endl;
        std::cout << "\nL: ";
        L->print();
        std::cout << "\nR: ";
        R->print();
    }

    std::vector<PtxAbstractNode*> getChildren() override {
        return {L, R};
    }

    PtxAbstractValueNode* eval() override {
        // Check if both children are of type PtxValueNode, if not, then we can't collapse further
        // if(auto* L_ = dynamic_cast<PtxValueNode>(L)) {
        //     if(auto* R_ = dynamic_cast<PtxValueNode>(R)) {
        //         return new PtxValueNode(L_->value + R_->value);
        //     }
        // }
        return nullptr;
    }

    ~PtxAddNode() override {
        delete L;
        delete R;
    }
};

class PtxMoveNode : public PtxAbstractNode {
private:
    PtxAbstractNode* src;
public:
    PtxMoveNode(PtxAbstractNode* _src) : src(_src) {};

    void print() override {
        std::cout << "MoveNode\n";
        std::cout << "src: ";
        src->print();
    }

    std::vector<PtxAbstractNode*> getChildren() override {
        return {src};
    }

    PtxAbstractNode* eval() override {
        return src->eval();
    }

    ~PtxMoveNode() override {
        delete src;
    }
};



class PtxTree {
private:
    PtxAbstractNode* root = nullptr;
    bool m_is_collapsed = false;
    bool m_is_param = false;
    
    // hashmap of registers of interests -> leaf node
    std::unordered_map<std::string, PtxAbstractNode**> registers_to_leafs;

public:
    PtxTree(PtxAbstractOperand* operand) {
        if (auto* reg = dynamic_cast<PtxRegister*>(operand)) {
            // In this case we don't have to register a node
            registers_to_leafs[reg->get_name_with_offset()] = &root;
        } else if (auto* param = dynamic_cast<PtxParameter*>(operand)) {
            root = new PtxParamValueNode(*param);
        } else {
            assert(false && "Unknown operand type");
        }

    };

    void print() const {
        std::cout << "CvtaNode:\n";
        root->print();
    }
    
    // Return the parent node of a register if it exists in tree
    // this operation has to be fast as it will be called for every instruction
    PtxAbstractNode** get_parent(std::string reg) {
        if (registers_to_leafs.find(reg) != registers_to_leafs.end()) {
            return registers_to_leafs[reg];
        }
        return nullptr;  
    }

    void add_node(PtxOpCode opcode, PtxAbstractNode** parent_to_child, std::vector<PtxAbstractOperand*> operands) {
        switch(opcode) {
            case MoveOp:
                assert(operands.size() == 1 && "Move operation must have 1 operand");
                if (auto* reg = dynamic_cast<PtxRegister*>(operands[0])) {
                    // In this case we don't have to register a node
                    registers_to_leafs[reg->get_name_with_offset()] = parent_to_child;
                } else if(auto* param = dynamic_cast<PtxParameter*>(operands[0])) {
                    *parent_to_child = new PtxMoveNode(new PtxParamValueNode(*param));
                } else {
                    assert(false && "Move operand is neither a register nor a parameter");
                }
                break;
        }
    }

    void collapse() {
        root = root->eval();
        // Check if root is of type PtxParamValueNode:
        // If yes, then the tree is fully collapsed and the last node is a parameter
        if(auto* collapsed_ = dynamic_cast<PtxParamValueNode*>(root)) {
            m_is_collapsed = true;
            m_is_param = true;
        } else if (auto* collapsed_ = dynamic_cast<PtxAbstractValueNode*>(root)) {
            m_is_collapsed = true;
        }
    }

    bool is_collapsed() {
        return m_is_collapsed;
    }

    bool is_param() {
        assert(this->m_is_collapsed && "Cannot tell if root is a parameter, as the tree is not collapsed");
        return m_is_param;
    }

    PtxParameter get_param() {
        assert(this->m_is_collapsed && "Cannot get param of non-collapsed Tree");
        // For now, throw this assertion as we want to see when this happens (e.g. global variables)
        assert(this->m_is_param && "Cannot get param of non-parameter Tree");
        // The dynamic cast is safe, as we checked that the node is a parameter node
        // (At least, I hope so)
        return dynamic_cast<PtxParamValueNode*>(root)->getParam();
    }

    ~PtxTree() {
        delete root;
    }
};


std::vector<PtxTree> parsePtxTrees(std::string& ss) {
    std::vector<PtxTree> trees;
    std::string line;

    auto it = ss.rbegin();
    const auto end_it = ss.rend();
    while(rgetline(it, end_it, line)) {
        std::cout << line << std::endl;
        if(startsWith(line, "cvta.to.global.u64")) {
            // Create a new tree and add it to the vector
            auto splitted_line = split_string(line, " ");
            // the operand is the last element, without the ;
            auto operand = splitted_line.back();
            operand.pop_back();
            if (operand[0] == '[') {
                operand = operand.substr(1, operand.size() - 2);
            }
            if (operand[0] == '%') {
                auto* reg = new PtxRegister(operand);
                trees.push_back(PtxTree(reg));  
            } else {
                auto* param = new PtxParameter(operand);
                trees.push_back(PtxTree(param));
            }
        } else if(startsWith(line, "mov.u64") || startsWith(line, "mov.b64")) {
            auto operands = split_string(line, ",");
            std::string dst = split_string(operands[0], " ").back();
            // Check if the dst is a register of interest
            for(auto& tree: trees) {
                auto* parent = tree.get_parent(dst);
                if(parent != nullptr) {
                    std::string src = split_string(operands[1], " ").back();
                    // Remove ; from src and if necessary [] from src
                    src.pop_back();
                    if(src[0] == '[') {
                        src = src.substr(1, src.size() - 2);
                    }
                    if(src[0] == '%') {
                        tree.add_node(MoveOp, parent, {new PtxRegister(src)});
                    } else {
                        tree.add_node(MoveOp, parent, {new PtxParameter(src)});
                    }
                    break;
                }
            }
        }
    }

    return trees;
};

struct NameWithOffset {
    std::string name;
    int offset;
};

std::vector<PtxTree> testKernel(const std::string &fname, const std::string& kernel_mangle) {
    std::ifstream s(fname);
    // get lines
    std::string line;
    std::stringstream ss;

    // 1. Get the kernel lines and find the parameters
    bool started = false;
    std::vector<NameWithOffset> params;
    std::vector<KParamInfo> raw_parameters;


    while(getline(s, line)) {
        if (line.find(kernel_mangle) != std::string::npos) {
            started = true;
        } else if (line.find(".entry") != std::string::npos) {
            started = false;
        }
        if (!started) continue;
        
        if (line.find(')') != std::string::npos) {
            break;
        }
        // NO parameters
        if(!startsWith(line, ".param")) {
            break;
        }
        assert(startsWith(line, ".param") && "Expected .param directive");
        auto splitted_line = split_string(line, " ");

        // Remove last comma
        auto last = splitted_line.back();
        if(endsWith(last, ",")) {
            splitted_line.back() = last.substr(0, last.size() - 1);
        }

        if(splitted_line[1] == ".align") {
            int param_align = std::stoi(splitted_line[2]);
            const std::string &name = splitted_line[4];
            std::vector<std::string> splitted_name = split_string(name, "[");
            const std::string &param_name = splitted_name[0];
            // Remove last ']' from the size
            int param_size = std::stoi(
                splitted_name[1].substr(0, splitted_name[1].size() - 1));

            std::string type_name = splitted_line[3].substr(1, splitted_line[3].size());
            PtxParameterType param_type = ptxParameterTypeFromString(type_name);
            int param_typeSize = byteSizePtxParameterType(param_type);

            KParamInfo param(param_name, ptxParameterTypeFromString(type_name), param_typeSize, param_align, param_size, 0);
            raw_parameters.push_back(param);

            for(int offset = 0; offset < param_size; offset += param_align) {
                params.push_back({param_name, offset});
            }
        } else {
            std::string &name = splitted_line[2];
            std::string typeName = splitted_line[1].substr(1, splitted_line[1].size()-1);
            auto type = ptxParameterTypeFromString(typeName);
            KParamInfo param(name, type, byteSizePtxParameterType(type), 0, 1, 0);

            raw_parameters.push_back(param);
            params.push_back({name ,0});
        }
           
    }

    // 2. Parse the trees
    std::string kernel_lines = ss.str();
    std::vector<PtxTree> trees = parsePtxTrees(kernel_lines);


    // 3. Update KParamInfos according to the trees

    // Go through every tree, collapse them and update the parameters if they're ptrs

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


using namespace PtxTreeParser;

void parsePtxTrees_tests() {
    // 1. Tree with 1 node, e.g.:
    {
    std::string s = "cvta.to.global.u64 %r1, [myPtr];\n";
    std::vector<PtxTree> trees = parsePtxTrees(s);

    for(auto &tree : trees) tree.print();
    }
    // // 2. Tree with 2 nodes, e.g.:
    // {
    // std::string s = "mov.u64 %r2, [myPtr];\ncvta.to.global.u64 %r1, %r2;\n";
    // std::vector<PtxTree> trees = parsePtxTrees(s);
    // for(const auto &tree : trees) tree.print();
    // }
    // // 2. Multiple Trees with 2 nodes, e.g.:
    // {
    // std::string s1 = "mov.u64 %r2, [myPtr1];\ncvta.to.global.u64 %r1, %r2;\n";
    // std::string s2 = "mov.u64 %r2, [myPtr2];\ncvta.to.global.u64 %r1, %r2;\n";
    // std::string s = s1 + s2;
    // std::vector<PtxTree> trees = parsePtxTrees(s);
    // for(const auto &tree : trees) tree.print();
    // }
}
void PtxTree_tests() {
    // 1. Tree with 1 node, e.g.:
    {
    // cvta.to.global.u64 %r1, [myPtr];
    PtxParameter* param = new PtxParameter("myPtr", PtxParameterType::u64, 0);
    PtxTree tree = PtxTree(param);
    tree.print();
    tree.collapse();
    tree.print();
    assert(tree.is_collapsed() && "Tree should be collapsed");
    assert(tree.is_param() && "Tree should be a parameter");
    std::cout << "First Test passed\n\n";
    }

    // 2. Tree with 2 nodes, e.g.:
    {
    // cvta.to.global.u64 %r1, %r2;
    PtxRegister* reg = new PtxRegister("%r2");
    PtxTree tree = PtxTree(reg);
    // mov.u64 %r2, [myPtr];
    PtxAbstractNode** parent = tree.get_parent("%r2");
    if (parent != nullptr) {
        PtxParameter* param = new PtxParameter("myPtr", PtxParameterType::u64, 0);
        tree.add_node(MoveOp, parent, {param});
    }

    tree.print();
    tree.collapse();
    tree.print();
    assert(tree.is_collapsed() && "Tree should be collapsed");
    assert(tree.is_param() && "Tree should be a parameter");
    std::cout << "Second Test passed\n\n";
    }

    // 3. Tree with 10 move nodes, e.g.:
    {
    // cvta.to.global.u64 %r1, %r2;
    PtxRegister* param = new PtxRegister("%r2");
    PtxTree tree = PtxTree(param);

    // mov.u64 %r(i-1), r(i);
    for(uint64_t i = 3; i < 13; ++i) {
        std::string dst = "%r" + std::to_string(i-1);
        std::string src = "%r" + std::to_string(i);

        PtxAbstractNode** parent = tree.get_parent(dst);
        if (parent != nullptr) {
            PtxRegister* reg = new PtxRegister(src);
            tree.add_node(MoveOp, parent, {reg});
        }
    }

    // mov.u64 %r12, [myPtr];
    PtxAbstractNode** parent = tree.get_parent("%r12");
    if (parent != nullptr) {
        PtxParameter* param = new PtxParameter("myPtr", PtxParameterType::u64, 0);
        tree.add_node(MoveOp, parent, {param});
    }

    tree.print();
    tree.collapse();
    tree.print();
    assert(tree.is_collapsed() && "Tree should be collapsed");
    assert(tree.is_param() && "Tree should be a parameter");
    std::cout << "Third Test passed\n\n";
    }
}

int main() {
    // PtxTree_tests();
    parsePtxTrees_tests();
    // std::string kernel_mangle = "_ZN6caffe24math54_GLOBAL__N__beabb26c_14_elementwise_cu_f0da4091_12388115AxpbyCUDAKernelIfdEEvlT_PKT0_S3_PS4_";
    // testKernel("dumped_ptx.ptx", kernel_mangle);
    return 0;
}