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

PtxParameterType
ptxParameterTypeFromString(const std::string &str) {
    auto it = getStrToPtxParameterType().find(str);
    if (it == getStrToPtxParameterType().end()) {
        return PtxParameterType::invalid;
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

class PtxAbstractOperand {
public:
    // Make this struct abstract
    virtual std::string get_name() = 0;
};

class PtxRegister: public PtxAbstractOperand {
private:
    std::string name;
public:
    PtxRegister(std::string _name) : name(_name) {};
    std::string get_name() override {
        return name;
    }
};

class PtxParameter : public PtxAbstractOperand {
private:
    std::string name;
    PtxParameterType type;
    uint64_t offset;
public:
    PtxParameter(std::string _name, PtxParameterType _type, uint64_t _offset) : name(_name), type(_type), offset(_offset) {};
    std::string get_name() override {
        return name;
    }
    PtxParameterType get_type() {
        return type;
    }
    uint64_t get_offset() {
        return offset;
    }
};


class PtxAbstractNode {
public:
    virtual void print() = 0;
    virtual PtxAbstractNode* eval() = 0;
    virtual std::vector<PtxAbstractNode*> getChildren() = 0;
    virtual ~PtxAbstractNode() = 0;
};


class PtxAbstractValueNode : public virtual PtxAbstractNode {
public:
    virtual ~PtxAbstractValueNode() = 0;
    virtual void print() = 0;
    PtxAbstractNode* eval() override {
        return this;
    }
    std::vector<PtxAbstractNode*> getChildren() override {
        return std::vector<PtxAbstractNode*>();
    }
};

class PtxParamValueNode : public PtxAbstractValueNode {
private:
    PtxParameter param;
public:
    PtxParamValueNode(PtxParameter _param) : param(_param) {};
    void print() override {
        std::cout << "PtxParamValueNode(" << param.get_name() << ", " << getPtxParameterTypeToStr()[param.get_type()] << ", " << param.get_offset() << ")\n";
    }
    ~PtxParamValueNode() override {}

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
        std::cout << "MoveNode" << std::endl;
        std::cout << "\nsrc: ";
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




typedef enum {
    PtxAddOp,
    PtxMoveOp,
} PtxOpCode;

class PtxTree {
private:
    PtxAbstractNode* root;
    bool m_is_collapsed = false;
    bool m_is_param = false;
    
    // hashmap of registers of intereset -> leaf node
    std::unordered_map<std::string, PtxAbstractNode**> registers_to_leafs;

public:
    PtxTree(PtxAbstractOperand* operand) {
        if (auto* reg = dynamic_cast<PtxRegister*>(operand)) {
            // In this case we don't have to register a node
            registers_to_leafs[reg->get_name()] = &root;
        } else if (auto* param = dynamic_cast<PtxParameter*>(operand)) {
            root = new PtxParamValueNode(*param);
        } else {
            assert(false && "Unknown operand type");
        }
    };

    void print() {
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
            case PtxMoveOp:
                assert(operands.size() == 1 && "Move operation must have 1 operand");
                if (auto* reg = dynamic_cast<PtxRegister*>(operands[0])) {
                    // In this case we don't have to register a node
                    registers_to_leafs[reg->get_name()] = parent_to_child;
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

std::vector<PtxTree> parsePtxTrees(std::istringstream& ss) {
    return {};
};


std::vector<KParamInfo> parsePtxParameters(const std::string &ptx_data,
                   const std::smatch &match) {
    const std::string &entry = match[1];
    const size_t str_idx = match.position(2)+1;
    std::istringstream ss(ptx_data.substr(str_idx, ptx_data.size() - str_idx));
    std::vector<PtxTree> trees = parsePtxTrees(ss);

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
int main() {
    // 1. Tree with 1 node, e.g.:
    // cvta.to.global.u64 %r1, [myPtr];
    PtxParameter* param = new PtxParameter("myPtr", PtxParameterType::u64, 0);
    PtxTree tree = PtxTree(param);
    tree.print();


}