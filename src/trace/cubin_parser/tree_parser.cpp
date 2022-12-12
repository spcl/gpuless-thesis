#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <regex>
#include <sstream>

#include "parser_util.h"
#include "ptx_tree.h"
#include "tree_parser.h"

namespace PtxTreeParser {

PtxNodeKind parseOperation(const std::string &op) {
    const static std::map<std::string, PtxNodeKind> str_to_kind = {
        {"cvta", PtxNodeKind::Cvta},
        {"mov", PtxNodeKind::MoveOp},
        {"add", PtxNodeKind::AddOp},
        {"sub", PtxNodeKind::SubOp},
        {"mul", PtxNodeKind::MulOp},
        {"div", PtxNodeKind::DivOp},
        {"rem", PtxNodeKind::RemOp},
        {"abs", PtxNodeKind::AbsOp},
        {"neg", PtxNodeKind::NegOp},
        {"min", PtxNodeKind::MinOp},
        {"max", PtxNodeKind::MaxOp},
        {"shr", PtxNodeKind::ShrOp},
        {"shl", PtxNodeKind::ShlOp},
        {"mad", PtxNodeKind::MadOp},
        {"sad", PtxNodeKind::SadOp},
        {"ld", PtxNodeKind::LdOp},
        };
    std::string opcode = split_string(op, ".")[0];

    if (auto it = str_to_kind.find(opcode); it != str_to_kind.end()) {
        return it->second;
    } else {
        return PtxNodeKind::invalidOp;
    }
}

PtxOperand parseArgument(std::string arg) {
    arg.pop_back();
    int64_t offset = 0;

    if (arg[0] == '[') {
        auto splitted_line = split_string(arg.substr(1, arg.size() - 2), "+");

        arg = splitted_line[0];
        if (splitted_line.size() > 1) {
            offset = std::stoll(splitted_line[1]);
        }

        return {PtxOperandKind::Parameter, arg, offset};
    }

    if (arg[0] == '%') {
        if (arg[1] == 'r')
            return {PtxOperandKind::Register, arg, 0};

        static std::unordered_map<char, int> dim_to_idx = {
            {'x', 0}, {'y', 1}, {'z', 2}};

        if (arg.substr(1, 4) == "ntid")
            return {PtxOperandKind::SpecialRegisterNTid, arg,
                    dim_to_idx[arg[6]]};
        if (arg.substr(1, 6) == "nctaid")
            return {PtxOperandKind::SpecialRegisterNCtaId, arg,
                    dim_to_idx[arg[8]]};
        if (arg.substr(1, 3) == "tid")
            return {PtxOperandKind::SpecialRegisterTid, arg,
                    dim_to_idx[arg[5]]};
        if (arg.substr(1, 5) == "ctaid")
            return {PtxOperandKind::SpecialRegisterCtaId, arg,
                    dim_to_idx[arg[7]]};

        return {PtxOperandKind::Register, arg, 0};
    }

    std::int64_t imm_value = std::stoll(arg); // Throws if not immediate
    return {PtxOperandKind::Immediate, arg, imm_value};
}

std::unique_ptr<PtxTree::node_type> produce_node(PtxNodeKind kind) {
    switch(kind) {
        case PtxNodeKind::AddOp:
            return std::make_unique<PtxAddNode>(nullptr, nullptr);
        case PtxNodeKind::SubOp:
            return std::make_unique<PtxSubNode>(nullptr, nullptr);
        case PtxNodeKind::MulOp:
            return std::make_unique<PtxMulNode>(nullptr, nullptr);
        case PtxNodeKind::MadOp:
            throw std::runtime_error("not implemented yet");
            return std::make_unique<PtxAddNode>(nullptr, std::make_unique<PtxMulNode>());
        case PtxNodeKind::SadOp:
            throw std::runtime_error("not implemented yet");
            return std::make_unique<PtxAddNode>(nullptr, std::make_unique<PtxAbsNode>());
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

std::vector<PtxTree> parsePtxTrees(std::string &ss) {
    std::vector<PtxTree> trees;
    std::string line;

    auto it = ss.rbegin();
    const auto end_it = ss.rend();

    while (rgetline(it, end_it, line)) {
        auto splitted_line = split_string(line, " ");
        PtxNodeKind op_kind = parseOperation(splitted_line[0]);

        if (op_kind == PtxNodeKind::invalidOp)
            continue; // Opeartion not relevant

        std::vector<PtxOperand> pars(splitted_line.size() - 1);
        for (size_t i = 1; i < splitted_line.size(); ++i) {
            pars[i-1] = parseArgument(splitted_line[i]);
        }

        if (op_kind == PtxNodeKind::Cvta) {
            auto &par = pars[1];

            switch (par.kind) {
            case PtxOperandKind::Register:
                trees.emplace_back(par.name);
                break;
            case PtxOperandKind::Parameter:
                trees.emplace_back("_t");
                trees.back().add_node(
                    std::make_unique<PtxParameter>(par.name, 0),
                    {{par.kind, par.name, par.offset}});
                break;
            default:
                throw std::runtime_error("Invalid operand to cvta.");
            }
        } else if(op_kind == PtxNodeKind::MoveOp) {
            if(pars[0].kind != PtxOperandKind::Register)
                throw std::runtime_error("Destination must be register.");

            switch (pars[1].kind) {

            case PtxOperandKind::Register:
                for (auto &tree : trees) {
                    tree.replace_register(pars[0].name, pars[1].name);
                }
                break;
            case PtxOperandKind::Parameter:
                for (auto &tree : trees) {
                    tree.add_node(
                        std::make_unique<PtxParameter>(pars[1].name, pars[1].offset),
                        {pars[0]});
                }
                break;
            case PtxOperandKind::Immediate:
                for (auto &tree : trees) {
                    tree.add_node(
                        std::make_unique<PtxImmediate>(pars[1].offset, s64),
                        {pars[0]});
                }
                break;
            case PtxOperandKind::SpecialRegisterTid:
            case PtxOperandKind::SpecialRegisterNTid:
            case PtxOperandKind::SpecialRegisterCtaId:
            case PtxOperandKind::SpecialRegisterNCtaId:
                for (auto &tree : trees) {
                    tree.add_node(
                        std::make_unique<PtxSpecialRegister>(pars[1].kind, pars[1].offset),
                        {pars[0]});
                }
                break;
            }
        } else {
            for(auto& tree : trees) {
                tree.add_node(produce_node(op_kind), pars);
            }
        }
    }

    return trees;
}

std::vector<KParamInfo> parsePtxParameters(const std::string &ptx_data,
                                           const std::smatch &match) {
    const std::string &entry = match[1];
    const size_t str_idx = match.position(2) + 1;
    std::string kernel_lines = ptx_data.substr(str_idx, match.length(2) - 2);
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

        std::vector<KParamInfo> param_infos = parsePtxParameters(ptx_data, m);

        const std::string &entry = m[1];
        tmp_map.emplace(std::make_pair(entry, param_infos));
    }

    return true;
}

} // namespace PtxTreeParser