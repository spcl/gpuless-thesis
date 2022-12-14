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

PtxNodeKind parseOperation(const std::string_view &op, int64_t &vec_op) {
    const static std::map<std::string, PtxNodeKind, std::less<void>>
        str_to_kind = {
            {"cvta", PtxNodeKind::Cvta}, {"mov", PtxNodeKind::MoveOp},
            {"add", PtxNodeKind::AddOp}, {"sub", PtxNodeKind::SubOp},
            {"mul", PtxNodeKind::MulOp}, {"div", PtxNodeKind::DivOp},
            {"rem", PtxNodeKind::RemOp}, {"abs", PtxNodeKind::AbsOp},
            {"neg", PtxNodeKind::NegOp}, {"min", PtxNodeKind::MinOp},
            {"max", PtxNodeKind::MaxOp}, {"shr", PtxNodeKind::ShrOp},
            {"shl", PtxNodeKind::ShlOp}, {"mad", PtxNodeKind::MadOp},
            {"sad", PtxNodeKind::SadOp}, {"ld", PtxNodeKind::MoveOp},
            {"cvt", PtxNodeKind::MoveOp}};
    auto splitted_string = split_string(op, ".");
    std::string_view opcode = splitted_string[0];

    // determine whether it is a vector instruction
    if (auto it =
            std::find(splitted_string.begin(), splitted_string.end(), "v4");
        it != splitted_string.end())
        vec_op = 4;
    else if (it = std::find(splitted_string.begin(), splitted_string.end(),
                            "v2");
             it != splitted_string.end())
        vec_op = 2;

    if (auto it = str_to_kind.find(opcode); it != str_to_kind.end()) {
        return it->second;
    } else {
        return PtxNodeKind::InvalidOp;
    }
}

PtxOperand parseArgument(std::string_view arg) {
    arg.remove_suffix(1);
    int64_t offset = 0;

    if (arg[0] == '[') {
        auto splitted_line = split_string(arg.substr(1, arg.size() - 2), "+");

        arg = splitted_line[0];
        if (splitted_line.size() > 1) {
            offset = std::stoll(splitted_line[1].data());
        }

        if (arg[0] == '%')
            return {PtxOperandKind::Register, std::string(arg), offset};
        else
            return {PtxOperandKind::Parameter, std::string(arg), offset};
    }

    if (arg[0] == '{')
        arg.remove_prefix(1);
    if (arg.back() == '}')
        arg.remove_suffix(1);

    if (arg[0] == '%') {
        if (arg[1] == 'r')
            return {PtxOperandKind::Register, std::string(arg), 0};

        static std::unordered_map<char, int> dim_to_idx = {
            {'x', 0}, {'y', 1}, {'z', 2}};

        if (arg.substr(1, 4) == "ntid")
            return {PtxOperandKind::SpecialRegisterNTid, std::string(arg),
                    dim_to_idx[arg[6]]};
        if (arg.substr(1, 6) == "nctaid")
            return {PtxOperandKind::SpecialRegisterNCtaId, std::string(arg),
                    dim_to_idx[arg[8]]};
        if (arg.substr(1, 3) == "tid")
            return {PtxOperandKind::SpecialRegisterTid, std::string(arg),
                    dim_to_idx[arg[5]]};
        if (arg.substr(1, 5) == "ctaid")
            return {PtxOperandKind::SpecialRegisterCtaId, std::string(arg),
                    dim_to_idx[arg[7]]};

        return {PtxOperandKind::Register, std::string(arg), 0};
    }

    try {
        std::int64_t imm_value =
            std::stoll(arg.data()); // Throws if not immediate (?)
        return {PtxOperandKind::Immediate, std::string(arg), imm_value};
    } catch (std::invalid_argument &) {
        return {PtxOperandKind::Parameter, std::string(arg), offset};
    }
}

std::vector<std::pair<std::unique_ptr<PtxTree>, std::string>>
parsePtxTrees(std::string_view ss) {
    std::vector<std::pair<std::unique_ptr<PtxTree>, std::string>> trees;

    const auto beg = ss.begin();
    auto end_it = ss.end();

    for (std::string_view line = rgetline(beg, end_it);;
         line = rgetline(beg, end_it)) {
        auto splitted_line = split_string(line, " ");
        int64_t vec_count = 1;
        PtxNodeKind op_kind = parseOperation(splitted_line[0], vec_count);

        if (op_kind == PtxNodeKind::InvalidOp) {
            if (beg == end_it)
                break;
            else
                continue;
        }

        std::vector<PtxOperand> pars(splitted_line.size() - 1);
        for (size_t i = 1; i < splitted_line.size(); ++i) {
            pars[i - 1] = parseArgument(splitted_line[i]);

            if (pars[i - 1].kind == PtxOperandKind::Parameter) {
                for (auto &[tree, par] : trees) {
                    if (tree->find_register_node(pars[0].name).first) {
                        par = pars[1].name;
                    }
                }
            }
        }

        if (op_kind == PtxNodeKind::Cvta) {
            auto &par = pars[1];

            switch (par.kind) {
            case PtxOperandKind::Register:
                trees.emplace_back(
                    std::make_pair(std::make_unique<PtxTree>(par.name), "Unknown"));
                break;
            case PtxOperandKind::Parameter:
                trees.emplace_back(std::make_pair(std::make_unique<PtxTree>("_t"), pars[1].name));
                trees.back().first->add_node(
                    std::make_unique<PtxParameter>(par.name, 0),
                    {{par.kind, par.name, par.value}});
                break;
            default:
                throw std::runtime_error("Invalid operand to cvta.");
            }
        } else if (op_kind == PtxNodeKind::MoveOp) {
            for (int64_t v = 0; v < vec_count; ++v) {
                if (pars[v].kind != PtxOperandKind::Register)
                    throw std::runtime_error("Destination must be register.");

                switch (pars[vec_count].kind) {
                case PtxOperandKind::Register:
                    for (auto &tree : trees) {
                        tree.first->replace_register(pars[v].name,
                                                    pars[vec_count].name);
                    }
                    break;
                case PtxOperandKind::Parameter:
                    for (auto &tree : trees) {
                        tree.first->add_node(
                            std::make_unique<PtxParameter>(
                                pars[vec_count].name, pars[vec_count].value),
                            {pars[v]});
                    }
                    break;
                case PtxOperandKind::Immediate:
                    for (auto &tree : trees) {
                        tree.first->add_node(std::make_unique<PtxImmediate>(
                                                pars[vec_count].value, s64),
                                            {pars[v]});
                    }
                    break;
                case PtxOperandKind::SpecialRegisterTid:
                case PtxOperandKind::SpecialRegisterNTid:
                case PtxOperandKind::SpecialRegisterCtaId:
                case PtxOperandKind::SpecialRegisterNCtaId:
                    for (auto &tree : trees) {
                        tree.first->add_node(
                            std::make_unique<PtxSpecialRegister>(
                                pars[vec_count].kind, pars[vec_count].value),
                            {pars[v]});
                    }
                    break;
                }
            }
        } else {
            for (auto &tree : trees) {
                for (int64_t v = 0; v < vec_count; ++v) {
                    std::vector<PtxOperand> p{pars[v]};
                    p.insert(p.end(), pars.begin() + vec_count, pars.end());
                    tree.first->add_node(produceNode(op_kind), p);
                }
            }
        }
    }

    return trees;
}

} // namespace PtxTreeParser