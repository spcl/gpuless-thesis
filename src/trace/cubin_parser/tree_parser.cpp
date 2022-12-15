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

PtxNodeKind TreeParser::parseOperation(const std::string_view &op,
                                       int64_t &vec_op) {
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
    auto splitted_string = splitString(op, ".");
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
        if (it->second == PtxNodeKind::Cvta &&
            (splitted_string.size() < 3 || splitted_string[2] != "global"))
            return PtxNodeKind::InvalidOp;

        return it->second;
    } else {
        return PtxNodeKind::InvalidOp;
    }
}

PtxOperand TreeParser::parseArgument(std::string arg) {
    arg.pop_back();
    int64_t offset = 0;

    if (arg[0] == '[') {
        auto splitted_line = splitStringCopy(arg.substr(1, arg.size() - 2), "+");

        arg = splitted_line[0];
        if (splitted_line.size() > 1) {
            offset = std::stoll(splitted_line[1]);
        }

        if (_param_names.find(arg) == _param_names.end())
            return {PtxOperandKind::Register, arg, offset};
        else
            return {PtxOperandKind::Parameter, arg, offset};
    }

    if (arg[0] == '{')
        arg = arg.substr(1, arg.size()-1);
    if (arg.back() == '}')
        arg.pop_back();

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

    if (_param_names.find(arg) != _param_names.end())
        return {PtxOperandKind::Parameter, arg, offset};

    try {
        std::int64_t imm_value =
            std::stoll(arg); // Throws if not immediate (?)
        return {PtxOperandKind::Immediate, arg, imm_value};
    } catch (std::invalid_argument &) {
        return {PtxOperandKind::Register, arg, offset};
    }
}

std::vector<std::pair<std::unique_ptr<PtxTree>, std::string>>
TreeParser::parsePtxTrees(std::string_view ss) {
    std::vector<std::pair<std::unique_ptr<PtxTree>, std::string>> trees;

    const auto beg = ss.begin();
    auto end_it = ss.end();

    for (std::string_view line = rgetline(beg, end_it);;
         line = rgetline(beg, end_it)) {
        auto splitted_line = splitString(line, " ");
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
            pars[i - 1] = parseArgument(std::string(splitted_line[i]));

            if (pars[i - 1].kind == PtxOperandKind::Parameter) {
                for (auto &[tree, par] : trees) {
                    if (!tree->find_register_node(pars[0].name).empty()) {
                        par = pars[1].name;
                    }
                }
            }
        }
        if (op_kind == PtxNodeKind::MoveOp || op_kind == PtxNodeKind::Cvta) {
            for (int64_t v = 0; v < vec_count; ++v) {
                if (pars[v].kind != PtxOperandKind::Register)
                    throw std::runtime_error("Destination must be register.");

                switch (pars[vec_count].kind) {
                case PtxOperandKind::Register:
                    for (auto &tree : trees) {
                        if(!pars[vec_count].value)
                            tree.first->replace_register(pars[v].name,
                                                         pars[vec_count].name);
                        else
                            tree.first->add_node(PtxNodeKind::AddOp,
                                                 {{PtxOperandKind::Register, pars[v].name, 0},
                                                 {PtxOperandKind::Register, pars[vec_count].name, 0},
                                                 {PtxOperandKind::Immediate, "", pars[vec_count].value}});
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

            if (op_kind == PtxNodeKind::Cvta) {
                auto &par = pars[1];

                switch (par.kind) {
                case PtxOperandKind::Register:
                    if (!par.value)
                        trees.emplace_back(std::make_pair(
                            std::make_unique<PtxTree>(par.name), "Unknown"));
                    else {
                        trees.emplace_back(std::make_pair(
                            std::make_unique<PtxTree>("_t"), pars[1].name));
                        trees.back().first->add_node(
                            PtxNodeKind::AddOp,
                            {{PtxOperandKind::Register, "_t", 0},
                             {PtxOperandKind::Immediate, "", par.value},
                             {PtxOperandKind::Register, par.name, 0}});
                    }
                    break;
                case PtxOperandKind::Parameter:
                    trees.emplace_back(std::make_pair(
                        std::make_unique<PtxTree>("_t"), pars[1].name));
                    trees.back().first->add_node(
                        std::make_unique<PtxParameter>(par.name, 0),
                        {{PtxOperandKind::Register, "_t", 0},
                         {par.kind, par.name, par.value}});
                    break;
                default:
                    throw std::runtime_error("Invalid operand to cvta.");
                }
            }
        } else {
            for (auto &tree : trees) {
                for (int64_t v = 0; v < vec_count; ++v) {
                    std::vector<PtxOperand> p{pars[v]};
                    p.insert(p.end(), pars.begin() + vec_count, pars.end());
                    tree.first->add_node(op_kind, p);
                }
            }
        }
    }

    return trees;
}

} // namespace PtxTreeParser