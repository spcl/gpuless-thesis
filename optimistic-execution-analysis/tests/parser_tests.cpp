#include "../tree_parser.h"
#include <gtest/gtest.h>

using namespace PtxTreeParser;

TEST(PtxTree, BasicTree) {
    PtxTree test_tree("test_register");
    std::vector<std::string> regs(2);
    regs[0] = "reg_a";
    regs[1] = "reg_b";
    test_tree.add_node(std::make_unique<PtxAddNode>(nullptr, nullptr),
                       "test_register", regs);
    std::vector<std::string> regs_b = {"test"};
    test_tree.add_node(std::make_unique<PtxImmediate>(42, s64), "reg_b",
                       regs_b);
    test_tree.print();
}

TEST(PtxTree, SimpleEval) {
    PtxTree test_tree("test_register");
    std::vector<std::string> regs = {"test"};
    test_tree.add_node(std::make_unique<PtxImmediate>(42, s64), "test_register",
                       regs);
    std::unique_ptr<PtxAbstractNode> evaluated = test_tree.eval(nullptr);
    EXPECT_EQ(evaluated->get_kind(), Immediate);

    auto &imm(dynamic_cast<PtxImmediate &>(*evaluated));
    EXPECT_EQ(imm.get_value(), 42);
}

TEST(PtxTree, AddEval) {
    PtxTree test_tree("test_register");
    std::vector<std::string> regs(2);
    regs[0] = "reg_a";
    regs[1] = "reg_b";
    test_tree.add_node(std::make_unique<PtxAddNode>(nullptr, nullptr),
                       "test_register", regs);

    // Add Immediate 42 to left
    std::vector<std::string> regs_empty;
    test_tree.add_node(std::make_unique<PtxImmediate>(42, s64), "reg_a",
                       regs_empty);

    // Add another add on right
    std::vector<std::string> regs2 = {"reg_c", "reg_d"};
    test_tree.add_node(std::make_unique<PtxAddNode>(nullptr, nullptr), "reg_b", regs2);

    // Add Immediate 10 to left
    test_tree.add_node(std::make_unique<PtxImmediate>(10, s64), "reg_c",
                       regs_empty);

    // Add Parameter to right
    test_tree.add_node(std::make_unique<PtxParameter>("test_param", 2, 0, s64), "reg_d", regs_empty);

    std::unique_ptr<PtxAbstractNode> evaluated = test_tree.eval(nullptr);
    EXPECT_EQ(evaluated->get_kind(), Parameter);

    auto &par(dynamic_cast<PtxParameter &>(*evaluated));
    EXPECT_EQ(par.get_offsets().size(), 1);
    EXPECT_EQ(par.get_offsets()[0], 54);
}

TEST(PtxParser, ParseSimpleCvta) {
    std::string s = "mov %r2, [myPtr]\n"
                    "cvta.to.global.u64 %r1, %r2;";
    std::vector<PtxTree> trees = parsePtxTrees(s);

    EXPECT_EQ(trees.size(), 1);

    trees[0].print();

}