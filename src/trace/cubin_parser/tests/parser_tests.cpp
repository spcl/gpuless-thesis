#include "../tree_parser.h"
#include <gtest/gtest.h>

using namespace PtxTreeParser;

TEST(PtxTree, BasicTree) {
    PtxTree test_tree("test_register");
    test_tree.add_node(std::make_unique<PtxAddNode>(nullptr, nullptr),
                       {{PtxOperandKind::Register, "test_register",0},
                        {PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::Register, "reg_b",0}});
    test_tree.add_node(std::make_unique<PtxImmediate>(42, s64),
                       {{PtxOperandKind::Register, "reg_a",0}});
    test_tree.print();
}

TEST(PtxTree, SimpleEval) {
    PtxTree test_tree("test_register");
    test_tree.add_node(std::make_unique<PtxImmediate>(42, s64),
                       {{PtxOperandKind::Register, "test_register",0}});
    std::unique_ptr<PtxAbstractNode> evaluated = test_tree.eval(nullptr);
    EXPECT_EQ(evaluated->get_kind(), PtxNodeKind::Immediate);

    auto &imm(dynamic_cast<PtxImmediate &>(*evaluated));
    EXPECT_EQ(imm.get_value(), 42);
}

TEST(PtxTree, AddEval) {
    PtxTree test_tree("test_register");
    test_tree.add_node(std::make_unique<PtxAddNode>(nullptr, nullptr),
                       {{PtxOperandKind::Register, "test_register",0},
                        {PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::Register, "reg_b",0}});

    // Add Immediate 42 to left
    test_tree.add_node(std::make_unique<PtxImmediate>(42, s64),
                       {{PtxOperandKind::Register, "reg_a",0}});

    // Add another add on right
    test_tree.add_node(std::make_unique<PtxAddNode>(nullptr, nullptr),
                       {{PtxOperandKind::Register, "reg_b",0},
                        {PtxOperandKind::Register, "reg_c",0},
                        {PtxOperandKind::Register, "reg_d",0}});

    // Add Immediate 10 to left
    test_tree.add_node(std::make_unique<PtxImmediate>(10, s64),
                       {{PtxOperandKind::Register, "reg_c",0}});

    // Add Parameter to right
    test_tree.add_node(std::make_unique<PtxParameter>("test_param", 2, 0, s64),
                       {{PtxOperandKind::Register, "reg_d",0}});

    std::unique_ptr<PtxAbstractNode> evaluated = test_tree.eval(nullptr);
    EXPECT_EQ(evaluated->get_kind(), PtxNodeKind::Parameter);

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

TEST(PtxParser, ParseAdd) {
    std::string s = "add %r2, [par+4], 2;\n"
                    "cvta.to.global.u64 %r1, %r2;";
    std::vector<PtxTree> trees = parsePtxTrees(s);
    EXPECT_EQ(trees.size(), 1);
    trees[0].print();

    std::unique_ptr<PtxAbstractNode> p = trees[0].eval(nullptr);

    EXPECT_EQ(p->get_kind(), PtxNodeKind::Parameter);
    auto &par(static_cast<PtxParameter&>(*p));
    EXPECT_EQ(par.get_offsets().size(), 1);
    EXPECT_EQ(par.get_name(), "par");
    EXPECT_EQ(par.get_offsets()[0], 6);
}