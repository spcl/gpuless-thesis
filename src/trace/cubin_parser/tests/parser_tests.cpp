#include "../tree_parser.h"
#include <fstream>
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
    EXPECT_EQ(imm.get_values()[0], 42);
}

TEST(PtxTree, AddEvalParam) {
    PtxTree test_tree("test_register");
    test_tree.add_node(std::make_unique<PtxAddNode>(nullptr, nullptr),
                       {{PtxOperandKind::Register, "test_register",0},
                        {PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::Register, "reg_b",0}});

    // Add Immediate 42 to left
    test_tree.add_node(std::make_unique<PtxAddNode>(nullptr, nullptr),
                       {{PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::Immediate, "", 22},
                        {PtxOperandKind::Immediate, "", 20}});

    // Add another add on right
    test_tree.add_node(std::make_unique<PtxAddNode>(nullptr, nullptr),
                       {{PtxOperandKind::Register, "reg_b",0},
                        {PtxOperandKind::Immediate, "", 10},
                        {PtxOperandKind::Parameter, "test_param", 2}});

    std::unique_ptr<PtxAbstractNode> evaluated = test_tree.eval(nullptr);
    EXPECT_EQ(evaluated->get_kind(), PtxNodeKind::Parameter);

    auto &par(dynamic_cast<PtxParameter &>(*evaluated));
    EXPECT_EQ(par.get_offsets().size(), 1);
    EXPECT_EQ(par.get_offsets()[0], 54);
}

TEST(PtxTree, AddEvalSimpleSpecialRegister) {
    PtxTree test_tree("test_register");
    test_tree.add_node(std::make_unique<PtxAddNode>(nullptr, nullptr),
                       {{PtxOperandKind::Register, "test_register",0},
                        {PtxOperandKind::SpecialRegisterCtaId, "",0},
                        {PtxOperandKind::Immediate, "",10}});

    std::unique_ptr<PtxAbstractNode> evaluated = test_tree.eval(nullptr);
    EXPECT_EQ(evaluated, nullptr);
    KLaunchConfig config{{3, 0, 0}, {3, 0, 0}};
    std::unique_ptr<PtxAbstractNode> collapsed = test_tree.eval(&config);
    EXPECT_NE(collapsed, nullptr);
    collapsed->print();
}

TEST(PtxTree, AddMultipleEvalsSpecialRegister) {
    PtxTree test_tree("test_register");
    test_tree.add_node(std::make_unique<PtxAddNode>(nullptr, nullptr),
                       {{PtxOperandKind::Register, "test_register",0},
                        {PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::Immediate, "",10}});

    test_tree.add_node(std::make_unique<PtxAddNode>(nullptr, nullptr),
                       {{PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::SpecialRegisterCtaId, "", 0},
                        {PtxOperandKind::Immediate, "",10}});

    test_tree.print();

    std::unique_ptr<PtxAbstractNode> evaluated = test_tree.eval(nullptr);
    EXPECT_EQ(evaluated, nullptr);
    KLaunchConfig config{{3, 0, 0}, {3, 0, 0}};
    std::unique_ptr<PtxAbstractNode> collapsed = test_tree.eval(&config);
    EXPECT_NE(collapsed, nullptr);
    collapsed->print();
}


TEST(PtxTree, SubEvalParam) {
    PtxTree test_tree("test_register");
    test_tree.add_node(std::make_unique<PtxSubNode>(nullptr, nullptr),
                       {{PtxOperandKind::Register, "test_register",0},
                        {PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::Register, "reg_b",0}});

    test_tree.add_node(std::make_unique<PtxSubNode>(nullptr, nullptr),
                       {{PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::Parameter, "test_param", 20},
                        {PtxOperandKind::Immediate, "", 10}});

    test_tree.add_node(std::make_unique<PtxSubNode>(nullptr, nullptr),
                       {{PtxOperandKind::Register, "reg_b",0},
                        {PtxOperandKind::Immediate, "", 30},
                        {PtxOperandKind::Immediate, "", 20}});


    std::unique_ptr<PtxAbstractNode> evaluated = test_tree.eval(nullptr);
    EXPECT_EQ(evaluated->get_kind(), PtxNodeKind::Parameter);

    auto &par(dynamic_cast<PtxParameter &>(*evaluated));
    EXPECT_EQ(par.get_offsets().size(), 1);
    EXPECT_EQ(par.get_offsets()[0], 0);
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

TEST(PtxParser, ParseMovV2) {
    std::string s = "mov.v2 {%r3, %r2}, [myPtr];\n"
                    "cvta.to.global.u64 %r1, %r2;";
    std::vector<PtxTree> trees = parsePtxTrees(s);
    EXPECT_EQ(trees.size(), 1);
    trees[0].print();
}

TEST(PtxParser, ParseAddV2) {
    std::string s = "add.v2 {%r3, %r2}, [myPtr+2], 4;\n"
                    "cvta.to.global.u64 %r1, %r3;\n"
                    "cvta.to.global.u64 %r4, %r2;";
    std::vector<PtxTree> trees = parsePtxTrees(s);
    EXPECT_EQ(trees.size(), 2);
    trees[0].print();
}
TEST(PtxParser, LoadV2) {
    std::string s = "ld.v2 {%r3, %r2}, [myPtr];\n"
                    "cvta.to.global.u64 %r1, %r3;\n"
                    "cvta.to.global.u64 %r4, %r2;";
    std::vector<PtxTree> trees = parsePtxTrees(s);
    EXPECT_EQ(trees.size(), 2);
    trees[0].print();
}

TEST(PtxParser, OriginalComplications) {
    std::ifstream s(TEST_RESOURCE_DIR"/test_ptx");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    std::vector<PtxTree> trees = parsePtxTrees(ptx_data);
    EXPECT_EQ(trees.size(), 2);
    auto empty = trees[0].eval(nullptr);
    EXPECT_EQ(empty, nullptr);
    auto par = trees[1].eval(nullptr);
    EXPECT_NE(par, nullptr);

    KLaunchConfig config{{3, 3, 0}, {3, 3, 0}};
    auto res = trees[0].eval(&config);
    EXPECT_NE(res, nullptr);
    res->print();
}

TEST(PtxSerializer, BasicSerialization) {
    PtxTree test_tree("test_register");
    test_tree.add_node(std::make_unique<PtxSubNode>(nullptr, nullptr),
                       {{PtxOperandKind::Register, "test_register",0},
                        {PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::Register, "reg_b",0}});

    test_tree.add_node(std::make_unique<PtxSubNode>(nullptr, nullptr),
                       {{PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::Parameter, "test_param", 20},
                        {PtxOperandKind::Immediate, "", 10}});

    test_tree.add_node(std::make_unique<PtxSubNode>(nullptr, nullptr),
                       {{PtxOperandKind::Register, "reg_b",0},
                        {PtxOperandKind::Immediate, "", 30},
                        {PtxOperandKind::Immediate, "", 20}});
    test_tree.print();

    std::stringstream ss;
    test_tree.serialize(ss);
    std::cout << ss.str() << std::endl;
    std::unique_ptr<PtxTree> new_tree(PtxTree::unserialize(ss));

    new_tree->print();
}