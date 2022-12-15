#include "../../cubin_analysis.hpp"
#include "../tree_parser.h"
#include "../parser_util.h"
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
    auto trees = parsePtxTrees(s);

    EXPECT_EQ(trees.size(), 1);

    trees[0].first->print();

}

TEST(PtxParser, ParseAdd) {
    std::string s = "add %r2, [par+4], 2;\n"
                    "cvta.to.global.u64 %r1, %r2;";
    auto trees = parsePtxTrees(s);
    EXPECT_EQ(trees.size(), 1);
    trees[0].first->print();

    std::unique_ptr<PtxAbstractNode> p = trees[0].first->eval(nullptr);

    EXPECT_EQ(p->get_kind(), PtxNodeKind::Parameter);
    auto &par(static_cast<PtxParameter&>(*p));
    EXPECT_EQ(par.get_offsets().size(), 1);
    EXPECT_EQ(par.get_offsets()[0], 6);
}

TEST(PtxParser, ParseMovV2) {
    std::string s = "mov.v2 {%r3, %r2}, [myPtr];\n"
                    "cvta.to.global.u64 %r1, %r2;";
    auto trees = parsePtxTrees(s);
    EXPECT_EQ(trees.size(), 1);
    trees[0].first->print();
}

TEST(PtxParser, ParseAddV2) {
    std::string s = "add.v2 {%r3, %r2}, [myPtr+2], 4;\n"
                    "cvta.to.global.u64 %r1, %r3;\n"
                    "cvta.to.global.u64 %r4, %r2;";
    auto trees = parsePtxTrees(s);
    EXPECT_EQ(trees.size(), 2);
    trees[0].first->print();
}
TEST(PtxParser, LoadV2) {
    std::string s = "ld.v2 {%r3, %r2}, [myPtr];\n"
                    "cvta.to.global.u64 %r1, %r3;\n"
                    "cvta.to.global.u64 %r4, %r2;";
    auto trees = parsePtxTrees(s);
    EXPECT_EQ(trees.size(), 2);
    trees[0].first->print();
}

TEST(PtxParser, OriginalComplications) {
    std::ifstream s(TEST_RESOURCE_DIR"/test_ptx");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    auto trees = parsePtxTrees(ptx_data);
    EXPECT_EQ(trees.size(), 2);
    auto empty = trees[0].first->eval(nullptr);
    EXPECT_EQ(empty, nullptr);
    auto par = trees[1].first->eval(nullptr);
    EXPECT_NE(par, nullptr);

    KLaunchConfig config{{3, 3, 0}, {3, 3, 0}};
    auto res = trees[0].first->eval(&config);
    EXPECT_NE(res, nullptr);
    res->print();
}

TEST(PtxParser, MoreComplications) {
    std::ifstream s(TEST_RESOURCE_DIR"/whathappeninthisptx");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    auto trees = parsePtxTrees(ptx_data);

    // There are 5 cvta in the file
    EXPECT_EQ(trees.size(), 5);
    for(const auto& tree: trees) {
        auto evaluated = tree.first->eval(nullptr);
        EXPECT_NE(evaluated, nullptr);
        evaluated->print();
    }
}

TEST(PtxParser, FirstComplications) {
    std::ifstream s(TEST_RESOURCE_DIR"/firstptx");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    auto trees = parsePtxTrees(ptx_data);

    for(const auto& tree: trees) {
        auto first_eval = tree.first->eval(nullptr);
        if(!first_eval) {
            KLaunchConfig kLaunchConfig{{3,3,0},{3,3,0}};
            auto second_eval = tree.first->eval(&kLaunchConfig);
            // Check if all tree actually collapse with a launchConfig
            EXPECT_NE(second_eval, nullptr);
        }
        EXPECT_NE(first_eval, nullptr);
    }
}

TEST(PtxParser, BuggyPtx) {
    std::ifstream s(TEST_RESOURCE_DIR"/buggyPtx");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    auto trees = parsePtxTrees(ptx_data);

    for(const auto& tree: trees) {
        auto first_eval = tree.first->eval(nullptr);
        if(!first_eval) {
            KLaunchConfig kLaunchConfig{{3,3,0},{3,3,0}};
            auto second_eval = tree.first->eval(&kLaunchConfig);
            // Check if all tree actually collapse with a launchConfig
            EXPECT_NE(second_eval, nullptr);
        }
        EXPECT_NE(first_eval, nullptr);
    }
}

TEST(PtxParser, BuggyPtx2) {
    std::ifstream s(TEST_RESOURCE_DIR"/buggyPtx2");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    auto trees = parsePtxTrees(ptx_data);

    for(const auto& tree: trees) {
        auto first_eval = tree.first->eval(nullptr);
        if(!first_eval) {
            KLaunchConfig kLaunchConfig{{3,3,0},{3,3,0}};
            auto second_eval = tree.first->eval(&kLaunchConfig);
            // Check if all tree actually collapse with a launchConfig
            EXPECT_NE(second_eval, nullptr);
        }
        EXPECT_NE(first_eval, nullptr);
    }
}

TEST(PtxParser, BuggyPtx3) {
    std::ifstream s(TEST_RESOURCE_DIR"/buggyPtx3");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    auto trees = parsePtxTrees(ptx_data);

    for(const auto& tree: trees) {
        auto first_eval = tree.first->eval(nullptr);
        if(!first_eval) {
            KLaunchConfig kLaunchConfig{{3,3,0},{3,3,0}};
            auto second_eval = tree.first->eval(&kLaunchConfig);
            // Check if all tree actually collapse with a launchConfig
            EXPECT_NE(second_eval, nullptr);
        }
        EXPECT_NE(first_eval, nullptr);
    }
}

TEST(PtxParser, BuggyPtx4) {
    std::ifstream s(TEST_RESOURCE_DIR"/buggyPtx4");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    auto trees = parsePtxTrees(ptx_data);

    for(const auto& tree: trees) {
        auto first_eval = tree.first->eval(nullptr);
        if(!first_eval) {
            KLaunchConfig kLaunchConfig{{3,3,0},{3,3,0}};
            auto second_eval = tree.first->eval(&kLaunchConfig);
            // Check if all tree actually collapse with a launchConfig
            EXPECT_NE(second_eval, nullptr);
        }
        EXPECT_NE(first_eval, nullptr);
    }
}

TEST(PtxParser, BuggyPtx5) {
    std::ifstream s(TEST_RESOURCE_DIR"/buggyPtx5");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    auto trees = parsePtxTrees(ptx_data);

    for(const auto& tree: trees) {
        auto first_eval = tree.first->eval(nullptr);
        if(!first_eval) {
            KLaunchConfig kLaunchConfig{{3,3,0},{3,3,0}};
            auto second_eval = tree.first->eval(&kLaunchConfig);
            // Check if all tree actually collapse with a launchConfig
            EXPECT_NE(second_eval, nullptr);
        }
        EXPECT_NE(first_eval, nullptr);
    }
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

TEST(FinalTest, TheBigBoy) {
    std::ifstream s(TEST_RESOURCE_DIR"/dumped_ptx");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    std::string::iterator it = ptx_data.begin();
    std::string::iterator file_end = ptx_data.end();

    std::map<std::string, std::vector<UncollapsedKParamInfo>> tmp_map;
    std::string line;

    auto getline = [](std::string::iterator &beg,
                       std::string::iterator end, std::string &line) {
        if (beg == end)
            return false;
        char c;
        line.clear();
        while ((beg != end) && ((c = *(beg++)) != '\n')) {
            line.push_back(c);
        }
        return true;
    };

    while (getline(it, file_end, line)) {
        if(line.empty())
            continue;
        if (line.find(".entry") == std::string::npos)
            continue;

        std::string entry = std::string(ptx_split_string(line, " ").back());

        //TODO: implement bfi operation
        if(ptxStartsWith(entry, "_ZN2at6native52_GLOBAL__N__2ec533aa_19_FakeQuantizeCore_cu_163ccd5445unrolled_elementwise_kernel_for_multi_outputs"))
            continue;
        // buggyPtx6
        if(ptxStartsWith(entry, "_ZN2at6native57_GLOBAL__N__39d96b06_24_ActivationPreluKernel_cu_b2f5ef0e45unrolled_elementwise_kernel_for_multi_outputsILi2EZZZNS0_47launch_prelu_cuda_backward_kernel_share_weightsERNS_18TensorIteratorBaseERKNS_10TensorBaseEENKUlvE_clEvENKUlvE0_clEvEUlddE_NS_6detail5ArrayIPcLi4EEE16OffsetCalculatorILi2EjLb0EESG_EEviT0_T1_T2_T3_"))
            continue;

        //TODO: mov operations can have weird registers names, but do we really care ?
        // IMO we could just ignore the instructions that have dst that
        // are not of interest and only then convert right side
        // buggyPtx1 and buggyPtx4
        if(ptxStartsWith(entry, "_ZN6caffe250_GLOBAL__N__ebbf0d44_10_relu_op_cu_f0da4091_12894314ReluCUDAKernelI7__half2EEviPKT_PS3_"))
            continue;
        if(ptxStartsWith(entry, "_ZN6caffe256_GLOBAL__N__d4835dcf_23_fp16_momentum_sgd_op_cu_8083ef1b25FP16MomentumSGDFP32KernelEiPK7__half2S3_PS1_S4_PKffbfS4_"))
            continue;

        //TODO: CVTA inside of a branch, in this case we could ignore it I guess
        // buggyPtx2
        if(ptxStartsWith(entry, "_ZN6caffe231ScaleBlobsCUDAKernelManyTensorsIfEEvfPKiPPT_S5_"))
            continue;
        //TODO: CVTA inside branches, level: hard
        // buggyPtx9
        if(ptxStartsWith(entry, "_ZN2at6native18elementwise_kernel"))
            continue;

        //TODO: What about loopings cvtas ? Also I guess checking for the next
        // .entry is not optimal.
        // buggyPtx3
        if(ptxStartsWith(entry, "_ZN6caffe220ScaleBlobsCUDAKernelIfEEvfiPKiPPT_S5_"))
            continue;

        //TODO: Global variables can be leaves of cvta
        // buggyPtx5
        if(ptxStartsWith(entry, "_ZN6caffe210gatherTopKIfLb1ElEEvPKT_iiiPS1_PT1_"))
            continue;

        //TODO: Somehow %rd14 is not a register of interest though used in an add?
        // ld.param on a register then add
        // buggyPtx7
        if(ptxStartsWith(entry, "_ZN2at6native46_GLOBAL__N__c660ad18_13_AmpKernels_cu_3810c27325"))
            continue;
        // buggyPtx10
        if(ptxStartsWith(entry, "_ZN2at6native55_GLOBAL__N__e92d5f40_22_ForeachBinaryOpList_cu_"))
            continue;
        if(ptxStartsWith(entry, "_ZN2at6native57_GLOBAL__N__8b59e72f_24_"))
            continue;

        //TODO: cvta.to.local != cvta.to.global !
        // BuggyPtx8
        if(ptxStartsWith(entry, "_ZN2at6native29vectorized_elementwise_kernel"))
            continue;
        if(ptxStartsWith(entry, "_ZN2at6native27unrolled_elementwise_kernel"))
            continue;

        entry.pop_back();
        auto entry_beg = it;
        while (getline(it, file_end, line) && line.find(".entry") == std::string::npos)
            ;
        // Go up one line, as we already found the next entry
        while(*(--it) != '\n')
            ;
        auto entry_end = it;
        std::string entry_lines(entry_beg, entry_end);

        auto trees = parsePtxTrees(entry_lines);
        KLaunchConfig kLaunchConfig{{3,3,0},{3,3,0}};
        for(auto& tree: trees) {
            auto first_eval = tree.first->eval(nullptr);
            if(!first_eval) {
                auto second_eval = tree.first->eval(&kLaunchConfig);
                // Check if all tree actually collapse if launchConfig is given
                EXPECT_NE(second_eval, nullptr);
                continue;
            }
            EXPECT_NE(first_eval, nullptr);
        }


    }

}


