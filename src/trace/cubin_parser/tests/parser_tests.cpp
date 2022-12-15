#include "../../cubin_analysis.hpp"
#include "../tree_parser.h"
#include "../parser_util.h"
#include <fstream>
#include <gtest/gtest.h>

using namespace PtxTreeParser;

TEST(PtxTree, BasicTree) {
    PtxTree test_tree("test_register");
    test_tree.add_node(PtxNodeKind::AddOp,
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
    test_tree.add_node(PtxNodeKind::AddOp,
                       {{PtxOperandKind::Register, "test_register",0},
                        {PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::Register, "reg_b",0}});

    // Add Immediate 42 to left
    test_tree.add_node(PtxNodeKind::AddOp,
                       {{PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::Immediate, "", 22},
                        {PtxOperandKind::Immediate, "", 20}});

    // Add another add on right
    test_tree.add_node(PtxNodeKind::AddOp,
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
    test_tree.add_node(PtxNodeKind::AddOp,
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
    test_tree.add_node(PtxNodeKind::AddOp,
                       {{PtxOperandKind::Register, "test_register",0},
                        {PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::Immediate, "",10}});

    test_tree.add_node(PtxNodeKind::AddOp,
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
    test_tree.add_node(PtxNodeKind::SubOp,
                       {{PtxOperandKind::Register, "test_register",0},
                        {PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::Register, "reg_b",0}});

    test_tree.add_node(PtxNodeKind::SubOp,
                       {{PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::Parameter, "test_param", 20},
                        {PtxOperandKind::Immediate, "", 10}});

    test_tree.add_node(PtxNodeKind::SubOp,
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
    std::string s = "mov %r2, [myPtr];\n"
                    "cvta.to.global.u64 %r1, %r2;";
    PtxTreeParser::TreeParser parser({"myPtr"});
    auto trees = parser.parsePtxTrees(s);

    EXPECT_EQ(trees.size(), 1);

    trees[0].first->print();

}

TEST(PtxParser, ParseAdd) {
    std::string s = "add %r2, [par+4], 2;\n"
                    "cvta.to.global.u64 %r1, %r2;";
    PtxTreeParser::TreeParser parser({"par"});
    auto trees = parser.parsePtxTrees(s);
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
    PtxTreeParser::TreeParser parser({"myPtr"});
    auto trees = parser.parsePtxTrees(s);
    EXPECT_EQ(trees.size(), 1);
    trees[0].first->print();
}

TEST(PtxParser, ParseAddV2) {
    std::string s = "add.v2 {%r3, %r2}, [myPtr+2], 4;\n"
                    "cvta.to.global.u64 %r1, %r3;\n"
                    "cvta.to.global.u64 %r4, %r2;";
    PtxTreeParser::TreeParser parser({"myPtr"});
    auto trees = parser.parsePtxTrees(s);
    EXPECT_EQ(trees.size(), 2);
    trees[0].first->print();
}
TEST(PtxParser, LoadV2) {
    std::string s = "ld.v2 {%r3, %r2}, [myPtr];\n"
                    "cvta.to.global.u64 %r1, %r3;\n"
                    "cvta.to.global.u64 %r4, %r2;";
    PtxTreeParser::TreeParser parser({"myPtr"});
    auto trees = parser.parsePtxTrees(s);
    EXPECT_EQ(trees.size(), 2);
    trees[0].first->print();
}

TEST(PtxParser, OriginalComplications) {
    std::ifstream s(TEST_RESOURCE_DIR"/test_ptx");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    PtxTreeParser::TreeParser parser({
        "_ZN2at6native40_GLOBAL__N__73da4bea_8_Shape_cu_49f7391c19CatArrayBatchedCopyIfjLi3ELi128ELi1EEEvPT_NS1_25CatArrInputTensorMetadataIS3_T0_XT2_EXT3_EEENS1_16TensorSizeStrideIS6_Lj4EEEiS6__param_0",
        "_ZN2at6native40_GLOBAL__N__73da4bea_8_Shape_cu_49f7391c19CatArrayBatchedCopyIfjLi3ELi128ELi1EEEvPT_NS1_25CatArrInputTensorMetadataIS3_T0_XT2_EXT3_EEENS1_16TensorSizeStrideIS6_Lj4EEEiS6__param_1",
        "_ZN2at6native40_GLOBAL__N__73da4bea_8_Shape_cu_49f7391c19CatArrayBatchedCopyIfjLi3ELi128ELi1EEEvPT_NS1_25CatArrInputTensorMetadataIS3_T0_XT2_EXT3_EEENS1_16TensorSizeStrideIS6_Lj4EEEiS6__param_2",
        "_ZN2at6native40_GLOBAL__N__73da4bea_8_Shape_cu_49f7391c19CatArrayBatchedCopyIfjLi3ELi128ELi1EEEvPT_NS1_25CatArrInputTensorMetadataIS3_T0_XT2_EXT3_EEENS1_16TensorSizeStrideIS6_Lj4EEEiS6__param_3",
        "_ZN2at6native40_GLOBAL__N__73da4bea_8_Shape_cu_49f7391c19CatArrayBatchedCopyIfjLi3ELi128ELi1EEEvPT_NS1_25CatArrInputTensorMetadataIS3_T0_XT2_EXT3_EEENS1_16TensorSizeStrideIS6_Lj4EEEiS6__param_4"
    });
    auto trees = parser.parsePtxTrees(ptx_data);
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
    PtxTreeParser::TreeParser parser({
        "_ZN2at6native45_GLOBAL__N__cc2aaa4d_12_attention_cu_6f94629945transform_bias_rescale_qkv_add_padding_kernelIddLb1EEEvNS_27GenericPackedTensorAccessorIT_Lm1ENS_17RestrictPtrTraitsElEES6_PKiS8_NS3_IS4_Lm5ES5_lEES4__param_0",
        "_ZN2at6native45_GLOBAL__N__cc2aaa4d_12_attention_cu_6f94629945transform_bias_rescale_qkv_add_padding_kernelIddLb1EEEvNS_27GenericPackedTensorAccessorIT_Lm1ENS_17RestrictPtrTraitsElEES6_PKiS8_NS3_IS4_Lm5ES5_lEES4__param_1",
        "_ZN2at6native45_GLOBAL__N__cc2aaa4d_12_attention_cu_6f94629945transform_bias_rescale_qkv_add_padding_kernelIddLb1EEEvNS_27GenericPackedTensorAccessorIT_Lm1ENS_17RestrictPtrTraitsElEES6_PKiS8_NS3_IS4_Lm5ES5_lEES4__param_2",
        "_ZN2at6native45_GLOBAL__N__cc2aaa4d_12_attention_cu_6f94629945transform_bias_rescale_qkv_add_padding_kernelIddLb1EEEvNS_27GenericPackedTensorAccessorIT_Lm1ENS_17RestrictPtrTraitsElEES6_PKiS8_NS3_IS4_Lm5ES5_lEES4__param_3",
        "_ZN2at6native45_GLOBAL__N__cc2aaa4d_12_attention_cu_6f94629945transform_bias_rescale_qkv_add_padding_kernelIddLb1EEEvNS_27GenericPackedTensorAccessorIT_Lm1ENS_17RestrictPtrTraitsElEES6_PKiS8_NS3_IS4_Lm5ES5_lEES4__param_4",
        "_ZN2at6native45_GLOBAL__N__cc2aaa4d_12_attention_cu_6f94629945transform_bias_rescale_qkv_add_padding_kernelIddLb1EEEvNS_27GenericPackedTensorAccessorIT_Lm1ENS_17RestrictPtrTraitsElEES6_PKiS8_NS3_IS4_Lm5ES5_lEES4__param_5"
    });
    auto trees = parser.parsePtxTrees(ptx_data);

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
    PtxTreeParser::TreeParser parser({
        "_ZN6caffe24math52_GLOBAL__N__d361319e_12_broadcast_cu_f0da4091_12375027AffineChannelNCHWCUDAKernelIfEEviiiPKT_S5_S5_PS3__param_0",
        "_ZN6caffe24math52_GLOBAL__N__d361319e_12_broadcast_cu_f0da4091_12375027AffineChannelNCHWCUDAKernelIfEEviiiPKT_S5_S5_PS3__param_1",
        "_ZN6caffe24math52_GLOBAL__N__d361319e_12_broadcast_cu_f0da4091_12375027AffineChannelNCHWCUDAKernelIfEEviiiPKT_S5_S5_PS3__param_2",
        "_ZN6caffe24math52_GLOBAL__N__d361319e_12_broadcast_cu_f0da4091_12375027AffineChannelNCHWCUDAKernelIfEEviiiPKT_S5_S5_PS3__param_3",
        "_ZN6caffe24math52_GLOBAL__N__d361319e_12_broadcast_cu_f0da4091_12375027AffineChannelNCHWCUDAKernelIfEEviiiPKT_S5_S5_PS3__param_4",
        "_ZN6caffe24math52_GLOBAL__N__d361319e_12_broadcast_cu_f0da4091_12375027AffineChannelNCHWCUDAKernelIfEEviiiPKT_S5_S5_PS3__param_5",
        "_ZN6caffe24math52_GLOBAL__N__d361319e_12_broadcast_cu_f0da4091_12375027AffineChannelNCHWCUDAKernelIfEEviiiPKT_S5_S5_PS3__param_6"
    });
    auto trees = parser.parsePtxTrees(ptx_data);

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
    PtxTreeParser::TreeParser parser({
        "_ZN6caffe250_GLOBAL__N__ebbf0d44_10_relu_op_cu_f0da4091_12894314ReluCUDAKernelI7__half2EEviPKT_PS3__param_0",
        "_ZN6caffe250_GLOBAL__N__ebbf0d44_10_relu_op_cu_f0da4091_12894314ReluCUDAKernelI7__half2EEviPKT_PS3__param_1",
        "_ZN6caffe250_GLOBAL__N__ebbf0d44_10_relu_op_cu_f0da4091_12894314ReluCUDAKernelI7__half2EEviPKT_PS3__param_2"
    });
    auto trees = parser.parsePtxTrees(ptx_data);

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
    PtxTreeParser::TreeParser parser({
        "_ZN6caffe231ScaleBlobsCUDAKernelManyTensorsIfEEvfPKiPPT_S5__param_0",
        "_ZN6caffe231ScaleBlobsCUDAKernelManyTensorsIfEEvfPKiPPT_S5__param_1",
        "_ZN6caffe231ScaleBlobsCUDAKernelManyTensorsIfEEvfPKiPPT_S5__param_2",
        "_ZN6caffe231ScaleBlobsCUDAKernelManyTensorsIfEEvfPKiPPT_S5__param_3"
    });
    auto trees = parser.parsePtxTrees(ptx_data);

    for(const auto& tree: trees) {
        auto first_eval = tree.first->eval(nullptr);
        if(!first_eval) {
            KLaunchConfig kLaunchConfig{{3,3,0},{3,3,0}};
            auto second_eval = tree.first->eval(&kLaunchConfig);
            // Check if all tree actually collapse with a launchConfig
            EXPECT_NE(second_eval, nullptr);
        }
    }
}

TEST(PtxParser, BuggyPtx3) {
    std::ifstream s(TEST_RESOURCE_DIR"/buggyPtx3");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    PtxTreeParser::TreeParser parser({
        "_ZN6caffe220ScaleBlobsCUDAKernelIfEEvfiPKiPPT_S5__param_0",
        "_ZN6caffe220ScaleBlobsCUDAKernelIfEEvfiPKiPPT_S5__param_1",
        "_ZN6caffe220ScaleBlobsCUDAKernelIfEEvfiPKiPPT_S5__param_2",
        "_ZN6caffe220ScaleBlobsCUDAKernelIfEEvfiPKiPPT_S5__param_3",
        "_ZN6caffe220ScaleBlobsCUDAKernelIfEEvfiPKiPPT_S5__param_4"
    });
    auto trees = parser.parsePtxTrees(ptx_data);

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
    PtxTreeParser::TreeParser parser({
        "_ZN6caffe256_GLOBAL__N__d4835dcf_23_fp16_momentum_sgd_op_cu_8083ef1b25FP16MomentumSGDFP32KernelEiPK7__half2S3_PS1_S4_PKffbfS4__param_0",
        "_ZN6caffe256_GLOBAL__N__d4835dcf_23_fp16_momentum_sgd_op_cu_8083ef1b25FP16MomentumSGDFP32KernelEiPK7__half2S3_PS1_S4_PKffbfS4__param_1",
        "_ZN6caffe256_GLOBAL__N__d4835dcf_23_fp16_momentum_sgd_op_cu_8083ef1b25FP16MomentumSGDFP32KernelEiPK7__half2S3_PS1_S4_PKffbfS4__param_2",
        "_ZN6caffe256_GLOBAL__N__d4835dcf_23_fp16_momentum_sgd_op_cu_8083ef1b25FP16MomentumSGDFP32KernelEiPK7__half2S3_PS1_S4_PKffbfS4__param_3",
        "_ZN6caffe256_GLOBAL__N__d4835dcf_23_fp16_momentum_sgd_op_cu_8083ef1b25FP16MomentumSGDFP32KernelEiPK7__half2S3_PS1_S4_PKffbfS4__param_4",
        "_ZN6caffe256_GLOBAL__N__d4835dcf_23_fp16_momentum_sgd_op_cu_8083ef1b25FP16MomentumSGDFP32KernelEiPK7__half2S3_PS1_S4_PKffbfS4__param_5",
        "_ZN6caffe256_GLOBAL__N__d4835dcf_23_fp16_momentum_sgd_op_cu_8083ef1b25FP16MomentumSGDFP32KernelEiPK7__half2S3_PS1_S4_PKffbfS4__param_6",
        "_ZN6caffe256_GLOBAL__N__d4835dcf_23_fp16_momentum_sgd_op_cu_8083ef1b25FP16MomentumSGDFP32KernelEiPK7__half2S3_PS1_S4_PKffbfS4__param_7",
        "_ZN6caffe256_GLOBAL__N__d4835dcf_23_fp16_momentum_sgd_op_cu_8083ef1b25FP16MomentumSGDFP32KernelEiPK7__half2S3_PS1_S4_PKffbfS4__param_8",
        "_ZN6caffe256_GLOBAL__N__d4835dcf_23_fp16_momentum_sgd_op_cu_8083ef1b25FP16MomentumSGDFP32KernelEiPK7__half2S3_PS1_S4_PKffbfS4__param_9"
    });
    auto trees = parser.parsePtxTrees(ptx_data);

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
    PtxTreeParser::TreeParser parser({
        "_ZN6caffe210gatherTopKIfLb1ElEEvPKT_iiiPS1_PT1__param_0",
        "_ZN6caffe210gatherTopKIfLb1ElEEvPKT_iiiPS1_PT1__param_1",
        "_ZN6caffe210gatherTopKIfLb1ElEEvPKT_iiiPS1_PT1__param_2",
        "_ZN6caffe210gatherTopKIfLb1ElEEvPKT_iiiPS1_PT1__param_3",
        "_ZN6caffe210gatherTopKIfLb1ElEEvPKT_iiiPS1_PT1__param_4",
        "_ZN6caffe210gatherTopKIfLb1ElEEvPKT_iiiPS1_PT1__param_5"
    });
    auto trees = parser.parsePtxTrees(ptx_data);

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

TEST(PtxParser, BuggyPtx7) {
    std::ifstream s(TEST_RESOURCE_DIR"/buggyPtx7");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    PtxTreeParser::TreeParser parser({
        "_ZN2at6native46_GLOBAL__N__c660ad18_13_AmpKernels_cu_3810c27325multi_tensor_apply_kernelINS1_18TensorListMetadataILi1EEENS1_14UnaryOpFunctorIfLi1ELi1ELi0EEEJZZZNS0_47_amp_foreach_non_finite_check_and_unscale_cuda_EN3c108ArrayRefINS_6TensorEEERS9_RKS9_ENKUlvE_clEvENKUlvE2_clEvEUlfE_EEEvT_T0_DpT1__param_0",
        "_ZN2at6native46_GLOBAL__N__c660ad18_13_AmpKernels_cu_3810c27325multi_tensor_apply_kernelINS1_18TensorListMetadataILi1EEENS1_14UnaryOpFunctorIfLi1ELi1ELi0EEEJZZZNS0_47_amp_foreach_non_finite_check_and_unscale_cuda_EN3c108ArrayRefINS_6TensorEEERS9_RKS9_ENKUlvE_clEvENKUlvE2_clEvEUlfE_EEEvT_T0_DpT1__param_1",
        "_ZN2at6native46_GLOBAL__N__c660ad18_13_AmpKernels_cu_3810c27325multi_tensor_apply_kernelINS1_18TensorListMetadataILi1EEENS1_14UnaryOpFunctorIfLi1ELi1ELi0EEEJZZZNS0_47_amp_foreach_non_finite_check_and_unscale_cuda_EN3c108ArrayRefINS_6TensorEEERS9_RKS9_ENKUlvE_clEvENKUlvE2_clEvEUlfE_EEEvT_T0_DpT1__param_2"
    });
    auto trees = parser.parsePtxTrees(ptx_data);

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

TEST(PtxParser, BuggyPtx8) {
    std::ifstream s(TEST_RESOURCE_DIR"/buggyPtx8");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    PtxTreeParser::TreeParser parser({
        "_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_56_GLOBAL__N__970b6095_16_ComplexKernel_cu_f0da4091_13469217polar_kernel_cudaERNS_14TensorIteratorEENKUlvE_clEvENKUlvE2_clEvEUlffE_NS_6detail5ArrayIPcLi3EEEEEviT0_T1__param_1",
        "_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_56_GLOBAL__N__970b6095_16_ComplexKernel_cu_f0da4091_13469217polar_kernel_cudaERNS_14TensorIteratorEENKUlvE_clEvENKUlvE2_clEvEUlffE_NS_6detail5ArrayIPcLi3EEEEEviT0_T1__param_2"
    });
    auto trees = parser.parsePtxTrees(ptx_data);

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

TEST(PtxParser, BuggyPtx9) {
    std::ifstream s(TEST_RESOURCE_DIR"/buggyPtx9");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    PtxTreeParser::TreeParser parser({
        "_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_15gpu_kernel_implIZZZNS0_56_GLOBAL__N__970b6095_16_ComplexKernel_cu_f0da4091_13469217polar_kernel_cudaERNS_14TensorIteratorEENKUlvE_clEvENKUlvE2_clEvEUlffE_EEvRNS_18TensorIteratorBaseERKT_EUliE_EEviT1__param_0",
        "_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_15gpu_kernel_implIZZZNS0_56_GLOBAL__N__970b6095_16_ComplexKernel_cu_f0da4091_13469217polar_kernel_cudaERNS_14TensorIteratorEENKUlvE_clEvENKUlvE2_clEvEUlffE_EEvRNS_18TensorIteratorBaseERKT_EUliE_EEviT1__param_1"
    });
    auto trees = parser.parsePtxTrees(ptx_data);

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

TEST(PtxParser, BuggyPtx12) {
    std::ifstream s(TEST_RESOURCE_DIR"/buggyPtx12");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    PtxTreeParser::TreeParser parser({
        "_ZN2at6native45_GLOBAL__N__cc2aaa4d_12_attention_cu_6f94629945transform_bias_rescale_qkv_add_padding_kernelIddLb1EEEvNS_27GenericPackedTensorAccessorIT_Lm1ENS_17RestrictPtrTraitsElEES6_PKiS8_NS3_IS4_Lm5ES5_lEES4__param_0",
        "_ZN2at6native45_GLOBAL__N__cc2aaa4d_12_attention_cu_6f94629945transform_bias_rescale_qkv_add_padding_kernelIddLb1EEEvNS_27GenericPackedTensorAccessorIT_Lm1ENS_17RestrictPtrTraitsElEES6_PKiS8_NS3_IS4_Lm5ES5_lEES4__param_1",
        "_ZN2at6native45_GLOBAL__N__cc2aaa4d_12_attention_cu_6f94629945transform_bias_rescale_qkv_add_padding_kernelIddLb1EEEvNS_27GenericPackedTensorAccessorIT_Lm1ENS_17RestrictPtrTraitsElEES6_PKiS8_NS3_IS4_Lm5ES5_lEES4__param_2",
        "_ZN2at6native45_GLOBAL__N__cc2aaa4d_12_attention_cu_6f94629945transform_bias_rescale_qkv_add_padding_kernelIddLb1EEEvNS_27GenericPackedTensorAccessorIT_Lm1ENS_17RestrictPtrTraitsElEES6_PKiS8_NS3_IS4_Lm5ES5_lEES4__param_3",
        "_ZN2at6native45_GLOBAL__N__cc2aaa4d_12_attention_cu_6f94629945transform_bias_rescale_qkv_add_padding_kernelIddLb1EEEvNS_27GenericPackedTensorAccessorIT_Lm1ENS_17RestrictPtrTraitsElEES6_PKiS8_NS3_IS4_Lm5ES5_lEES4__param_4",
        "_ZN2at6native45_GLOBAL__N__cc2aaa4d_12_attention_cu_6f94629945transform_bias_rescale_qkv_add_padding_kernelIddLb1EEEvNS_27GenericPackedTensorAccessorIT_Lm1ENS_17RestrictPtrTraitsElEES6_PKiS8_NS3_IS4_Lm5ES5_lEES4__param_5"
    });
    auto trees = parser.parsePtxTrees(ptx_data);

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

TEST(PtxParser, SlowPtx) {
    std::ifstream s(TEST_RESOURCE_DIR"/slowPtx");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();
    PtxTreeParser::TreeParser parser({
        "_ZN6caffe24math54_GLOBAL__N__beabb26c_14_elementwise_cu_f0da4091_12388116SinCosCUDAKernelIfEEviPKT_PS3_S6__param_0",
        "_ZN6caffe24math54_GLOBAL__N__beabb26c_14_elementwise_cu_f0da4091_12388116SinCosCUDAKernelIfEEviPKT_PS3_S6__param_1",
        "_ZN6caffe24math54_GLOBAL__N__beabb26c_14_elementwise_cu_f0da4091_12388116SinCosCUDAKernelIfEEviPKT_PS3_S6__param_2",
        "_ZN6caffe24math54_GLOBAL__N__beabb26c_14_elementwise_cu_f0da4091_12388116SinCosCUDAKernelIfEEviPKT_PS3_S6__param_3"
    });
    auto trees = parser.parsePtxTrees(ptx_data);

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
    test_tree.add_node(PtxNodeKind::SubOp,
                       {{PtxOperandKind::Register, "test_register",0},
                        {PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::Register, "reg_b",0}});

    test_tree.add_node(PtxNodeKind::SubOp,
                       {{PtxOperandKind::Register, "reg_a",0},
                        {PtxOperandKind::Parameter, "test_param", 20},
                        {PtxOperandKind::Immediate, "", 10}});

    test_tree.add_node(PtxNodeKind::SubOp,
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

std::string_view range_to_view(std::string::iterator first,
                               std::string::iterator last) {
    if (first != last)
        return {first.operator->(), static_cast<size_t>(last - first)};
    else
        return {nullptr, 0};
}

TEST(FinalTest, TheBigBoy) {
    int cnt = 0, instant_eval = 0, kernel_count = 0, error_cnt = 0;
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

        std::string entry = std::string(splitString(line, " ").back());

        /*//TODO: implement bfi operation
        if(startsWith(
                entry,
                "_ZN2at6native52_GLOBAL__N__2ec533aa_19_FakeQuantizeCore_cu_163ccd5445unrolled_elementwise_kernel_for_multi_outputs"))
            continue;
        // buggyPtx6
        if(startsWith(
                entry,
                "_ZN2at6native57_GLOBAL__N__39d96b06_24_ActivationPreluKernel_cu_b2f5ef0e45unrolled_elementwise_kernel_for_multi_outputsILi2EZZZNS0_47launch_prelu_cuda_backward_kernel_share_weightsERNS_18TensorIteratorBaseERKNS_10TensorBaseEENKUlvE_clEvENKUlvE0_clEvEUlddE_NS_6detail5ArrayIPcLi4EEE16OffsetCalculatorILi2EjLb0EESG_EEviT0_T1_T2_T3_"))
            continue;

        //TODO: CVTA inside of a branch, in this case we could ignore it I guess
        // buggyPtx2
        //if(startsWith(
        //        entry,
        //        "_ZN6caffe231ScaleBlobsCUDAKernelManyTensorsIfEEvfPKiPPT_S5_"))
        //    continue;
        //TODO: CVTA inside branches, level: easy
        // buggyPtx9
        //if(startsWith(entry, "_ZN2at6native18elementwise_kernel"))
        //    continue;

        //TODO: What about loopings cvtas ? Also I guess checking for the next
        // .entry is not optimal.
        // buggyPtx3
        //if(startsWith(entry,
        //               "_ZN6caffe220ScaleBlobsCUDAKernelIfEEvfiPKiPPT_S5_"))
        //    continue;

        //TODO: Global variables can be leaves of cvta
        // buggyPtx5
        //if(startsWith(entry,
        //               "_ZN6caffe210gatherTopKIfLb1ElEEvPKT_iiiPS1_PT1_"))
        //    continue;

        //TODO: Somehow %rd14 is not a register of interest though used in an add?
        // ld.param on a register then add
        // buggyPtx7
        if(startsWith(entry, "_ZN2at6native46_GLOBAL__N__c660ad18_13_AmpKernels_cu_3810c27325"))
            continue;
        // buggyPtx10
        if(startsWith(entry, "_ZN2at6native55_GLOBAL__N__e92d5f40_22_ForeachBinaryOpList_cu_"))
            continue;
        if(startsWith(entry, "_ZN2at6native57_GLOBAL__N__8b59e72f_24_"))
            continue;
*/
        ++kernel_count;

        entry.pop_back();
        auto entry_beg = it;
        while (getline(it, file_end, line)) {
            if(line.find(".entry") != std::string::npos) {
                while(*(--it) != '\n')
                    ;
                break;
            }
        }
        auto entry_end = it;

        // Extract raw parameters from ptx
        std::unordered_set<std::string> param_names;

        while (getline(entry_beg, entry_end, line)) {
            if (line.find(')') != std::string::npos) {
                break;
            }
            // NO parameters
            if (!startsWith(line, ".param")) {
                break;
            }
            auto splitted_line = splitString(line, " ");

            // Remove last comma
            auto last = splitted_line.back();
            if (endsWith(last, ",")) {
                splitted_line.back() = last.substr(0, last.size() - 1);
            }

            if (splitted_line[1] == ".align") {
                int param_align = std::stoi(splitted_line[2].data());
                const std::string_view &name = splitted_line[4];
                std::vector<std::string_view> splitted_name =
                    splitString(name, "[");
                const std::string_view &param_name = splitted_name[0];
                param_names.insert(std::string(param_name));
            } else {
                std::string_view &name = splitted_line[2];
                param_names.insert(std::string(name));
            }
        }

        PtxTreeParser::TreeParser parser{param_names};
        auto trees = parser.parsePtxTrees(range_to_view(entry_beg, entry_end));
        KLaunchConfig kLaunchConfig{{3,3,0},{3,3,0}};
        for(auto& tree: trees) {
            ++cnt;
            try{
                auto first_eval = tree.first->eval(nullptr);
                if(!first_eval) {
                    auto second_eval = tree.first->eval(&kLaunchConfig);
                    // Check if all tree actually collapse if launchConfig is given
                    EXPECT_NE(second_eval, nullptr);
                    continue;
                } else {
                    ++instant_eval;
                }
            }catch(...){
                ++error_cnt;
            }

        }


    }

    std::cout << "Number of parameters analyzed: " << cnt << ", Number of parameters without tree: " << instant_eval
    << ", number of parameters not working: " << error_cnt << std::endl;
}


