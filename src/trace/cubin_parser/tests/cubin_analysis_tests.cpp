#include <fstream>
#include <gtest/gtest.h>

#include "../../cubin_analysis.hpp"

TEST(CubinAnalyis, BasicFunctionality) {
    std::ifstream s(TEST_RESOURCE_DIR"/test_ptx_with_param");
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();

    auto res = CubinAnalyzer::parsePtxParameters(ptx_data.begin(), ptx_data.end());

    EXPECT_EQ(res.size(), 5);

    PtxTreeParser::KLaunchConfig config{{3,3,0}, {3,3,0}};
    auto eval = res[1].trees[0].first->eval(&config);
    EXPECT_NE(eval, nullptr);
    eval->print();
}

TEST(CubinAnalyis, FullPtx) {
    std::string const &f = TEST_RESOURCE_DIR"/test_regular_ptx";
    std::ifstream s(f);
    std::stringstream ss;
    ss << s.rdbuf();
    std::string ptx_data = ss.str();

    CubinAnalyzer analyzer;
    auto data = analyzer.analyzeTxt(ptx_data, 1,2);
    EXPECT_EQ(data.size(), 2);
}
