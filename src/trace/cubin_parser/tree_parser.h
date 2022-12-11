#ifndef GPULESS_TREE_PARSER_H
#define GPULESS_TREE_PARSER_H

#include <array>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ptx_tree.h"
#include "../cubin_analysis.hpp"


namespace PtxTreeParser {

std::vector<PtxTree> parsePtxTrees(std::string &ss);

} // namespace PtxTreeParser

#endif // GPULESS_TREE_PARSER_H
