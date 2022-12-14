#ifndef GPULESS_TREE_PARSER_H
#define GPULESS_TREE_PARSER_H

#include <array>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../cubin_analysis.hpp"
#include "ptx_tree.h"

namespace PtxTreeParser {

std::vector<std::pair<std::unique_ptr<PtxTree>, std::string>>
parsePtxTrees(std::string_view ss);

} // namespace PtxTreeParser

#endif // GPULESS_TREE_PARSER_H
