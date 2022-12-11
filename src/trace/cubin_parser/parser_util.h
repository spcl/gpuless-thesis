#ifndef GPULESS_PARSER_UTIL_H
#define GPULESS_PARSER_UTIL_H

#include <array>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

std::vector<std::string> split_string(std::string str,
                                      const std::string &delimiter);
bool startsWith(const std::string &str, const std::string &prefix);
bool endsWith(const std::string &str, const std::string &suffix);
bool rgetline(std::string::reverse_iterator &it,
              const std::string::reverse_iterator &end, std::string &line);

#endif // GPULESS_PARSER_UTIL_H
