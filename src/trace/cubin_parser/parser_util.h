#ifndef GPULESS_PARSER_UTIL_H
#define GPULESS_PARSER_UTIL_H

#include <array>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>


std::vector<std::string_view> splitString(std::string_view str,
                                      const std::string &delimiter);
std::vector<std::string> splitStringCopy(const std::string &str,
                                          const std::string &delimiter);
bool startsWith(const std::string_view &str, const std::string_view &prefix);
bool endsWith(const std::string_view &str, const std::string_view &suffix);
std::string_view rgetline(const std::string_view::iterator &beg,
              std::string_view::iterator &end);

#endif // GPULESS_PARSER_UTIL_H
