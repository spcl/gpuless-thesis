#include "parser_util.h"

#include <algorithm>
#include <sstream>

std::vector<std::string> split_string(std::string str,
                                      const std::string &delimiter) {
    std::vector<std::string> result;

    size_t pos = 0;
    std::string token;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        token = str.substr(0, pos);
        result.push_back(token);
        str.erase(0, pos + delimiter.length());
    }
    result.push_back(str);
    return result;
}

bool startsWith(const std::string &str, const std::string &prefix) {
    return str.rfind(prefix, 0) == 0;
}

bool endsWith(const std::string &str, const std::string &suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool rgetline(std::string::reverse_iterator &it,
              const std::string::reverse_iterator &end, std::string &line) {
    if (it == end)
        return false;
    std::stringstream ss;
    char t;
    while (it != end && (t = *it++) != '\n') {
        ss << t;
    }
    line = ss.str();
    std::reverse(line.begin(), line.end());
    return true;
}