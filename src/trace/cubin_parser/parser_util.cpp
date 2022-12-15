#include "parser_util.h"

#include <algorithm>
#include <sstream>

std::vector<std::string_view> splitString(std::string_view str,
                                      const std::string &delimiter) {
    std::vector<std::string_view> result;

    size_t old_pos = 0;
    for(size_t pos = 0; (pos = str.find(delimiter, old_pos )) != std::string::npos; old_pos = pos+1) {
        result.push_back(str.substr(old_pos, pos - old_pos));
    }
    result.push_back(str.substr(old_pos, str.size() - old_pos));

    return result;
}

std::vector<std::string> splitStringCopy(const std::string& str,
                                          const std::string &delimiter) {
    std::vector<std::string> result;

    size_t old_pos = 0;
    for(size_t pos = 0; (pos = str.find(delimiter, old_pos )) != std::string::npos; old_pos = pos+1) {
        result.push_back(str.substr(old_pos, pos - old_pos));
    }
    result.push_back(str.substr(old_pos, str.size() - old_pos));

    return result;
}

bool startsWith(const std::string_view &str, const std::string_view &prefix) {
    return str.rfind(prefix, 0) == 0;
}

bool endsWith(const std::string_view &str, const std::string_view &suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string_view rgetline(const std::string_view::iterator &beg,
              std::string_view::iterator &end) {

    for(size_t len = 0; true ;++len) {
        if(end == beg) {
             return {end, len};
        }

        if(*(--end) == '\n') {
            return {end+1, len};
        }
    }
}