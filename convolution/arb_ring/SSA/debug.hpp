#include <cassert>
#include <iostream>
#include <span>
#include <vector>

std::string pad(std::string s, int n) {
    while (s.size() < n) s += ' ';
    return s;
}

#define debug(x) std::cerr << __FILE__ << ":" << __LINE__ << "  " << pad(#x, 5) << " = " << (x)
#define dbg(x) "  " << pad(#x, 0) << " = " << (x)

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::span<T>& vec) {
    out << '{';
    for (int i = 0; i < vec.size(); i++) {
        out << vec[i];
        if (i + 1 != vec.size()) {
            out << ", ";
        }
    }
    return out << '}';
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
    return out << std::span(vec);
}
