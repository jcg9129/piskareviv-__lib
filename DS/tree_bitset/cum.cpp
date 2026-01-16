#pragma GCC target("bmi2")
#include <immintrin.h>
#include <stdint.h>

#include <array>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>

template <typename T>
struct Vector {
    T x, y;

    Vector() : x(0), y(0) { ; }

    Vector(T x, T y) : x(x), y(y) { ; }

    // template <typename T2>
    // operator Vector<T2>() const {
    //     return Vector<T2>(x, y);
    // }

    Vector operator+(const Vector& other) const {
        return Vector(x + other.x, y + other.y);
    }

    Vector operator-() const {
        return Vector(-x, -y);
    }

    Vector operator-(const Vector& other) const {
        return Vector(x - other.x, y - other.y);
    }

    T operator*(const Vector& other) const {
        return x * other.x + y * other.y;
    }

    T operator%(const Vector& other) const {
        return x * other.y - y * other.x;
    }

    Vector<double> operator*(const double& val) const {
        return Vector(x * val, y * val);
    }

    bool operator==(const Vector& other) const {
        return x == other.x && y == other.y;
    }
};

struct Svg {
    std::stringstream sout;

    static constexpr double scale = 900;
    static constexpr double shift = 50;

    Svg() {
        clear();
    }

    void clear() {
        sout = std::stringstream();
        sout.precision(5);
        sout << std::fixed;
        sout << "<svg width=\"1000px\" height=\"1000px\" style=\"background-color:lightgreen\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    }

    void print() {
        std::string s = sout.str();
        s += "</svg>\n";

        std::ofstream fout("cum.svg");
        fout << s << "\n";
        fout.flush();
        fout.close();
    }

    void line(Vector<double> pt1, Vector<double> pt2, std::string color, double width = 1) {
        sout << "<line ";
        sout << "x1=\"" << (float)(pt1.x * scale + shift) << "\" ";
        sout << "y1=\"" << (float)((scale - pt1.y * scale) + shift) << "\" ";
        sout << "x2=\"" << (float)(pt2.x * scale + shift) << "\" ";
        sout << "y2=\"" << (float)((scale - pt2.y * scale) + shift) << "\" ";
        sout << " stroke=\"" << color << "\"";
        sout << " stroke-width=\"" << float(width) << "\"";
        sout << "/>\n";
    }

    void circle(Vector<double> pt, double r, std::string color, double width = 1) {
        sout << "<circle ";
        sout << "cx=\"" << float(pt.x * scale + shift) << "\" ";
        sout << "cy=\"" << float((scale - pt.y * scale) + shift) << "\" ";
        sout << "r=\"" << float(r) << "\" ";
        sout << " stroke=\"" << color << "\"";
        sout << " stroke-width=\"" << float(width) << "\"";
        sout << "/>\n";
    };
};

int32_t main() {
    constexpr int n = 64;

    std::vector<std::array<int, 3>> vec;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            auto hilbert = [](int x, int y) {
                constexpr int LG = 20;
                int64_t res = 0;
                int rt = 0;
                int flip = 1;
                for (int i = LG - 1; i >= 0; i--) {
                    int a = (x >> i) & 1;
                    int b = (y >> i) & 1;
                    int x = a == 0 ? b : 3 - b;
                    x = (x + rt) & 3;
                    int64_t cnt = 1LL << 2 * i;
                    res += (flip == 1 ? x : 3 - x) * cnt;
                    if (x == 0) {
                        rt = (rt + 3) & 3;
                        flip *= -1;
                    } else if (x == 3) {
                        rt = (rt + 1) & 3;
                        flip *= -1;
                    }
                }
                return res;
            };

            int ord = hilbert(i, j);

            // const uint64_t mask = 0x5555'5555'5555'5555;
            // int ord = _pdep_u64(i, mask << 1) | (_pdep_u64(j, mask));
            // int ord = _pdep_u64(i, 0x69'69'69'69) | (_pdep_u64(j, 0x96'96'96'96));

            vec.push_back({ord, i, j});
        }
    }

    Svg svg;
    std::sort(vec.begin(), vec.end());
    // constexpr int K = 8;
    // std::sort(vec.begin(), vec.end(), [&](auto a, auto b) {
    //     auto [o1, x1, y1] = a;
    //     auto [o2, x2, y2] = b;
    //     if (x1 / K % 2) {
    //         std::swap(y1, y2);
    //     }
    //     int p1 = y1 % 2 ? x1 : -x1;
    //     int p2 = y1 % 2 ? x2 : -x2;
    //     // p1 = p2 = 0;
    //     return std::array{x1 / K, y1, p1} < std::array{x2 / K, y2, p2};
    // });
    for (int i = 1; i < vec.size(); i++) {
        auto [o1, x1, y1] = vec[i - 1];
        auto [o2, x2, y2] = vec[i];
        svg.line({double(x1) / n, double(y1) / n}, {double(x2) / n, double(y2) / n}, "green", 2);
    }
    svg.print();
}