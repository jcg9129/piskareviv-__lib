#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>

#pragma GCC target("avx2,fma")
#pragma GCC optimize("O3,fast-math,unroll-loops")

using cum = double;
using fuck = uint64_t __attribute__((may_alias, aligned(1)));

#define double float

constexpr int n = 256;

cum cum_read(void* ptr, uint64_t dlt) {
    uint64_t val;
    val = *((fuck*)(ptr + dlt));
    cum res;
    memcpy(&res, &val, 8);
    return res;
}

cum multiply_matrices(cum a_[n][n], cum b_[n][n]) {
    cum res_0 = 0;
    float res = 0;
    {
        double a[n][n], b[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = a_[i][j];
                b[i][j] = b_[i][j];
            }
        }
        double c[n][n];
        memset(c, 0, sizeof(c));

        for (int i1 = 0; i1 < n; i1 += 32) {
            for (int k1 = 0; k1 < n; k1 += 32) {
                for (int k0 = k1; k0 < k1 + 32; k0 += 8) {
                    for (int i0 = i1; i0 < i1 + 32; i0 += 8) {
                        for (int j = 0; j < n; j++) {
                            for (int i = i0; i < i0 + 8; i++) {
                                for (int k = k0; k < k0 + 8; k++) {
                                    c[i][j] += a[i][k] * b[k][j];
                                }
                            }
                        }
                    }
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                res += c[i][j] * c[i][j];
            }
        }
    }
    for (int i = 32;; i++) {
        cum res2 = cum_read(&res_0, i);
        if (std::abs(res - res2) <= std::max<double>(1e-5, std::abs(res) / 1e5)) {
            std::cerr << i << " ";
            return res2;
        }
    }
    return res;
}

#undef double

double multiply_matrices_naive(double a[n][n], double b[n][n]) {
    double c[n][n];
    memset(c, 0, sizeof(c));

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    double res = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res += c[i][j] * c[i][j];
        }
    }
    return res;
}

#ifdef LOCAL

int32_t main() {
    std::mt19937 rnd;
    double a[n][n], b[n][n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = rnd() % 100;
            b[i][j] = rnd() % 100;
        }
    }

    double c[n][n], d[n][n];

    std::function<double(std::function<double(double[n][n], double[n][n])>)> run = [&](auto f) {
        memcpy(c, a, sizeof(a));
        memcpy(d, b, sizeof(b));
        return f(c, d);
    };

    double res1 = run(multiply_matrices_naive);
    // res1 *= 10;
    double res2 = run(multiply_matrices);
    std::cout << res1 << " " << res2 << " " << res1 - res2 << "\n";

    clock_t beg = 0;
    double res = 0;
    for (int i = 0; i < 1024; i++) {
        res += run(multiply_matrices);
    }
    if (res == 0) {
        std::cout << res << " cum\n";
    }
    double tm = (clock() - beg) * 1.0 / CLOCKS_PER_SEC;

    std::cout.precision(1);
    std::cout << std::fixed;
    std::cout << tm * 1000 << " ms" << std::endl;

    // res1 /= 10;
    assert(std::abs(res1 - res2) <= 1e-6);
}

#endif
