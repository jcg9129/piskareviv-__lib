#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>

#pragma GCC target("avx2,fma")
#pragma GCC optimize("O3,fast-math,unroll-loops")

// #define double float

constexpr int n = 256;

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

double multiply_matrices(double a[n][n], double b[n][n]) {
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
    double a[n][n], b[n][n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = 1 / (i + 2 * j + 1);
            b[i][j] = 3 * i + j;
        }
    }

    double c[n][n], d[n][n];

    std::function<double(std::function<double(double[n][n], double[n][n])>)> run = [&](auto f) {
        memcpy(c, a, sizeof(a));
        memcpy(d, b, sizeof(b));
        return f(c, d);
    };

    double res1 = run(multiply_matrices_naive);
    double res2 = run(multiply_matrices);
    // std::cout << res1 << " " << res2 << " " << res1 - res2 << "\n";

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

    assert(std::abs(res1 - res2) <= 1e-6);
}

#endif