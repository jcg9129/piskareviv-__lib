#include <iostream>
// #pragma GCC push_options
#pragma GCC optimize("O3,unroll-loops,fast-math")
#pragma GCC target("avx2,fma,bmi2")
// #include <bits/stdc++.h>
#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <thread>

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

using f64x4 = __m256d;

template <int X, int Z>
// [[gnu::noinline]]
void kernel(int i0, int k0, double a[n][n], double b[n][n], double c[n][n]) {
    // for (int j = 0; j < n; j++) {
    //     for (int i = i0; i < i0 + X; i++) {
    //         for (int k = k0; k < k0 + Z; k++) {
    //             c[i][j] += a[i][k] * b[k][j];
    //         }
    //     }
    // }
    // return;

    using f64x4 = __m256d;
    alignas(64) f64x4 ai[X][Z];
    for (int x = 0; x < X; x++) {
        for (int z = 0; z < Z; z++) {
            ai[x][z] = _mm256_set1_pd(a[i0 + x][k0 + z]);
        }
    }
    for (int j = 0; j < n; j += 4) {
        alignas(64) f64x4 ci[X];
        alignas(64) f64x4 bi[Z];
        for (int z = 0; z < Z; z++) {
            bi[z] = _mm256_load_pd(&b[k0 + z][j]);
        }
        for (int x = 0; x < X; x++) {
            ci[x] = _mm256_load_pd(&c[i0 + x][j]);
        }
        for (int z = 0; z < Z; z++) {
            for (int x = 0; x < X; x++) {
                ci[x] = _mm256_fmadd_pd(ai[x][z], bi[z], ci[x]);
            }
        }
        for (int x = 0; x < X; x++) {
            _mm256_store_pd(&c[i0 + x][j], ci[x]);
        }
    }
}

double multiply_matrices(double a[n][n], double b[n][n]) {
    alignas(64) double c[n][n];
    memset(c, 0, sizeof(c));

    constexpr int X = 2;
    constexpr int Z = 4;

    // for (int i = 0; i + X <= n; i += X) {
    //     for (int k = 0; k + Z <= n; k += Z) {
    //         kernel<X, Z>(i, k, a, b, c);
    //     }
    // }

    constexpr int Oi = 32;
    constexpr int Ok = 16;

    for (int i1 = 0; i1 < n; i1 += Oi) {
        for (int k1 = 0; k1 < n; k1 += Ok) {
            for (int i = i1; i < i1 + Oi; i += X) {
                for (int k = k1; k < k1 + Ok; k += Z) {
                    kernel<X, Z>(i, k, a, b, c);
                    // kernel<X, Z>(i - i1, k - k1, a, b, c);
                }
            }
        }
    }

    {
        constexpr int G = 8;
        f64x4 sum[G];
        memset(sum, 0, sizeof(sum));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j += 4 * G) {
                for (int k = 0; k < G; k++) {
                    f64x4 vec = _mm256_load_pd(&c[i][j + 4 * k]);
                    sum[k] = _mm256_fmadd_pd(vec, vec, sum[k]);
                }
            }
        }
        for (int i = G - 1; i > 0; i--) {
            sum[i / 2] += sum[i];
        }
        double res = sum[0][0] + sum[0][1] + sum[0][2] + sum[0][3];
        return res;
    }
}

#ifdef LOCAL

#ifndef NO_CUM

int32_t main() {
    alignas(64) double a[n][n], b[n][n];
    std::mt19937 rnd;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = rnd() % 10;
            b[i][j] = rnd() % 10;
        }
    }

    double c[n][n], d[n][n];

    memcpy(c, a, sizeof(a));
    memcpy(d, b, sizeof(b));

    std::function<double(std::function<double(double[n][n], double[n][n])>)> run = [&](auto f) {
        // memcpy(c, a, sizeof(a));
        // memcpy(d, b, sizeof(b));
        return f(c, d);
    };

    double res1 = run(multiply_matrices_naive);
    double res2 = run(multiply_matrices);
    // std::cout << res1 << " " << res2 << " " << res1 - res2 << "\n";

    clock_t beg = 0;
    double res = 0;
    const int64_t ITERS = 1024 * pow(256.0 / n, 3);
    for (int i = 0; i < ITERS; i++) {
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

#else

int32_t main() {
    alignas(64) double a[n][n], b[n][n], c[n][n];
    std::mt19937 rnd;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = rnd() % 10;
            b[i][j] = rnd() % 10;
        }
    }

    constexpr int X = 2;
    constexpr int Z = 4;

    clock_t beg = 0;
    constexpr int I = 32, K = 16;
    const int64_t ITERS = 1024 * pow(256, 2) / (I * n);
    for (int it = 0; it < ITERS; it++) {
        for (int k0 = 0; k0 < n; k0 += K) {
            for (int i = 0; i < I; i += X) {
                for (int k = k0; k < k0 + K; k += Z) {
                    kernel<X, Z>(i, k, a, b, c);
                }
            }
        }
        // kernel<X, Z>(0, 0, a, b, c);
    }
    double tm = (clock() - beg) * 1.0 / CLOCKS_PER_SEC;
    std::cout.precision(1);
    std::cout << std::fixed;
    std::cout << tm * 1000 << " ms" << std::endl;
}

#endif

#endif
