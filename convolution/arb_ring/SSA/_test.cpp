#include <bits/allocator.h>
#pragma GCC target("pclmul")
#include <iostream>

#include "conv.hpp"
#include "field_64.hpp"

void stress_test() {
    using namespace Field_64;

    std::mt19937_64 rnd;

    for (int k = 0; k < 100; k++) {
        for (int n = 0; n <= k; n++) {
            int m = k - n;

            constexpr int ITERS = 0;
            for (int iter = 0; iter <= ITERS; iter++) {
                std::vector<Field> a(n), b(m);
                if (iter == ITERS) {
                    for (auto& i : a) i = Field(rnd());
                    for (auto& i : b) i = Field(rnd());
                } else {
                    if (n == 0 || m == 0) {
                        continue;
                    }
                    a[rnd() % a.size()] = Field(1);
                    b[rnd() % b.size()] = Field(1);
                }
                std::vector<Field> c1 = conv::convolve_naive(a, b);
                std::vector<Field> c2 = conv::convolve(a, b);
                // auto c1 = c2;
                // auto c2 = c1;

                if (c1 != c2) {
                    std::cerr << n << " " << m << "\n";
                    for (int i = 0; i < c1.size(); i++) {
                        std::cerr << c1[i].get() << " \n"[i + 1 == c1.size()];
                    }
                    for (int i = 0; i < c2.size(); i++) {
                        std::cerr << c2[i].get() << " \n"[i + 1 == c2.size()];
                    }
                }
                assert(c1 == c2);
            }
        }
    }
}

void bench() {
    using R = Field_64::Field;
    std::mt19937_64 rnd;

    int n = 5e5;
    clock_t beg = clock();

    std::vector<R> a(n), b(n);
    for (int i = 0; i < n; i++) {
        a[i] = R(rnd());
        b[i] = R(rnd());
    }
    conv::convolve(a, b);
    double tm = (clock() - beg) * 1.0 / CLOCKS_PER_SEC;

    std::cerr.precision(2);
    std::cerr << std::fixed;
    std::cerr << tm * 1000 << "ms" << "\n";
}

int32_t main() {
    stress_test();
    bench();
}
