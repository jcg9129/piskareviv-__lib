

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;

struct Montgomery {
    u32 mod;    //
    u32 mod2;   // mod * 2
    u32 n_inv;  // n_inv * mod == -1 (mod 2^32)
    u32 r;      // 2^32 % mod
    u32 r2;     // r^2 % mod;

    Montgomery() = default;
    Montgomery(u32 mod) {
        assert(mod % 2);
        assert(mod < (1 << 30));
        this->mod = mod;
        mod2 = 2 * mod;
        n_inv = 1;
        for (int i = 0; i < 5; i++) {
            n_inv *= 2 + n_inv * mod;
        }
        assert(n_inv * mod == u32(-1));
        r = (u64(1) << 32) % mod;
        r2 = u64(r) * r % mod;
    }

    u32 shrink(u32 val) const {
        return std::min(val, val - mod);
    }

    // result * 2^32 == val
    template <bool strict = true>
    u32 reduce(u64 val) const {
        u32 res = val + u32(val) * n_inv * u64(mod) >> 32;
        if constexpr (strict) {
            res = shrink(res);
        }
        return res;
    }

    // result * 2^32 == a * b
    template <bool strict = true>
    u32 mul(u32 a, u32 b) const {
        return reduce<strict>(u64(a) * b);
    }

    template <bool b_in_space = false, bool result_in_space = false>
    u32 power(u32 b, uint32_t e) const {
        if (!b_in_space) {
            b = mul<false>(b, r2);
        }
        u32 res = result_in_space ? r : 1;
        for (; e > 0; e >>= 1) {
            if (e & 1)
                res = mul<false>(res, b);
            b = mul<false>(b, b);
        }
        res = shrink(res);
        return res;
    }
};

struct NTT {
    // static    constexpr int LG = 32;
    u32 mod, pr_root;
    Montgomery mt;

    // mod must be prime
    u32 find_pr_root(u32 mod) const {
        u32 m = mod - 1;
        std::vector<u32> vec;
        for (u32 i = 2; u64(i) * i <= m; i++) {
            if (m % i == 0) {
                vec.push_back(i);
                do {
                    m /= i;
                } while (m % i == 0);
            }
        }
        if (m != 1) {
            vec.push_back(m);
        }
        for (u32 i = 2; i < mod; i++) {
            if (std::all_of(vec.begin(), vec.end(), [&](u32 f) { return 1 != mt.power<false, false>(i, (mod - 1) / f); })) {
                return i;
            }
        }
        assert(false);
    }

    std::vector<u32> w, wr;

    NTT() = default;
    NTT(u32 mod, u32 pr_root_0 = 0) : mod(mod), pr_root(pr_root_0), mt(mod) {
        sizeof(NTT);
        if (pr_root == 0) {
            pr_root = find_pr_root(mod);
        }
    }

    void expand(int lg) {
        while (w.size() < (1 << lg)) {
            if (w.size() == 0) {
                w = wr = {mt.r};
                continue;
            }
            int k = std::__lg(w.size());
            w.resize(1 << k + 1);
            wr.resize(1 << k + 1);
            u32 f = mt.power<false, true>(pr_root, mod - 1 >> k + 2), fr = mt.power<true, true>(f, mod - 2);
            for (int i = 0; i < (1 << k); i++) {
                w[i + (1 << k)] = mt.mul<true>(w[i], f);
                wr[i + (1 << k)] = mt.mul<true>(wr[i], fr);
            }
        }
    }

    template <bool transposed>
    void butterfly_x2(u32& a, u32& b, u32 w) {
        if (!transposed) {
            u32 a0 = a, b0 = b, c = mt.mul<true>(b, w);
            a = mt.shrink(a0 + c);
            b = mt.shrink(a0 + mt.mod - c);
        } else {
            u32 a0 = a, b0 = b;
            a = mt.shrink(a0 + b0);
            b = mt.mul<true>(a0 + mt.mod - b0, w);
        }
    }

    int bit_rev(int lg, int val) {
        int res = 0;
        for (int i = 0; i < lg; i++) {
            res |= (val >> lg - i - 1 & 1) << i;
        }
        return res;
    }

    void transform_1(int lg, u32* data) {
        // for (int k = lg - 1; k >= 0; k--) {
        //     for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
        //         for (int j = 0; j < (1 << k); j++) {
        //             butterfly_x2<false>(data[i + j], data[i + (1 << k) + j], w[i >> k + 1]);
        //         }
        //     }
        // }
        for (int k = lg - 1; k >= 0; k--) {
            for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                for (int j = 0; j < (1 << k); j++) {
                    butterfly_x2<true>(data[i + j], data[i + (1 << k) + j], w[bit_rev(k, j)]);
                }
            }
        }
    }

    void transform_2(int lg, u32* data) {
        for (int k = 0; k < lg; k++) {
            for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                for (int j = 0; j < (1 << k); j++) {
                    butterfly_x2<true>(data[i + j], data[i + (1 << k) + j], wr[i >> k + 1]);
                }
            }
        }
    }

    void print_fuck(int lg, auto f) {
        expand(lg - 1);
        std::vector<u32> vec;
        for (int i = 0; i < (1 << lg); i++) {
            vec.assign(1 << lg, 0);
            vec[i] = 1;
            f(lg, vec.data());
            for (int i = 0; i < (1 << lg); i++) {
                std::string s;
                int a = vec[i];
                if (a == mod - 1) {
                    s = "-1";
                } else {
                    while (a) {
                        s.push_back('0' + a % 10);
                        a /= 10;
                    }
                    if (s.size() == 0) {
                        s = "0";
                    }
                    std::reverse(s.begin(), s.end());
                }

                s.resize(12, ' ');
                std::cout << s << "\t\n"[i == (1 << lg) - 1];
            }
        }
        std::cout << "\n";
    }

    std::vector<u32> inv_fps(std::vector<u32> vec) {
        assert(vec.size() && vec[0] != 0);
        for (int i = 0; i < vec.size(); i++) {
            vec[i] = mt.mul<true>(vec[i], mt.r2);
        }
        int k = 0;
        std::vector<u32> inv = {mt.power<true, true>(vec[0], mod - 2)};
        std::vector<u32> tmp1, tmp2, tmp3;
        while ((1 << k) < vec.size()) {
            int n = 1 << k;
            vec.resize(std::max<int>(vec.size(), 2 * n));

            expand(k);

            tmp1.assign(2 * n, 0);
            std::copy(inv.begin(), inv.begin() + n, tmp1.begin());
            transform_1(k + 1, tmp1.data());

            tmp2.assign(2 * n, 0);
            std::copy(vec.begin(), vec.begin() + 2 * n, tmp2.begin());
            transform_1(k + 1, tmp2.data());

            const u32 fix = mt.power<false, true>(mod + 1 >> 1, k + 1);
            for (int i = 0; i < 2 * n; i++) {
                tmp2[i] = mt.mul<true>(fix, mt.mul<false>(tmp1[i], tmp2[i]));
            }
            transform_2(k + 1, tmp2.data());
            for (int i = 0; i < n; i++) {
                tmp2[i] = 0;
            }

            transform_1(k + 1, tmp2.data());

            for (int i = 0; i < 2 * n; i++) {
                tmp1[i] = mt.mul<true>(fix, mt.mul<true>(tmp1[i], mt.shrink(mt.r + mod - tmp2[i])));
            }

            transform_2(k + 1, tmp1.data());
            inv.resize(2 * n);
            std::copy(tmp1.begin() + n, tmp1.begin() + 2 * n, inv.begin() + n);

            k++;
        }

        inv.resize(vec.size());
        for (int i = 0; i < inv.size(); i++) {
            inv[i] = mt.mul<true>(inv[i], 1);
        }
        return inv;
    }
};

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    NTT ntt(998'244'353);

    // ntt.print_fuck(3, [&](auto... args...) { ntt.transform_1(args...); });
    // ntt.print_fuck(3, [&](auto... args...) { ntt.transform_4(args...); });
    // // ntt.print_fuck(3, [&](auto... args...) { ntt.transform_2(args...); });
    // return 0;

    int n;
    std::cin >> n;
    std::vector<u32> input(n);
    for (auto& i : input) {
        std::cin >> i;
    }

    clock_t beg = clock();
    auto inv = ntt.inv_fps(input);
    std::cerr << "work: " << double(clock() - beg) / CLOCKS_PER_SEC * 1000 << "ms\n";

    for (int i = 0; i < n; i++) {
        std::cout << inv[i] << " \n"[i == n - 1];
    }

    return 0;
}
