#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;

constexpr u32 mod = 998'244'353;
constexpr u32 pr_root = 3;

u32 mul(u32 a, u32 b) {
    return u64(a) * b % mod;
}
u32 add(u32 a, u32 b) {
    return a + b - mod * (a + b >= mod);
}
void add_to(u32& a, u32 b) {
    a = add(a, b);
}
u32 power(u32 b, u32 e) {
    u32 r = 1;
    for (; e; e >>= 1) {
        if (e & 1) {
            r = mul(r, b);
        }
        b = mul(b, b);
    }
    return r;
}

struct NTT {
    std::vector<u32> wd, wrd;

    NTT() {
        int lg = __builtin_ctz(mod - 1);
        wd.assign(lg, 0), wrd.assign(lg, 0);
        for (int i = 0; i < lg - 1; i++) {
            u32 a = power(pr_root, mod - 1 >> i + 2);
            u32 b = power(pr_root, (mod - 1 >> i + 1) * ((1 << i) - 1));
            u32 f = mul(a, power(b, mod - 2));
            wd[i] = f, wrd[i] = power(f, mod - 2);
        }
    }

    template <bool transposed>
    void butterfly_x2(u32& a, u32& b, u32 w) {
        if (!transposed) {
            u32 a1 = a, b1 = mul(b, w);
            a = add(a1, b1), b = add(a1, mod - b1);
        } else {
            u32 a2 = add(a, b), b2 = mul(add(a, mod - b), w);
            a = a2, b = b2;
        }
    }

    void transform_forward(int lg, u32* data) {
        for (int k = lg - 1; k >= 0; k--) {
            u32 wi = 1;
            for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                for (int j = 0; j < (1 << k); j++) {
                    butterfly_x2<false>(data[i + j], data[i + (1 << k) + j], wi);
                }
                wi = mul(wi, wd[__builtin_ctz(~i >> k + 1)]);
            }
        }
    }

    void transform_inverse(int lg, u32* data) {
        for (int k = 0; k < lg; k++) {
            u32 wi = 1;
            for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                for (int j = 0; j < (1 << k); j++) {
                    butterfly_x2<true>(data[i + j], data[i + (1 << k) + j], wi);
                }
                wi = mul(wi, wrd[__builtin_ctz(~i >> k + 1)]);
            }
        }
        u32 inv = power(mod + 1 >> 1, lg);
        for (int i = 0; i < (1 << lg); i++) {
            data[i] = mul(data[i], inv);
        }
    }

    void convolve_cyclic(int lg, u32* a, u32* b) {
        transform_forward(lg, a);
        transform_forward(lg, b);
        for (int i = 0; i < (1 << lg); i++) {
            a[i] = mul(a[i], b[i]);
        }
        transform_inverse(lg, a);
    }
} ntt;

namespace polynomial {
    using poly = std::vector<u32>;

    void remove_zeros(poly& p) {
        while (p.size() && p.back() == 0) {
            p.pop_back();
        }
    }

    poly poly_mul(poly a, poly b) {
        if (a.empty() || b.empty()) {
            return {};
        }
        int n = a.size(), m = b.size();
        if (n <= 2 || m <= 2) {
            std::vector<u32> c(n + m - 1);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    add_to(c[i + j], mul(a[i], b[j]));
                }
            }
            return c;
        }
        int lg = std::bit_width<size_t>(n + m - 2);
        assert((1 << lg) >= (n + m - 1));
        a.resize(1 << lg), b.resize(1 << lg);
        ntt.convolve_cyclic(lg, a.data(), b.data());
        a.resize(n + m - 1);
        remove_zeros(a);
        return a;
    }

    poly poly_add(poly a, poly b) {
        if (a.size() < b.size()) {
            a.swap(b);
        }
        for (int i = 0; i < b.size(); i++) {
            add_to(a[i], b[i]);
        }
        return a;
    }

    poly poly_neg(poly a) {
        for (int i = 0; i < a.size(); i++) {
            a[i] = add(0, mod - a[i]);
        }
        return a;
    }

    poly poly_sub(poly a, poly b) {
        return poly_add(a, poly_neg(b));
    }

    // mul by x^k
    poly mul_xk(poly a, int k) {
        a.insert(a.begin(), k, 0);
        return a;
    }

}  // namespace polynomial

using namespace polynomial;

template <int N, int M>
using poly_mat = std::array<std::array<poly, M>, N>;

template <int N>
using poly_vec = poly_mat<N, 1>;

template <int N, int M, int K>
poly_mat<N, K> poly_mat_mul(poly_mat<N, M> a, poly_mat<M, K> b) {
    poly_mat<N, K> c;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < M; k++) {
                c[i][j] = poly_add(c[i][j], poly_mul(a[i][k], b[k][j]));
            }
        }
    }
    return c;
}

std::vector<u32> fisting_ass(const std::vector<u32>& a) {
    int n = a.size();

    std::vector<u32> rec = {1};  // current relation
    std::vector<u32> cor;        // correction relation
    int cor_end = 0;

    for (int i = 0; i < n; i++) {
        u32 val = 0;
        for (int j = 0; j < rec.size(); j++) {
            add_to(val, mul(rec[j], a[i - j]));
        }
        if (val == 0) {
            continue;
        }
        if (rec.size() == 1) {
            cor_end = i, cor = {power(val, mod - 2)}, rec.resize(i + 2, 0);
            continue;
        }
        int d = i - cor_end;
        std::vector<u32> new_cor;
        if (rec.size() < d + cor.size()) {
            new_cor.resize(rec.size());
            u32 f = power(val, mod - 2);
            for (int j = 0; j < new_cor.size(); j++) {
                new_cor[j] = mul(rec[j], f);
            }
            rec.resize(d + cor.size());
        }
        for (int j = 0; j < cor.size(); j++) {
            add_to(rec[d + j], mul(mod - val, cor[j]));
        }
        if (new_cor.size()) {
            cor_end = i, cor = new_cor;
        }
    }
    return rec;
}

std::vector<u32> fucking_cumming(const std::vector<u32>& a) {
    int n = a.size();
    int it = std::find_if(a.begin(), a.end(), [&](u32 x) { return x != 0; }) - a.begin();
    if (it == a.size()) {
        return {1};
    }

    std::vector<u32> r(it + 2, 0);
    std::vector<u32> c = {0, 1};
    r[0] = 1;

    if (it + 1 == a.size()) {
        return r;
    }

    int r_size = it + 2, c_size = 2;

    int iter = 0;

    auto mat = [&](this auto fisting, poly_vec<2> rc_eval) -> poly_mat<2, 2> {
        int n = rc_eval[0][0].size();
        if (n == 1) {
            iter++;
            std::cerr << iter << " " << r_size << std::endl;

            u32 val = rc_eval[0][0].front();
            if (val == 0) {
                c_size += 1;
                return poly_mat<2, 2>{{{{{1}, {}}}, {{{}, {0, 1}}}}};
            } else {
                u32 f = mul(mod - val, power(rc_eval[1][0].front(), mod - 2));
                if (c_size <= r_size) {
                    c_size += 1;
                    return poly_mat<2, 2>{{{{{1}, {f}}}, {{{}, {0, 1}}}}};
                } else {
                    // c_size = r_size;
                    std::swap(c_size, r_size);
                    c_size += 1;
                    return poly_mat<2, 2>{{{{{1}, {f}}}, {{{0, 1}, {}}}}};
                }
            }
        }

        int m = n / 2;

        auto r = rc_eval[0][0], c = rc_eval[1][0];
        auto mt1 = fisting({{{poly(r.begin(), r.begin() + m)}, {poly(c.begin(), c.begin() + m)}}});
        rc_eval = poly_mat_mul<2, 2, 1>(mt1, rc_eval);

        r = rc_eval[0][0], c = rc_eval[1][0];
        r.resize(n), c.resize(n);
        auto mt2 = fisting({{{poly(r.begin() + m, r.begin() + n)}, {poly(c.begin() + m, c.begin() + n)}}});

        return poly_mat_mul<2, 2, 2>(mt2, mt1);
    }(poly_vec<2>{{{std::vector<u32>(a.begin() + it + 1, a.end())}, {std::vector<u32>(a.begin() + it, a.begin() + (n - 1))}}});

    poly res = poly_mat_mul<2, 2, 1>(mat, poly_mat<2, 1>{{{r}, {c}}})[0][0];
    res.resize(r_size);
    return res;
}

// void test() {
//     poly_mat<2, 2> mt1;
//     poly_mat<2, 2> mt2;
//     poly_mat<2, 2> mt3 = poly_mat_mul(mt1, mt2);
// }

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n;
    std::cin >> n;
    std::vector<u32> input(n);
    for (auto& i : input) {
        std::cin >> i;
    }

    std::vector<u32> rec = fucking_cumming(input);
    std::cout << rec.size() - 1 << "\n";
    for (int i = 1; i < rec.size(); i++) {
        std::cout << add(0, mod - rec[i]) << " \n"[i + 1 == rec.size()];
    }

    return 0;
}
