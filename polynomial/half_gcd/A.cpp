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
    // using poly = std::vector<u32>;

    using poly_base = std::vector<u32>;
    struct poly : poly_base {
        using poly_base::poly_base;

        void remove_zeros();

        u32 coeff(size_t ind) const {
            if (ind < size()) {
                return operator[](ind);
            }
            return 0;
        }
    };

    void poly_remove_zeros(poly& p) {
        while (p.size() && p.back() == 0) {
            p.pop_back();
        }
    }

    poly poly_mul(poly a, poly b) {
        a.remove_zeros(), b.remove_zeros();
        if (a.empty() || b.empty()) {
            return {};
        }
        int n = a.size(), m = b.size();
        int lg = std::bit_width<size_t>(n + m - 2);
        if (std::min(n, m) <= std::max(2 * lg, 2)) {
            poly c(n + m - 1);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    add_to(c[i + j], mul(a[i], b[j]));
                }
            }
            return c;
        }
        assert((1 << lg) >= (n + m - 1));
        a.resize(1 << lg), b.resize(1 << lg);
        ntt.convolve_cyclic(lg, a.data(), b.data());
        a.remove_zeros();
        return a;
    }

    poly poly_add(poly a, poly b) {
        if (a.size() < b.size()) {
            a.swap(b);
        }
        for (int i = 0; i < b.size(); i++) {
            add_to(a[i], b[i]);
        }
        a.remove_zeros();
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

    poly inv(poly a, int n) {
        a.resize(n);
        assert(a.size() >= 1 && a[0] != 0);
        poly res = {{power(a[0], mod - 2)}};
        for (int k = 0; (1 << k) < n; k++) {
            poly b(a.begin(), a.begin() + std::min<size_t>(a.size(), 1 << k + 1));
            res = poly_mul(res, poly_add({{2}}, poly_neg(poly_mul(res, b))));
            res.resize(1 << k + 1);
        }
        res.resize(n);
        {
            // std::vector<u32> c = poly_mul(a, res);
            // c.resize(n);
            // c[0] -= 1;
            // assert(c == std::vector<u32>(n, 0));
        }
        return res;
    }

    std::pair<poly, poly> poly_divmod(poly a, poly b) {
        int n = a.size(), m = b.size();
        if (n < m) {
            return {{}, a};
        }
        int d = n - m + 1;
        poly A = a, B = b;
        std::reverse(A.begin(), A.end()), std::reverse(B.begin(), B.end());
        A.resize(d);
        poly q = poly_mul(A, inv(B, d));
        q.resize(d);
        std::reverse(q.begin(), q.end());
        poly r = poly_add(a, poly_neg(poly_mul(b, q)));
        r.remove_zeros();
        assert(r.size() <= m);
        r.resize(m);
        // assert(a == poly_add(poly_mul(b, q), r));
        return {q, r};
    }

    void poly::remove_zeros() { poly_remove_zeros(*this); }

    poly operator+(const poly& a, const poly& b) { return poly_add(a, b); }
    poly operator-(const poly& a) { return poly_neg(a); }
    poly operator-(const poly& a, const poly& b) { return a + (-b); }
    poly operator*(const poly& a, const poly& b) { return poly_mul(a, b); }
    poly operator/(const poly& a, const poly& b) { return poly_divmod(a, b).first; }
    poly operator%(const poly& a, const poly& b) { return poly_divmod(a, b).second; }

    poly& operator+=(poly& a, const poly& b) { return a = (a + b); }
    poly& operator-=(poly& a, const poly& b) { return a = (a - b); }
    poly& operator*=(poly& a, const poly& b) { return a = (a * b); }

}  // namespace polynomial

using polynomial::poly;
using polynomial::poly_divmod;

template <size_t N, size_t M>
using poly_mat = std::array<std::array<poly, M>, N>;

template <size_t N, size_t M, size_t K>
poly_mat<N, K> poly_mat_mul(poly_mat<N, M> a, poly_mat<M, K> b) {
    poly_mat<N, K> c;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < M; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

void mul_by_matrix(const poly_mat<2, 2>& mat, poly& a, poly& b) {
    auto vec = poly_mat_mul(mat, poly_mat<2, 1>{{{{a}}, {{b}}}});
    a = vec[0][0], b = vec[1][0];
}

std::array<poly, 3> naive_ext_gcd(poly a, poly b) {
    std::vector<poly> cum;
    while (true) {
        b.remove_zeros();
        if (b.empty()) {
            poly gcd = a;
            poly u = {1};
            poly v = {};
            for (int i = 0; i < cum.size(); i++) {
                const poly& q = cum.rbegin()[i];
                poly u2 = v, v2 = u - v * q;
                u = u2, v = v2;
            }
            return {gcd, u, v};
        }
        auto [q, r] = poly_divmod(a, b);
        q.shrink_to_fit();
        cum.push_back(q);
        a.swap(b), b = r;
    }
}

std::array<poly, 3> slow_ext_gcd(poly a, poly b) {
    int n = std::max(a.size(), b.size());
    std::vector<std::pair<size_t, u32>> cum;
    while (true) {
        a.remove_zeros(), b.remove_zeros();
        if (b.empty()) {
            break;
        }
        size_t sh = 0;
        u32 f = 0;
        if (a.size() >= b.size()) {
            sh = a.size() - b.size();
            f = mul(mod - a.back(), power(b.back(), mod - 2));
            for (size_t i = 0; i < b.size(); i++) {
                add_to(a[i + sh], mul(b[i], f));
            }
        }
        cum.push_back({sh, f});
        a.swap(b);
    }
    poly u = {1}, v = {}, u2, v2;
    u.reserve(n), v.reserve(n), u2.reserve(n), v2.reserve(n);

    if (1) {
        for (size_t i = 0; i < cum.size(); i++) {
            auto [sh, f] = cum.rbegin()[i];
            if (f != 0) {
                u2.assign(v.begin(), v.end());
                v2.assign(u.begin(), u.end());
                v2.resize(std::max(v2.size(), v.size() + sh));
                for (size_t i = 0; i < v.size(); i++) {
                    add_to(v2[i + sh], mul(v[i], f));
                }
                u.swap(u2), v.swap(v2);
            } else {
                std::swap(u, v);
            }
        }
    } else {
        auto mt = [&](this auto fisting, size_t l, size_t r) -> poly_mat<2, 2> {
            if (l == r) {
                return {{{{{1}, {}}}, {{{}, {1}}}}};
            }
            if (l + 1 == r) {
                auto [sh, f] = cum.rbegin()[l];
                return {{{{{}, {1}}}, {{{1}, polynomial::mul_xk(poly{f}, sh)}}}};
            } else {
                int m = (l + r) / 2;
                return poly_mat_mul(fisting(m, r), fisting(l, m));
            }
        }(0, cum.size());
        mul_by_matrix(mt, u, v);
    }
    return {a, u, v};
}

poly_mat<2, 2> fisting(size_t d, poly a, poly b, int dp = 0) {
    assert(std::has_single_bit(d));
    a.remove_zeros(), b.remove_zeros();
    assert(a.size() <= d && b.size() <= d);

    poly_mat<2, 2> res = {{{{{1}, {}}}, {{{}, {1}}}}};
    if (a.size() < b.size()) {
        a.swap(b);
        std::swap(res[0], res[1]);
    }
    if (b.size() <= d / 2) {
        return res;
    }
    if (d == 1) {
        u32 f = mul(b[0], power(a[0], mod - 2));
        res = poly_mat_mul(poly_mat<2, 2>{{{{{1}, {}}}, {{{add(0, mod - f)}, {1}}}}}, res);
        return res;
    }
    poly a1(d / 2), b1(d / 2);
    size_t s = a.size() - d / 2;
    for (size_t i = 0; i < d / 2; i++) {
        a1[i] = a.coeff(s + i);
        b1[i] = b.coeff(s + i);
    }
    a1.remove_zeros(), b1.remove_zeros();

    // if (dp >= 2) {
    //     std::cerr << dp << "\n";
    // }

    poly_mat<2, 2> mt1;
    mt1 = fisting(d / 2, a1, b1);
    mul_by_matrix(mt1, a, b);

    poly q = a / b;
    a -= q * b;
    mt1 = poly_mat_mul(poly_mat<2, 2>{{{{{1}, {-q}}}, {{{}, {1}}}}}, mt1);

    poly_mat<2, 2> mt2 = fisting(d, a, b, dp + 1);
    return poly_mat_mul(mt2, poly_mat_mul(mt1, res));
}

std::array<poly, 3> ext_gcd(poly a, poly b) {
    std::vector<poly_mat<2, 2>> cum;
    size_t d = std::bit_ceil(std::max(a.size(), b.size()));
    for (; d >= 1 && b.size(); d /= 2) {
        poly_mat<2, 2> mt = fisting(d, a, b);
        mul_by_matrix(mt, a, b);
        b.remove_zeros();

        cum.push_back(mt);

        if (!b.empty()) {
            poly q = a / b;
            a -= b * q;
            a.swap(b);
            cum.push_back(poly_mat<2, 2>{{{{{}, {1}}}, {{{1}, {-q}}}}});
            // mt = poly_mat_mul(poly_mat<2, 2>{{{{{}, {1}}}, {{{1}, {-q}}}}}, mt);
        }
    }

    poly_mat<2, 2> mt = {{{{{1}, {}}}, {{{}, {1}}}}};
    for (size_t i = 0; i < cum.size(); i++) {
        mt = poly_mat_mul(mt, cum.rbegin()[i]);
    }
    a.remove_zeros(), b.remove_zeros();
    assert(b.empty());
    return {a, mt[0][0], mt[0][1]};
}

std::optional<poly> mod_inv(poly f, poly g) {
    int n = f.size(), m = g.size();
    assert(n != 0 && m != 0);
    assert(f.back() != 0 && g.back() != 0);
    if (m == 1) {
        return poly{};
    }
    auto [gcd, u, v] = ext_gcd(f, g);  // u * f + v * g = g
    assert(gcd.size() > 0);
    if (gcd.size() > 1) {
        return std::nullopt;
    }
    u32 s = power(gcd[0], mod - 2);
    poly res = u * poly({s}) % g;

    poly cum = (res * f - poly{1}) % g;
    cum.remove_zeros(), assert(cum.empty());

    return res;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    std::cin >> n >> m;
    poly f(n), g(m);
    for (auto& i : f) {
        std::cin >> i;
    }
    for (auto& i : g) {
        std::cin >> i;
    }

    auto res = mod_inv(f, g);
    if (res.has_value()) {
        poly vec = res.value();
        vec.remove_zeros();
        std::cout << vec.size() << "\n";
        for (int i = 0; i < vec.size(); i++) {
            std::cout << vec[i] << " \n"[i + 1 == vec.size()];
        }
    } else {
        std::cout << "-1\n";
    }

    return 0;
}
