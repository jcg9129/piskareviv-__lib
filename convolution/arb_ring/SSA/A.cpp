#include <bits/allocator.h>
#pragma GCC target("pclmul")
#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <type_traits>
#include <vector>

// #pragma once

namespace Field_64 {

using u64 = uint64_t;
using u128 = __uint128_t;

__m128i clmul_vec(u64 a, u64 b) {
    __m128i tmp = _mm_clmulepi64_si128(_mm_cvtsi64_si128(a), _mm_cvtsi64_si128(b), 0);
    return tmp;
}

u128 clmul(u64 a, u64 b) {
    __m128i tmp = clmul_vec(a, b);
    u128 res;
    memcpy(&res, &tmp, 16);
    return res;
}

constexpr u128 clmul_constexpr(u64 a, u64 b) {
    u128 res = 0;
    for (int i = 0; i < 64; i++) {
        if (a >> i & 1) {
            res ^= u128(b) << i;
        }
    }
    return res;
}

int lg_u128(u128 val) {
    u64 a = val, b = val >> 64;
    return b ? 64 + (63 - __builtin_clzll(b)) : (a ? 63 - __builtin_clzll(a) : -1);
}

u128 take_mod(u128 val, u128 mod) {
    int lg = lg_u128(mod);
    for (int i = lg_u128(val); i >= lg; i = lg_u128(val)) {
        val ^= mod << i - lg;
    }
    return val;
}

u128 pow_mod(u128 b, u128 exp, u128 mod) {
    assert(lg_u128(mod) <= 64);
    u128 r = 1;
    for (; exp; exp >>= 1) {
        if (exp & 1) {
            r = take_mod(clmul(r, b), mod);
        }
        b = take_mod(clmul(b, b), mod);
    }
    return r;
}

u128 poly_gcd(u128 a, u128 b) {
    while (b) {
        a = take_mod(a, b), std::swap(a, b);
    }
    return a;
}

bool is_irreducible_naive(u128 mod) {
    int lg = lg_u128(mod);
    for (u128 i = 2; i < (u128(1) << lg / 2 + 1) && i < mod; i++) {
        if (take_mod(mod, i) == 0) {
            return false;
        }
    }
    return true;
}

bool is_irreducible(u128 mod) {
    const int n = lg_u128(mod);
    auto get = [&](int k) {
        return take_mod(pow_mod(2, u128(1) << k, mod) ^ 2, mod);
    };
    bool result = true;
    if (poly_gcd(get(n), mod) != mod) {
        result = false;
    } else {
        for (int i = 1; i < n; i++) {
            if (n % i == 0 && poly_gcd(get(i), mod) != 1) {
                result = false;
                break;
            }
        }
    }
    // if (result != is_irreducible_naive(mod)) {
    //     std::cerr << result << " " << (u64)mod << "\n";
    // }
    // assert(result == is_irreducible_naive(mod));
    return result;
}

// std::mt19937_64 rnd_64;

u128 find_irreducible_poly(int deg) {
    // while (true) {
    //     u128 mod = rnd_64() | u128(rnd_64()) << 64;
    //     mod >>= 127 - deg;
    //     mod |= u128(1) << deg;
    //     if (is_irreducible(mod)) {
    //         return mod;
    //     }
    // }
    for (u128 mod = u128(1) << deg; (mod >> deg + 1) == 0; mod++) {
        if (is_irreducible(mod)) {
            return mod;
        }
    }
    assert(false);
}

struct Field {
   public:
    static constexpr u128 mod = 0b11011 | u128(1) << 64;

   private:
    static constexpr u64 inv = [] {
        auto clmul = clmul_constexpr;
        u64 a = 1;
        for (int i = 0; i < 6; i++) {
            a = clmul(a, clmul(a, (u64)mod));
        }
        return a;
    }();
    static constexpr auto pow = [](int exp) {
        u128 r = 1;
        for (int i = 0; i < exp; i++) {
            r <<= 1;
            if (r >> 64 & 1) {
                r ^= mod;
            }
        }
        return (u64)r;
    };

    static constexpr u64 r = pow(64);
    static constexpr u64 r2 = pow(128);

    static_assert((u64)clmul_constexpr(u64(mod), inv) == 1);

   public:
    static u64 reduce(u128 val) {
        u64 f = clmul(val, inv);
        // return (val ^ clmul(f, (u64)mod) ^ u128(f) << 64) >> 64;
        return val >> 64 ^ clmul(f, (u64)mod) >> 64 ^ f;
    }

    // private:
   public:
    u64 val;

   public:
    Field(u64 val, int) : val(val) { ; }

   public:
    explicit Field() : val(0) { ; }
    explicit Field(u64 val) : val(reduce(clmul(val, r2))) { ; }

    static Field n(int64_t n) { return Field(n & 1); }

    Field operator-() const { return Field(val, 0); }
    Field operator+(const Field& other) const { return Field(val ^ other.val, 0); }
    Field operator-(const Field& other) const { return Field(val ^ other.val, 0); }

    // Field operator*(const Field& other) const { return Field(take_mod(clmul(val, other.val), mod)); }
    Field operator*(const Field& other) const {
        return Field(reduce(clmul(val, other.val)), 0);
    }

    Field power(u64 exp) const {
        Field r = n(1);
        Field b = *this;
        for (; exp; exp >>= 1) {
            if (exp & 1) r *= b;
            b *= b;
        }
        return r;
    }

    Field inverse() const {
        Field res = power(~u64() - 1);
        assert(res * *this == n(1));
        return res;
    }

    Field& operator+=(const Field& other) { return *this = *this + other; }
    Field& operator-=(const Field& other) { return *this = *this - other; }
    Field& operator*=(const Field& other) { return *this = *this * other; }

    // u64 get() const { return val; }
    u64 get() const { return reduce(val); }

    bool operator==(const Field& other) const { return val == other.val; }

    // friend std::ostream& operator<<(std::ostream& out, const Field& val) { return out << val.val; }
};

};  // namespace Field_64

namespace conv {

size_t total = 0;

template <typename R>
std::vector<R> convolve_naive(const std::vector<R>& a, const std::vector<R>& b) {
    if (a.empty() || b.empty()) {
        return {};
    }
    using namespace Field_64;
    if constexpr (std::is_same_v<R, Field>) {
        size_t sz = a.size() + b.size() - 1;
        __m128i* ptr = (__m128i*)_mm_malloc(16 * sz, 16);
        memset(ptr, 0, 16 * sz);

        for (size_t i = 0; i < a.size(); i++) {
            for (size_t j = 0; j < b.size(); j++) {
                ptr[i + j] ^= clmul_vec(a[i].val, b[j].val);
            }
        }

        std::vector<R> c(sz);
        for (size_t i = 0; i < sz; i++) {
            c[i] = R(R::reduce((u128)ptr[i]), 0);
        }
        _mm_free(ptr);
        return c;
    } else {
        std::vector<R> c(a.size() + b.size() - 1);
        for (size_t i = 0; i < a.size(); i++) {
            for (size_t j = 0; j < b.size(); j++) {
                c[i + j] += a[i] * b[j];
            }
        }
        return c;
    }
}

static constexpr size_t pow3(int k) {
    constexpr auto ar = [&] {
        std::array<size_t, 32> ar;
        ar[0] = 1;
        for (size_t i = 1; i < ar.size(); i++) {
            ar[i] = ar[i - 1] * 3;
        }
        return ar;
    }();
    return ar[k];
}

template <typename R>
struct Meow {
    // template <typename R>
    struct L {
        R x, y;

        L(R x = R(), R y = R()) : x(x), y(y) { ; }

        L operator-() const { return L(-x, -y); }
        friend L operator+(const L& a, const L& b) { return L(a.x + b.x, a.y + b.y); }
        friend L operator-(const L& a, const L& b) { return L(a.x - b.x, a.y - b.y); }
        friend L operator*(const L& a, const L& b) {
            R xx = a.x * b.x;
            R yy = a.y * b.y;
            R magic = (a.x + a.y) * (b.x + b.y);
            R xy_yx = magic - xx - yy;
            return L(xx - yy, xy_yx - yy);
        }

        L& operator+=(const L& other) { return *this = *this + other; }
        L& operator-=(const L& other) { return *this = *this - other; }
        L& operator*=(const L& other) { return *this = *this * other; }

        L mul_by_w() const { return L(-y, x - y); }
        L mul_by_w2() const { return L(y - x, -x); }

        L conj() const { return L(x - y, -y); }

        bool operator==(const L& other) const { return x == other.x && y == other.y; }

        // friend std::ostream& operator<<(std::ostream& out, const L& val) { return out << "(" << val.x << "," << val.y << ")"; }
    };

    using K = L;
    using W = int;

    void mul_by_w(std::span<K> s, int w) {
        if (w == 1) {
            for (K& i : s) {
                i = i.mul_by_w();
            }
        } else if (w == 2) {
            for (K& i : s) {
                i = i.mul_by_w2();
            }
        } else {
            // assert(w == 0);
        }
    }

    template <int lg, bool inverse>
    void butterfly_x3(W w, std::span<K> a, std::span<K> b, std::span<K> c) {
        int sh1 = w % pow3(lg);
        int tw1 = w / pow3(lg);
        int sh2 = (2 * w) % pow3(lg);
        int tw2 = (2 * w) / pow3(lg) % 3;

        if (!inverse) {
            std::rotate(b.begin(), b.end() - sh1, b.end());
            std::rotate(c.begin(), c.end() - sh2, c.end());

            mul_by_w(b.subspan(0, sh1), (1 + tw1) % 3);
            mul_by_w(c.subspan(0, sh2), (1 + tw2) % 3);
            mul_by_w(b.subspan(sh1), tw1);
            mul_by_w(c.subspan(sh2), tw2);

            for (int i = 0; i < pow3(lg); i++) {
                K x = a[i], y = b[i], z = c[i];
                a[i] = x + y + z;
                b[i] = x + y.mul_by_w() + z.mul_by_w2();
                c[i] = x + y.mul_by_w2() + z.mul_by_w();
            }
        } else {
            for (int i = 0; i < pow3(lg); i++) {
                K x = a[i], y = b[i], z = c[i];
                a[i] = x + y + z;
                b[i] = x + y.mul_by_w2() + z.mul_by_w();
                c[i] = x + y.mul_by_w() + z.mul_by_w2();
            }

            mul_by_w(b.subspan(0, sh1), (5 - tw1) % 3);
            mul_by_w(c.subspan(0, sh2), (5 - tw2) % 3);
            mul_by_w(b.subspan(sh1), (3 - tw1) % 3);
            mul_by_w(c.subspan(sh2), (3 - tw2) % 3);

            std::rotate(b.begin(), b.begin() + sh1, b.end());
            std::rotate(c.begin(), c.begin() + sh2, c.end());
        }
    }

    template <int lg, int lg2, bool inverse>
    void aux_transform(std::span<K> a, bool conj = false) {
        auto rec = [&](auto rec, int k, int i, int w) -> void {
            size_t n = pow3(k);
            size_t m = pow3(lg2);
            auto recurse = [&] {
                for (int j = 0; j < 3; j++) {
                    rec(rec, k - 1, i + n * j, (w / 3 + pow3(lg2) * j) % pow3(lg2 + 1));
                }
            };
            if (inverse && k >= lg2 + 1) {
                recurse();
            }
            for (int j = 0; j < n; j += m) {
                butterfly_x3<lg2, inverse>(w / 3, a.subspan(i + j, m), a.subspan(i + j + n, m), a.subspan(i + j + 2 * n, m));
            }
            if (!inverse && k >= lg2 + 1) {
                recurse();
            }
        };
        rec(rec, lg - 1, 0, (1 + conj) * pow3(lg2));
    }

    std::vector<std::vector<K>> vec;

    std::array<std::span<K>, 2> get(int lg) {
        while (vec.size() <= lg) {
            int k = vec.size();
            vec.emplace_back(2 * pow3(k));
        }
        std::span<K> sp = vec[lg];
        std::fill(sp.begin(), sp.end(), K());
        return {sp.subspan(0, pow3(lg)), sp.subspan(pow3(lg))};
    }

    template <int lg>
    void mul_mod_naive(std::span<K> a, std::span<K> b) {
        // std::array<K, pow3(LG)> out;
        // out.fill(K());
        constexpr size_t n = pow3(lg);

        if constexpr (std::is_same_v<R, Field_64::Field>) {
            using namespace Field_64;

            alignas(64) __m128i out_x[n], out_y[n];
            memset(out_x, 0, sizeof(out_x));
            memset(out_y, 0, sizeof(out_y));

            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    auto [x1, y1] = a[i];
                    auto [x2, y2] = b[j];

                    __m128i xx = clmul_vec(x1.val, x2.val);
                    __m128i yy = clmul_vec(y1.val, y2.val);
                    __m128i xy = clmul_vec(x1.val, y2.val);
                    __m128i yx = clmul_vec(y1.val, x2.val);

                    __m128i x = xx ^ yy, y = xy ^ yx ^ yy;

                    if (i + j < n) {
                        out_x[i + j] ^= x, out_y[i + j] ^= y;
                    } else {
                        std::swap(x, y), y ^= x;
                        out_x[i + j - n] ^= x, out_y[i + j - n] ^= y;
                    }
                }
            }

            for (int i = 0; i < n; i++) {
                a[i] = K(R(R::reduce((u128)out_x[i]), 0), R(R::reduce((u128)out_y[i]), 0));
            }
        } else {
            K out[n];
            std::fill(out, out + n, K());
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    K p = a[i] * b[j];
                    if (i + j < n) {
                        out[i + j] += p;
                    } else {
                        out[i + j - n] += p.mul_by_w();
                    }
                }
            }
            std::copy(out, out + n, a.begin());
        }
    }

    template <int lg>
    void convolve_aux(std::span<K> a, std::span<K> b) {
        // int lg = 0;
        // size_t n = 1;
        // while (n < a.size()) n *= 3, lg++;
        // assert(n == a.size());

        size_t n = pow3(lg);
        assert(n == a.size());

        constexpr int LG = 3;
        if (lg <= LG) {
            mul_mod_naive<lg>(a, b);
            return;
        }

        constexpr int lg2 = (lg + 1) / 2;
        size_t m = pow3(lg2);

        std::span<K> xa = a, xb = b;
        // std::vector<K> ya(n), yb(n);
        auto [ya, yb] = get(lg);
        for (size_t i = 0; i < n; i++) {
            ya[i] = a[i].conj(), yb[i] = b[i].conj();
        }

        aux_transform<lg, lg2, false>(xa);
        aux_transform<lg, lg2, false>(xb);
        aux_transform<lg, lg2, false>(ya, true);
        aux_transform<lg, lg2, false>(yb, true);

        for (int i = 0; i < n; i += m) {
            convolve_aux<lg2>(std::span(xb).subspan(i, m), std::span(xa).subspan(i, m));
            convolve_aux<lg2>(std::span(yb).subspan(i, m), std::span(ya).subspan(i, m));
        }

        aux_transform<lg, lg2, true>(xb);
        aux_transform<lg, lg2, true>(yb, true);

        for (K& k : yb) {
            k = k.conj();
        }

        std::fill(a.begin(), a.end(), K());

        for (int i = 0; i < n; i += m) {
            for (int j = 0; j < m; j++) {
                // X == A  (mod x^m - w  )
                // X == B  (mod x^m - w^2)
                // X = A / (w - w^2) * (x^m - w^2)
                //   + B / (w^2 - w) * (x^m - w  )
                //
                // 1 / (a + bw) =
                // (a + b * conj(w)) / ((a + bw) * (a + b * conj(w))) =
                // (a - b - bw) / (a^2 - ab + b^2)
                //
                // w - w^2 = 1 + 2w = a + bw
                // a = 1, b = 2   ->  1 / (a + bw) = (-1 - 2w) / (1 - 2 + 4) = (-1 - 2w) / 3

                K x1 = xb[i + j].mul_by_w2() + yb[i + j].mul_by_w();
                K x2 = xb[i + j] + yb[i + j];

                a[i + j] += x1;
                if (i + j + m < n) {
                    a[i + j + m] += x2;
                } else {
                    a[i + j + m - n] += x2.mul_by_w();
                }
            }
        }
    }

    template <int lg = 0>
    void convolve(std::span<K> a, std::span<K> b) {
        if (pow3(lg) == a.size()) {
            convolve_aux<lg>(a, b);
        } else if constexpr (pow3(lg) < 1e7) {
            convolve<lg + 1>(a, b);
        } else {
            assert(false);
        }
    }
};

template <typename R>
std::vector<R> convolve(const std::vector<R>& a, const std::vector<R>& b) {
    if (a.empty() || b.empty()) {
        return {};
    }
    int lg = 0;
    while (2 * pow3(lg) < a.size() + b.size() - 1) lg++;
    size_t n = pow3(lg);

    total += n;
    if (a.size() * b.size() <= n * lg * 10) {
        return convolve_naive(a, b);
    }

    Meow<R> mw;
    using K = Meow<R>::K;

    std::vector<K> f(n), g(n);
    for (int i = 0; i < a.size() && i < 2 * n; i++) {
        (i < n ? f[i].x : f[i - n].y) = a[i];
    }
    for (int i = 0; i < b.size() && i < 2 * n; i++) {
        (i < n ? g[i].x : g[i - n].y) = b[i];
    }

    mw.convolve(f, g);

    std::vector<R> ans(a.size() + b.size() - 1);
    for (size_t i = 0; i < ans.size() && i < 2 * n; i++) {
        ans[i] = i < n ? f[i].x : f[i - n].y;
    }
    return ans;
}

template <typename R>
std::vector<R> convolve_transposed(const std::vector<R>& a, const std::vector<R>& b) {
    size_t n = a.size(), m = b.size();
    assert(n != 0);
    if (m < n) {
        return {};
    }
    size_t d = m - n + 1;

    if (0) {
        std::vector<R> res(d);
        for (int i = 0; i < d; i++) {
            R r = R();
            for (int j = 0; j < n; j++) {
                r += a[j] * b[i + j];
            }
            res[i] = r;
        }
        return res;
    }

    std::vector<R> p = convolve(std::vector<R>(a.rbegin(), a.rend()), b);
    p.erase(p.begin(), p.begin() + (n - 1));
    p.resize(d);
    return p;
}

};  // namespace conv

namespace meow {

using R = Field_64::Field;
using poly = std::vector<R>;

using conv::convolve, conv::convolve_transposed;

void remove_zeros(poly& p) {
    while (p.size() && p.back() == R()) p.pop_back();
}
R coeff(const poly& p, size_t ind) {
    return ind < p.size() ? p[ind] : R();
}

poly operator*(const poly& a, const poly& b) {
    size_t sz = a.size() + b.size();
    poly res = convolve(a, b);
    remove_zeros(res);
    return res;
}
poly operator+(const poly& a, const poly& b) {
    poly c(std::max(a.size(), b.size()));
    for (size_t i = 0; i < a.size(); i++) c[i] += a[i];
    for (size_t i = 0; i < b.size(); i++) c[i] += b[i];
    remove_zeros(c);
    return c;
}
poly operator-(const poly& a) {
    poly c(a.size());
    for (size_t i = 0; i < a.size(); i++) c[i] = -a[i];
    remove_zeros(c);
    return c;
}
poly operator-(const poly& a, const poly& b) {
    poly c(std::max(a.size(), b.size()));
    for (size_t i = 0; i < a.size(); i++) c[i] += a[i];
    for (size_t i = 0; i < b.size(); i++) c[i] -= b[i];
    remove_zeros(c);
    return c;
}

poly mod_xk(poly a, size_t k) {
    if (a.size() > k) a.resize(k);
    return a;
}

poly inv_series(const poly& a, size_t sz) {
    poly b = {a[0]};
    auto get = [&](int a, int b) {
        int r = (b + a - 1) / a;
        return std::__lg(r - 1) + 1;
    };
    while (b.size() < sz) {
        size_t n = b.size();
        // while (n > 1 && get(n, sz) == get(n - 1, sz)) {
        //     b.pop_back(), n--;
        // }
        // b = b * (poly{R::n(2)} - mod_xk(a, std::min(sz, 2 * n)) * b);
        b = b * b * mod_xk(a, std::min(sz, 2 * n));
        b.resize(2 * n);
    }
    b.resize(sz);
    return b;
}

std::pair<poly, poly> divmod(const poly& a, const poly& b) {
    size_t n = a.size(), m = b.size();
    if (n < m) {
        return {poly{}, a};
    }
    size_t d = n - m + 1;
    poly ra(a.rbegin(), a.rbegin() + d);
    poly rb(b.rbegin(), b.rbegin() + std::min(d, m));
    poly q = mod_xk(ra * inv_series(rb, d), d);
    std::reverse(q.begin(), q.end());
    poly r = a - b * q;
    assert(r.size() < m);
    return {q, r};
}

poly operator%(const poly& a, const poly& b) {
    return divmod(a, b).second;
}

R eval(const poly& a, R pt) {
    R res = R();
    for (size_t i = 0; i < a.size(); i++) {
        res = res * pt + a.rbegin()[i];
    }
    return res;
}

std::vector<R> evaluate_naive(const poly& a, const std::vector<R>& pts) {
    if (pts.empty()) {
        return {};
    }

    size_t m = pts.size();
    std::vector<poly> vec(2 * m);
    for (int i = 0; i < m; i++) {
        vec[m + i] = poly{-pts[i], R::n(1)};
    }
    for (int i = m - 1; i > 0; i--) {
        vec[i] = vec[2 * i] * vec[2 * i + 1];
    }
    vec[1] = a % vec[1];
    for (int i = 2; i < 2 * m; i++) {
        vec[i] = vec[i / 2] % vec[i];
    }
    std::vector<R> res(m);
    for (int i = 0; i < m; i++) {
        res[i] = coeff(vec[m + i], 0);
    }
    return res;
}

std::vector<R> evaluate(const poly& a, const std::vector<R>& pts) {
    if (pts.empty()) {
        return {};
    }
    // return evaluate_naive(a, pts);

    size_t n = std::max<size_t>(a.size(), 1), m = pts.size();
    std::vector<poly> vec(2 * m);
    for (int i = 0; i < m; i++) {
        vec[m + i] = poly{R::n(1), -pts[i]};
    }
    for (int i = m - 1; i > 0; i--) {
        vec[i] = vec[2 * i] * vec[2 * i + 1];
    }

    poly a2 = a;
    a2.resize(n + m - 1);
    vec[1] = convolve_transposed(inv_series(vec[1], n), a2);

    for (int i = 1; i < m; i++) {
        poly tmp = vec[2 * i];
        vec[2 * i] = convolve_transposed(vec[2 * i + 1], vec[i]);
        vec[2 * i + 1] = convolve_transposed(tmp, vec[i]);
    }
    std::vector<R> res(m);
    for (int i = 0; i < m; i++) {
        res[i] = coeff(vec[m + i], 0);
    }
    return res;
}

poly interpolate(std::vector<std::pair<R, R>> pts) {
    size_t n = pts.size();

    std::vector<poly> vec(2 * n);
    for (int i = 0; i < n; i++) {
        vec[n + i] = poly{R::n(1), -pts[i].first};
    }
    for (int i = n - 1; i > 0; i--) {
        vec[i] = convolve(vec[2 * i], vec[2 * i + 1]);
    }
    std::vector<poly> vec2 = vec;

    std::vector<R> p = vec[1];
    std::reverse(p.begin(), p.end());
    p.erase(p.begin());
    for (size_t i = 1; i < p.size(); i += 2) {
        p[i] = R();
    }
    p.resize(n + n - 1);

    vec[1] = convolve_transposed(inv_series(vec[1], n), p);

    for (int i = 1; i < n; i++) {
        vec[2 * i] = convolve_transposed(vec2[2 * i + 1], vec[i]);
        vec[2 * i + 1] = convolve_transposed(vec2[2 * i], vec[i]);
    }

    for (int i = 0; i < n; i++) {
        R f = coeff(vec[n + i], 0);
        vec[n + i] = poly{f.inverse() * pts[i].second};
    }
    for (int i = 1; i < 2 * n; i++) {
        std::reverse(vec2[i].begin(), vec2[i].end());
    }
    for (int i = n - 1; i > 0; i--) {
        vec[i] = vec[2 * i] * vec2[2 * i + 1] + vec[2 * i + 1] * vec2[2 * i];
    }

    poly res = std::move(vec[1]);
    // for (auto [x, y] : pts) {
    //     std::cerr << eval(res, x).get() << " " << y.get() << "\n";
    //     assert(eval(res, x) == y);
    // }
    return res;
}

};  // namespace meow

using meow::R, meow::poly;

void test_eval() {
    std::mt19937_64 rnd;

    int n = 1e5;

    poly p(n / 10);
    std::vector<R> pts(n);
    for (auto& i : p) i = R(rnd());
    for (auto& i : pts) i = R(rnd());

    std::vector<R> vals = meow::evaluate(p, pts);

    for (int i = 0; i < 100; i++) {
        int ind = rnd() % pts.size();
        assert(vals[ind] == meow::eval(p, pts[ind]));
    }

    if (1) {
        for (int i = 0; i < 10; i++) {
            poly p(rnd() % n);
            std::vector<R> pts(rnd() % n + 1);
            for (auto& i : p) i = R(rnd());
            for (auto& i : pts) i = R(rnd());

            std::vector<R> vals = meow::evaluate(p, pts);

            for (int i = 0; i < 10; i++) {
                int ind = rnd() % pts.size();
                assert(vals[ind] == meow::eval(p, pts[ind]));
            }
        }
    }
}

void test_interpolate() {
    std::mt19937_64 rnd;

    int n = 1e5;

    std::vector<std::pair<R, R>> pts(n);
    for (auto& [x, y] : pts) x = R(rnd()), y = R(rnd());

    poly p = meow::interpolate(pts);

    for (int i = 0; i < 100; i++) {
        int ind = rnd() % n;
        assert(meow::eval(p, pts[ind].first) == pts[ind].second);
    }
    std::vector<R> x(n);
    for (int i = 0; i < n; i++) {
        x[i] = pts[i].first;
    }

    std::vector<R> vals = meow::evaluate(p, x);
    for (int i = 0; i < n; i++) {
        assert(vals[i] == pts[i].second);
    }
}

#include <set>

int main() {
    // freopen("cum.in", "r", stdin);

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // test_eval();
    // test_interpolate();

    // return 0;

    using meow::R;

    int tp;
    std::cin >> tp;
    int n;
    std::cin >> n;

    if (tp == 1) {
        std::vector<std::pair<uint64_t, uint64_t>> input(n);
        for (auto& [a, b] : input) {
            std::cin >> a >> b;
        }
        std::set<uint64_t> set;
        for (auto& [a, b] : input) {
            set.insert(a);
        }
        assert(set.size() == n);

        std::vector<std::pair<R, R>> pts(n);
        for (int i = 0; i < n; i++) {
            pts[i] = {R(input[i].first), R(input[i].second)};
        }

        poly p = meow::interpolate(pts);

        std::cout << p.size() << "\n";
        for (int i = 0; i < p.size(); i++) {
            std::cout << p[i].get() << "\n\n"[i + 1 == p.size()];
        }
    } else {
        poly p(n);
        for (auto& r : p) {
            uint64_t val;
            std::cin >> val;
            r = R(val);
        }

        int q;
        std::cin >> q;
        std::vector<R> pts(q);
        for (auto& r : pts) {
            uint64_t val;
            std::cin >> val;
            r = R(val);
        }

        std::vector<R> res = meow::evaluate(p, pts);

        for (int i = 0; i < res.size(); i++) {
            std::cout << res[i].get() << "\n\n"[i + 1 == res.size()];
        }
    }
}
