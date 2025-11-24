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

namespace aux {
static constexpr u128 mod = 0b11011 | u128(1) << 64;

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

}  // namespace aux

using namespace aux;

struct Field {
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

    Field operator-() const { return Field() - *this; }
    Field& operator+=(const Field& other) { return val ^= other.val, *this; }
    Field& operator-=(const Field& other) { return val ^= other.val, *this; }
    Field& operator*=(const Field& other) {
        val = reduce(clmul(val, other.val));
        return *this;
    }

    friend Field operator+(Field a, Field b) { return a += b; }
    friend Field operator-(Field a, Field b) { return a -= b; }
    friend Field operator*(Field a, Field b) { return a *= b; }

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

    // u64 get() const { return val; }
    u64 get() const { return reduce(val); }

    bool operator==(const Field& other) const { return val == other.val; }

    // friend std::ostream& operator<<(std::ostream& out, const Field& val) { return out << val.val; }
};

};  // namespace Field_64

namespace conv {

using F = Field_64::Field;

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

std::vector<F> gen(std::vector<F> bs) {
    int lg = bs.size();
    std::vector<F> G(1 << lg);
    G[0] = F();
    for (int k = 0; k < lg; k++) {
        for (int i = 0; i < (1 << k); i++) {
            G[(1 << k) + i] = G[i] + bs[k];
        }
    }
    return G;
}

struct FFT {
    struct Data {
        std::vector<F> bt, gm, dt;
        std::vector<F> G;

        std::vector<F> pw, pw_inv;

        F bt_m_inv;

        mutable std::vector<F> aux;

        void init() {
            int lg = bt.size();

            aux.resize(1 << lg);

            F b = bt.back(), b_inv = bt.back().inverse();

            gm.assign(lg - 1, F());
            for (int i = 0; i < lg - 1; i++) {
                gm[i] = bt[i] * b_inv;
            }
            dt.assign(lg - 1, F());
            for (int i = 0; i < lg - 1; i++) {
                dt[i] = gm[i] * gm[i] + gm[i];
            }

            G = gen(gm);

            pw.assign(1 << lg, F());
            pw_inv.assign(1 << lg, F());
            F x = F::n(1), y = F::n(1);
            for (int i = 0; i < (1 << lg); i++, x *= b, y *= b_inv) {
                pw[i] = x, pw_inv[i] = y;
            }
            bt_m_inv = b_inv;
        }
    };

    std::vector<Data> data;

    void prepare(int lg) {
        if (data.size() > lg) {
            return;
        }

        // data.resize(lg + 1);
        // data[lg].bt.resize(lg);
        // for (int i = 0; i < lg; i++) {
        //     data[lg].bt[i] = F(1 << i);
        // }
        // for (int k = lg; k > 0; k--) {
        //     data[k].init();
        //     data[k - 1].bt = data[k].dt;
        // }
        // return;

        {
            std::mt19937_64 rnd;
            std::vector<F> vec;
            while (vec.size() < lg) {
                vec.clear();
                for (F x = F(rnd()); x != F(); x = x * x + x) {
                    vec.push_back(x);
                }
            }
            vec.erase(vec.begin(), vec.begin() + (vec.size() - lg));

            data.resize(lg + 1);
            data[lg].bt = vec;

            for (int k = lg; k > 0; k--) {
                data[k].init();
                data[k - 1].bt = data[k].dt;
            }
        }
    }
};

FFT fft_aux;

template <bool inverse = false>
__attribute__((optimize("O3")))
// __attribute__((optimize("O3"), target("avx2")))
void taylor(std::span<F> f) {
    for (size_t n = inverse ? 1 : f.size() / 4; inverse ? n * 4 <= f.size() : n >= 1; inverse ? n *= 2 : n /= 2) {
        for (size_t t = 0; t < f.size(); t += 4 * n) {
            for (size_t i = 0; i < n; i++) {
                F b = f[t + 1 * n + i], c = f[t + 2 * n + i], d = f[t + 3 * n + i];
                f[t + 1 * n + i] = (inverse ? b + c : b + c + d), f[t + 2 * n + i] = c + d;
            }
        }
    }
}

template <bool inverse = false>
void fft(std::span<F> f) {
    if (f.size() == 1) {
        return;
    }
    size_t n = f.size();
    int lg = std::__lg(n);

    const FFT::Data& d = fft_aux.data[lg];

    F bt_m_inv = d.bt_m_inv;
    const std::vector<F>& G = d.G;

    if (n == 2) {
        if (!inverse) {
            // f[1] = f[0] + f[1] * d.bt.back();
            f[1] = f[0] + f[1];
        } else {
            // f[1] = (f[0] + f[1]) * bt_m_inv;
            f[1] = f[0] + f[1];
        }
        return;
    }

    std::span<F> u(d.aux.begin(), d.aux.begin() + n / 2);
    std::span<F> v(d.aux.begin() + n / 2, d.aux.begin() + n);

    if (!inverse) {
        // if (d.bt.back() != F::n(1)) {
        // for (size_t i = 0; i < n; i++) f[i] *= d.pw[i];
        // }

        taylor(f);

        for (int i = 0; i < n / 2; i++) {
            u[i] = f[2 * i], v[i] = f[2 * i + 1];
        }

        fft(u);
        fft(v);

        for (int i = 0; i < n / 2; i++) {
            F a = u[i] + G[i] * v[i];
            F b = a + v[i];

            f[i] = a, f[i + n / 2] = b;
        }
    } else {
        for (int i = 0; i < n / 2; i++) {
            F a = f[i], b = f[n / 2 + i];

            v[i] = a + b;
            u[i] = a + G[i] * v[i];
        }

        fft<true>(u);
        fft<true>(v);

        for (int i = 0; i < n / 2; i++) {
            f[2 * i] = u[i], f[2 * i + 1] = v[i];
        }

        taylor<true>(f);

        // if (d.bt.back() != F::n(1)) {
        // for (size_t i = 0; i < n; i++) f[i] *= d.pw_inv[i];
        // }
    }
}

F eval(const std::vector<F>& vec, F pt) {
    F res = F();
    for (size_t i = 0; i < vec.size(); i++) {
        res = res * pt + vec.rbegin()[i];
    }
    return res;
}

std::vector<F> convolve(std::vector<F> a, std::vector<F> b) {
    if (a.empty() || b.empty()) {
        return {};
    }

    size_t n = a.size(), m = b.size();
    int lg = 0;
    while ((1 << lg) < (n + m - 1)) lg++;

    if (n * m <= int64_t(1 << lg) * (lg + 1) * (lg + 1)) {
        return convolve_naive(a, b);
    }

    if (3 < lg && n + m - 1 == (1 << lg - 1) + 1) {
        // std::cerr << (1 << lg) << " " << (n + m - 1) << "\n";
        std::vector<F> dlt(m);
        for (int i = 0; i < m; i++) {
            dlt[i] = a.back() * b[i];
        }
        a.pop_back();

        std::vector<F> res = convolve(std::move(a), std::move(b));
        res.push_back(F());
        for (int i = 0; i < m; i++) {
            res[(n - 1) + i] += dlt[i];
        }
        return res;
    }

    fft_aux.prepare(lg);

    a.resize(1 << lg), b.resize(1 << lg);

    fft(a);
    fft(b);

    for (int i = 0; i < (1 << lg); i++) a[i] *= b[i];

    fft<true>(a);

    a.resize(n + m - 1);
    return a;
}

void test_fft() {
    using F = Field_64::Field;

    std::mt19937_64 rnd;

    for (int k = 0; k <= 10; k++) {
        std::vector<F> p(1 << k);
        for (auto& i : p) i = F(1);

        // std::vector<F> bs(k);
        // for (auto& i : bs) i = F(rnd() & 7);
        // for (int i = 0; i < k; i++) {
        //     bs[i] = F(1 << i);
        // }

        conv::fft_aux.prepare(k);
        std::vector<F> bs = fft_aux.data[k].bt;

        std::vector<F> vals = p;
        conv::fft(vals);

        std::vector<F> pts = conv::gen(bs);

        for (int i = 0; i < pts.size(); i++) {
            F a = vals[i];
            F b = conv::eval(p, pts[i]);
            if (a != b) {
                std::cerr << p.size() << " " << i << " " << a.get() << " " << b.get() << std::endl;
            }
            assert(vals[i] == conv::eval(p, pts[i]));
        }

        std::vector<F> p2 = vals;
        conv::fft<true>(p2);
        assert(p2 == p);
    }
}

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

// int32_t main() {
//     conv::test_fft();
//     conv::stress_test();
//     for (int i = 0; i < 5; i++) conv::bench();
// }

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
    poly b = {a[0].inverse()};
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

void test_eval() {
    std::mt19937_64 rnd;

    int n = 1e5;

    poly p(n / 10);
    std::vector<R> pts(n);
    for (auto& i : p) i = R(rnd());
    for (auto& i : pts) i = R(rnd());

    std::vector<R> vals = meow::evaluate(p, pts);

    for (int i = 0; i < 10; i++) {
        int ind = rnd() % pts.size();
        assert(vals[ind] == meow::eval(p, pts[ind]));
    }

    if (0) {
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

    for (int i = 0; i < 10; i++) {
        int ind = rnd() % n;
        assert(meow::eval(p, pts[ind].first) == pts[ind].second);
    }
    // std::vector<R> x(n);
    // for (int i = 0; i < n; i++) {
    //     x[i] = pts[i].first;
    // }

    // std::vector<R> vals = meow::evaluate(p, x);
    // for (int i = 0; i < n; i++) {
    //     assert(vals[i] == pts[i].second);
    // }
}

};  // namespace meow

using meow::R, meow::poly;
using meow::test_eval, meow::test_interpolate;

#include <set>

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

#ifdef LOCAL
    conv::bench();
    conv::test_fft();
    conv::stress_test();
    test_eval();
    test_interpolate();
    return 0;
#endif

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
            std::cout << p[i].get() << " \n"[i + 1 == p.size()];
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
