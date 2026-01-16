#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

using u64 = uint64_t;

struct Cum {
    u64 a, b;

    constexpr Cum(u64 a = 0, u64 b = 0) : a(a), b(b) { ; }

    constexpr Cum operator+(const Cum& other) const { return Cum(a + other.a, b + other.b); }
    constexpr Cum operator-(const Cum& other) const { return Cum(a - other.a, b - other.b); }
    constexpr Cum operator*(const Cum& other) const { return Cum(a * other.a - b * other.b, a * other.b + b * other.a - b * other.b); }

    constexpr void operator+=(const Cum& other) { *this = *this + other; }
    constexpr void operator-=(const Cum& other) { *this = *this - other; }
    constexpr void operator*=(const Cum& other) { *this = *this * other; }

    constexpr Cum operator-() const { return Cum(-a, -b); }

    constexpr bool operator==(const Cum& other) const { return a == other.a && b == other.b; }
    constexpr bool operator!=(const Cum& other) const { return a != other.a || b != other.b; }
};

template <auto base, size_t N = 100>
constexpr decltype(base) pow_fixed(size_t ind) {
    using T = decltype(base);
    constexpr std::array<T, N> data = []() {
        std::array<T, N> res;
        res[0] = T(1);
        for (size_t i = 1; i < N; i++) {
            res[i] = res[i - 1] * base;
        }
        return res;
    }();
    return data[ind];
}

class SSA {
   private:
    void fill(int lg, int lg2, std::span<Cum> data, std::span<const Cum> a) {
        int n = pow_fixed<3u>(lg), m = pow_fixed<3u>(lg2);
        for (int i = 0; i < n; i += m / 3) {
            for (int j = 0; j < m / 3; j++) {
                data[i * 3 + j] = a[i + j];
            }
        }
    }

    static Cum cum_inv(Cum w) {
        constexpr Cum x = Cum(0, 1), x2 = x * x;
        Cum w_inv = w == 1 ? 1 : (w == x ? x2 : (assert(w == x2), x));
        assert(w_inv * w == 1);
        return w_inv;
    }

    template <bool inverse>
    static void butterfly_x3(int m, std::span<Cum> sp1, std::span<Cum> sp2, std::span<Cum> sp3, std::span<Cum> tmp, int pw, Cum w) {
        constexpr Cum x = Cum(0, 1), x2 = x * x;
        assert(x * x2 == 1);
        Cum w2 = w * w;
        int pw2 = (2 * pw) % m;
        if (2 * pw >= m) {
            w2 *= x;
        }
        if (!inverse) {
            std::span<Cum> tmp1 = tmp.subspan(0, m), tmp2 = tmp.subspan(m, m), tmp3 = tmp.subspan(2 * m, m);
            std::copy(sp1.begin(), sp1.end(), tmp1.begin()), std::copy(sp2.begin(), sp2.end(), tmp2.begin()), std::copy(sp3.begin(), sp3.end(), tmp3.begin());
            std::rotate(tmp2.begin(), tmp2.end() - pw, tmp2.end()), std::rotate(tmp3.begin(), tmp3.end() - pw2, tmp3.end());

            ({ for (int i = 0; i < pw; i++) tmp2[i] *= x; for (int i = 0; i < pw2; i++) tmp3[i] *= x; });
            for (int i = 0; i < m; i++) {
                sp1[i] = tmp1[i] + tmp2[i] * w + tmp3[i] * w2;
                sp2[i] = tmp1[i] + tmp2[i] * (w * x) + tmp3[i] * (w2 * x2);
                sp3[i] = tmp1[i] + tmp2[i] * (w * x * x) + tmp3[i] * (w2 * x2 * x2);
            }
        } else {
            for (int i = 0; i < m; i++) {
                Cum p = sp1[i], q = sp2[i], r = sp3[i];
                sp1[i] = p + q + r;
                sp2[i] = p + q * x2 + r * x;
                sp3[i] = p + q * x + r * x2;
            }

            Cum w_inv = cum_inv(w), w2_inv = cum_inv(w2);

            for (int i = 0; i < pw; i++) sp2[i] *= x2;
            for (int i = 0; i < pw2; i++) sp3[i] *= x2;

            for (int i = 0; i < m; i++) sp2[i] *= w_inv;
            for (int i = 0; i < m; i++) sp3[i] *= w2_inv;

            std::rotate(sp2.begin(), sp2.begin() + pw, sp2.end()), std::rotate(sp3.begin(), sp3.begin() + pw2, sp3.end());
        }
    }

    template <bool inverse>
    void transform(int lg, int lg2, std::span<Cum> data) {
        int n = pow_fixed<3u>(lg), m = pow_fixed<3u>(lg2);
        std::vector<Cum> tmp(3 * m);

        [&](this auto fuck, std::span<Cum> sp, int k, int pw) -> void {
            assert(sp.size() == 3 * k);
            std::span<Cum> sp1 = sp.subspan(0, k), sp2 = sp.subspan(k, k), sp3 = sp.subspan(2 * k, k);
            for (int step : (!inverse ? std::array{0, 1} : std::array{1, 0})) {
                if (step == 0) {
                    for (int i = 0; i < k; i += m) {
                        butterfly_x3<inverse>(m, sp1.subspan(i, m), sp2.subspan(i, m), sp3.subspan(i, m), tmp, pw % m, pow_fixed<Cum(0, 1), 3>(pw / m));
                    }
                }
                if (step == 1) {
                    if (k > m) {
                        assert(pw % 3 == 0);
                        fuck(sp1, k / 3, (pw + m * 0) / 3 % (3 * m));
                        fuck(sp2, k / 3, (pw + m * 1) / 3 % (3 * m));
                        fuck(sp3, k / 3, (pw + m * 2) / 3 % (3 * m));
                    } else {
                        assert(k == m);
                    }
                }
            }
        }(data, n, m / 3);
        if (inverse) {
            constexpr u64 inv3 = (2 * (__uint128_t(1) << 64) + 1) / 3;
            u64 pw_inv3 = pow_fixed<inv3>(lg - lg2 + 1);
            for (auto& cm : data) {
                cm *= pw_inv3;
            }
        }
    }

    // mod u^3^lg - x
    void convolve_aux(int lg, std::span<const Cum> a, std::span<const Cum> b, std::span<Cum> out) {
        int n = pow_fixed<3u>(lg);
        int lg2 = lg / 2 + 1;

        if (lg2 >= lg - 1) {
            assert(out.size() == n);
            std::fill(out.begin(), out.end(), Cum());
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n - i; j++) {
                    out[i + j] += a[i] * b[j];
                }
                for (int j = n - i; j < n; j++) {
                    out[i + j - n] += a[i] * b[j] * Cum(0, 1);
                }
            }
            return;
        } else {
            std::cerr << lg << "->" << lg2 << " \n";

            int m = pow_fixed<3u>(lg2);
            std::vector<Cum> data_a(3 * n), data_b(3 * n);

            fill(lg, lg2, data_a, a);
            fill(lg, lg2, data_b, b);
            transform<false>(lg, lg2, data_a);
            transform<false>(lg, lg2, data_b);

            std::vector<Cum> tmp(m);
            for (int i = 0; i < 3 * n; i += m) {
                convolve_aux(lg2, std::span<Cum>(data_a).subspan(i, m), std::span<Cum>(data_b).subspan(i, m), tmp);
                std::copy(tmp.begin(), tmp.end(), data_a.begin() + i);
            }
            transform<true>(lg, lg2, data_a);

            std::fill(out.begin(), out.end(), Cum());
            for (int i = 0; i < n; i += m / 3) {
                for (int j = 0; j < m / 3; j++) {
                    out[i + j] += data_a[3 * i + j];
                }
                for (int j = 0; j < m / 3; j++) {
                    int ind = i + j + m / 3;
                    Cum val = data_a[3 * i + j + (m / 3)];
                    if (ind < n) {
                        out[ind] += val;
                    } else {
                        out[ind - n] += val * Cum(0, 1);
                    }
                }
            }
        }
    }

   public:
    std::vector<u64> convolve(std::vector<u64> a, std::vector<u64> b) {
        int lg = 0;
        while (pow_fixed<3u>(lg) <= std::max(a.size(), b.size())) {
            lg++;
        }
        int n = pow_fixed<3u>(lg);
        std::vector<Cum> A(n), B(n), C(n);
        std::copy(a.begin(), a.end(), A.begin()), std::copy(b.begin(), b.end(), B.begin());
        convolve_aux(lg, A, B, C);
        std::vector<u64> c(std::max(0, (int)a.size() + (int)b.size() - 1));
        for (int i = 0; i < c.size(); i++) {
            c[i] = i < n ? C[i].a : C[i - n].b;
        }
        return c;
    }
};

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cout.tie(nullptr);

    int n, m;
    n = m = 1 << 19;
    // std::cin >> n >> m;
    std::vector<u64> a(n), b(m);
    // for (auto& i : a) {
    //     std::cin >> i;
    // }
    // for (auto& i : b) {
    //     std::cin >> i;
    // }

    SSA ssa;
    std::vector<u64> c = ssa.convolve(a, b);

    for (int i = 0; i < c.size(); i++) {
        std::cout << c[i] << " \n"[i + 1 == c.size()];
    }

    return 0;
}
