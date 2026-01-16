#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>

using u64 = uint64_t;

struct Cum {
    u64 a, b;

    constexpr Cum(u64 a = 0, u64 b = 0) : a(a), b(b) { ; }

    constexpr Cum operator+(const Cum& other) const { return Cum(a + other.a, b + other.b); }
    constexpr Cum operator-(const Cum& other) const { return Cum(a - other.a, b - other.b); }

    constexpr Cum operator*(const Cum& other) const {
        u64 p = a * other.a, q = b * other.b, r = (a + b) * (other.a + other.b) - p - q;
        return Cum(p - q, r - q);
    }
    constexpr Cum mul_w() const { return Cum(-b, a - b); }
    constexpr Cum mul_w2() const { return Cum(b - a, -a); }

    constexpr friend Cum operator+(u64 val, const Cum& cum) { return Cum(val) + cum; }
    constexpr friend Cum operator-(u64 val, const Cum& cum) { return Cum(val) - cum; }
    constexpr friend Cum operator*(u64 val, const Cum& cum) { return Cum(val) * cum; }

    constexpr void operator+=(const Cum& other) { *this = *this + other; }
    constexpr void operator-=(const Cum& other) { *this = *this - other; }
    constexpr void operator*=(const Cum& other) { *this = *this * other; }

    constexpr Cum operator-() const { return Cum(-a, -b); }
    constexpr Cum conj() const { return Cum(a - b, -b); }

    constexpr bool operator==(const Cum& other) const { return a == other.a && b == other.b; }
    constexpr bool operator!=(const Cum& other) const { return a != other.a || b != other.b; }

    static constexpr u64 inv3 = (2 * (__uint128_t(1) << 64) + 1) / 3;
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
    static Cum cum_inv(Cum w) {
        constexpr Cum x = Cum(0, 1), x2 = x * x;
        Cum w_inv = w == 1 ? 1 : (w == x ? x2 : (assert(w == x2), x));
        assert(w_inv * w == 1);
        return w_inv;
    }

    static void mul_w(std::span<Cum> sp, Cum p) {
        if (p == Cum(0, 1)) {
            for (auto& val : sp) val = val.mul_w();
        } else if (p == Cum(-1ull, -1ull)) {
            for (auto& val : sp) val = val.mul_w2();
        } else {
            assert(p == 1);
        }
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
            std::rotate(sp2.begin(), sp2.end() - pw, sp2.end()), std::rotate(sp3.begin(), sp3.end() - pw2, sp3.end());

            mul_w(sp2.subspan(0, pw), x * w), mul_w(sp3.subspan(0, pw2), x * w2);
            mul_w(sp2.subspan(pw), w), mul_w(sp3.subspan(pw2), w2);

            for (int i = 0; i < m; i++) {
                Cum a = sp1[i], b = sp2[i], c = sp3[i];
                sp1[i] = a + b + c;
                sp2[i] = a + b.mul_w() + c.mul_w2();
                sp3[i] = a + b.mul_w2() + c.mul_w();
            }

        } else {
            for (int i = 0; i < m; i++) {
                Cum p = sp1[i], q = sp2[i], r = sp3[i];
                sp1[i] = p + q + r;
                sp2[i] = p + q.mul_w2() + r.mul_w();
                sp3[i] = p + q.mul_w() + r.mul_w2();
            }

            Cum w_inv = cum_inv(w), w2_inv = cum_inv(w2);

            mul_w(sp2.subspan(0, pw), x2 * w_inv), mul_w(sp3.subspan(0, pw2), x2 * w2_inv);
            mul_w(sp2.subspan(pw), w_inv), mul_w(sp3.subspan(pw2), w2_inv);

            std::rotate(sp2.begin(), sp2.begin() + pw, sp2.end()), std::rotate(sp3.begin(), sp3.begin() + pw2, sp3.end());
        }
    }

    template <bool inverse>
    void transform(int lg, int lg2, std::span<Cum> data, bool conj = false) {
        int n = pow_fixed<3u>(lg), m = pow_fixed<3u>(lg2);
        // std::vector<Cum> tmp(3 * m);
        std::vector<Cum> tmp;

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
        }(data, n / 3, !conj ? m / 3 : 2 * (m / 3));
        if (inverse) {
            u64 pw_inv3 = pow_fixed<Cum::inv3>(lg - lg2);
            for (auto& cm : data) {
                cm *= pw_inv3;
            }
        }
    }

    void mul_small(int n, std::span<const Cum> a, std::span<const Cum> b, std::span<Cum> out) {
        constexpr int N = 1000;
        std::array<Cum, N> data;
        std::fill(data.begin(), data.begin() + 2 * n, Cum());
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                data[i + j] += a[i] * b[j];
            }
        }
        for (int i = 0; i < n; i++) {
            out[i] = data[i] + data[n + i].mul_w();
        }
    }

    std::map<int, std::array<std::vector<Cum>, 4>> map;

    // mod u^3^lg - x
    void convolve_aux(int lg, std::span<const Cum> a, std::span<const Cum> b, std::span<Cum> out) {
        int n = pow_fixed<3u>(lg);
        int lg2 = (lg + 1) / 2;

        if (lg2 >= lg - 1) {
            assert(out.size() == n);
            mul_small(n, a, b, out);
            return;
        } else {
            // std::cerr << lg << "->" << lg2 << " \n";

            int m = pow_fixed<3u>(lg2);
            // std::vector<Cum>
            //     data0_a(a.begin(), a.end()), data1_a(a.begin(), a.end()),
            //     data0_b(b.begin(), b.end()), data1_b(b.begin(), b.end());

            auto& [data0_a, data0_b, data1_a, data1_b] = map[lg];
            data0_a.assign(a.begin(), a.end()), data1_a.assign(a.begin(), a.end());
            data0_b.assign(b.begin(), b.end()), data1_b.assign(b.begin(), b.end());

            for (auto& val : data1_a) val = val.conj();
            for (auto& val : data1_b) val = val.conj();

            transform<false>(lg, lg2, data0_a);
            transform<false>(lg, lg2, data0_b);
            transform<false>(lg, lg2, data1_a, true);
            transform<false>(lg, lg2, data1_b, true);

            std::vector<Cum> tmp(m);
            for (int i = 0; i < n; i += m) {
                convolve_aux(lg2, std::span<Cum>(data0_a).subspan(i, m), std::span<Cum>(data0_b).subspan(i, m), tmp);
                std::copy(tmp.begin(), tmp.end(), data0_a.begin() + i);
            }
            for (int i = 0; i < n; i += m) {
                convolve_aux(lg2, std::span<Cum>(data1_a).subspan(i, m), std::span<Cum>(data1_b).subspan(i, m), tmp);
                std::copy(tmp.begin(), tmp.end(), data1_a.begin() + i);
            }

            transform<true>(lg, lg2, data0_a);

            transform<true>(lg, lg2, data1_a, true);

            for (auto& val : data1_a) val = val.conj();
            // for (auto& val : data1_b) val = val.conj();

            std::fill(out.begin(), out.end(), Cum());
            for (int i = 0; i < n; i += m) {
                Cum f = Cum(1, 2) * Cum::inv3;
                for (int j = 0; j < m; j++) {
                    // Cum A = data0_a[i + j], B = data1_a[i + j];
                    // Cum L = A * (f * x2) - B * (f * x);
                    // Cum H = A * -f + B * f;

                    Cum A = data0_a[i + j] * f, B = data1_a[i + j] * f;
                    Cum L = A.mul_w2() - B.mul_w();
                    Cum H = -A + B;

                    out[i + j] += L;

                    int ind = i + j + m;
                    if (ind < n) {
                        out[ind] += H;
                    } else {
                        out[ind - n] += H.mul_w();
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

struct cum_timer {
    clock_t beg;
    std::string s;

    cum_timer(std::string s) : s(s) {
        reset();
    }

    void reset() {
        beg = clock();
    }

    double elapsed(bool reset = false) {
        clock_t clk = clock();
        double res = (clk - beg) * 1.0 / CLOCKS_PER_SEC;
        if (reset) {
            beg = clk;
        }
        return res;
    }

    void print() {
        std::cerr << s << ": " << elapsed() << std::endl;
    }

    ~cum_timer() {
        print();
    }
};

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cout.tie(nullptr);

    int n, m;
    n = m = 1 << 19;
    // std::cin >> n >> m;
    std::vector<u64> a(n, 1), b(m, 1);
    // for (auto& i : a) {
    //     std::cin >> i;
    // }
    // for (auto& i : b) {
    //     std::cin >> i;
    // }

    std::vector<u64> c;

    {
        cum_timer cum("work");
        SSA ssa;
        c = ssa.convolve(a, b);
    }

    for (int i = 0; i < c.size(); i++) {
        // std::cout << c[i] << " \n"[i + 1 == c.size()];
    }

    return 0;
}
