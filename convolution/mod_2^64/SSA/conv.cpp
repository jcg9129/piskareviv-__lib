#include <iostream>

#pragma GCC target("avx2")
#include <immintrin.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>

using i256 = __m256i;
using u64 = uint64_t;
using u64x4 = u64 __attribute__((vector_size(32)));

u64x4 set1_u64x4(u64 val) {
    return (u64x4)_mm256_set1_epi64x(val);
}

template <typename T>
struct CumT {
    T a, b;

    constexpr CumT(T a = T(), T b = T()) : a(a), b(b) { ; }

    constexpr CumT operator+(const CumT& other) const { return CumT(a + other.a, b + other.b); }
    constexpr CumT operator-(const CumT& other) const { return CumT(a - other.a, b - other.b); }

    constexpr CumT operator*(const CumT& other) const {
        T p = a * other.a, q = b * other.b, r = (a + b) * (other.a + other.b) - p - q;
        return CumT(p - q, r - q);
    }
    constexpr CumT mul_w() const { return CumT(-b, a - b); }
    constexpr CumT mul_w2() const { return CumT(b - a, -a); }

    constexpr friend CumT operator+(T val, const CumT& cum) { return CumT(val) + cum; }
    constexpr friend CumT operator-(T val, const CumT& cum) { return CumT(val) - cum; }
    constexpr friend CumT operator*(T val, const CumT& cum) { return CumT(val) * cum; }

    constexpr void operator+=(const CumT& other) { *this = *this + other; }
    constexpr void operator-=(const CumT& other) { *this = *this - other; }
    constexpr void operator*=(const CumT& other) { *this = *this * other; }

    constexpr CumT operator-() const { return CumT(-a, -b); }
    constexpr CumT conj() const { return CumT(a - b, -b); }

    constexpr bool operator==(const CumT& other) const { return a == other.a && b == other.b; }
    constexpr bool operator!=(const CumT& other) const { return a != other.a || b != other.b; }

    static constexpr u64 inv3 = (2 * (__uint128_t(1) << 64) + 1) / 3;
};

using Cum = CumT<u64>;
using Cum_x4 = CumT<u64x4>;

// Cum_x4 perm(Cum_x4 val, int mask) {
//     return Cum_x4((u64x4)_mm256_permute4x64_epi64((i256)val.a, mask), (u64x4)_mm256_permute4x64_epi64((i256)val.a, mask));
// }
template <int a, int b, int c, int d>
Cum_x4 perm(Cum_x4 val) {
    return Cum_x4(__builtin_shufflevector(val.a, val.a, a, b, c, d), __builtin_shufflevector(val.b, val.b, a, b, c, d));
}

template <int N>
[[gnu::always_inline]] inline void cum_mul(const Cum_x4* ptr1, const Cum_x4* ptr2, Cum_x4* out) {
    if constexpr (N == 1) {
        u64 fuck1[2 * 4], fuck2[2 * 4];
        memset(fuck1, 0, sizeof(fuck1)), memset(fuck2, 0, sizeof(fuck2));

        Cum_x4 a = ptr1[0], b = ptr2[0];
        for (int i = 0; i < 4; i++) {
            Cum_x4 c = Cum_x4(set1_u64x4(b.a[i]), set1_u64x4(b.b[i]));
            Cum_x4 d = c * a;
            // for (int j = 0; j < 4; j++) {
            //     fuck1[i + j] += d.a[j];
            //     fuck2[i + j] += d.b[j];
            // }
            *(u64x4*)(fuck1 + i) += d.a;
            *(u64x4*)(fuck2 + i) += d.b;
        }

        Cum_x4 r1 = Cum_x4(((u64x4*)(fuck1))[0], ((u64x4*)(fuck2))[0]);
        Cum_x4 r2 = Cum_x4(((u64x4*)(fuck1))[1], ((u64x4*)(fuck2))[1]);

        out[0] = r1, out[1] = r2;
    } else {
        static_assert(N % 2 == 0);
        Cum_x4 tmp1[N / 2], tmp2[N / 2];
        for (int i = 0; i < N / 2; i++) {
            tmp1[i] = ptr1[i] + ptr2[N / 2 + i];
            tmp2[i] = ptr1[N / 2 + i] + ptr2[i];
        }
        Cum_x4 out1[N], out2[N], out3[N];

        cum_mul<N / 2>(ptr1, ptr2, out1);
        cum_mul<N / 2>(ptr1 + N / 2, ptr2 + N / 2, out2);
        cum_mul<N / 2>(tmp1, tmp2, out3);

        std::fill(out, out + 2 * N, Cum_x4());
        for (int i = 0; i < N; i++) {
            out[i] += out1[i];
            out[N + i] += out2[i];
            out[N / 2 + i] += out3[i] - out1[i] - out2[i];
        }
        // for (int i = 0; i < N; i++) {
        // }
    }
}

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
        if (n == 27 && 0) {
            Cum_x4 A[8], B[8];
            for (auto [x, X] : {std::pair{a, A}, std::pair{b, B}}) {
                memset(X, 0, sizeof(Cum_x4) * 8);
                for (int i = 0; i < 27; i++) {
                    auto [q, r] = div(i, 4);
                    X[q].a[r] = a[i].a;
                    X[q].b[r] = a[i].b;
                }
            }
            Cum_x4 C[16];
            cum_mul<8>(A, B, C);

            for (int i = 0; i < 27; i++) {
                auto [q1, r1] = div(i, 4);
                auto [q2, r2] = div(i + 27, 4);
                out[i] = Cum(C[q1].a[r1], C[q1].b[r1]) + Cum(C[q2].a[r2], C[q2].b[r2]).mul_w();
            }

            return;
        }
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

    template <int n, int s>
    void mul_many_aux(int N, int S, std::span<Cum_x4> a, std::span<Cum_x4> b, std::span<Cum_x4> out) {
        assert(n == N);

        assert(a.size() == n * s);
        assert(b.size() == n * s);
        assert(out.size() == 2 * n * s);
        if constexpr (n <= 8) {
            // for (int i = 0; i < s; i++) {
            //     out[i] = a[i] * b[i];
            // }
            std::fill(out.begin(), out.end(), Cum_x4());
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    for (int t = 0; t < s; t++) {
                        out[(i + j) * s + t] += a[i * s + t] * b[j * s + t];
                    }
                }
            }
        } else {
            assert(n % 2 == 0);
            // int m = n / 2;
            int M = n / 2;
            constexpr int m = n / 2;
            mul_many_aux<n / 2, s>(m, s, a.subspan(0, s * m), b.subspan(0, s * m), out.subspan(0, s * n));
            mul_many_aux<n / 2, s>(m, s, a.subspan(s * m, s * m), b.subspan(s * m, s * m), out.subspan(s * n, s * n));
            // std::vector<Cum_x4> tmp1(s * m), tmp2(s * m);
            Cum_x4 tmp1[s * m], tmp2[s * m];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < s; j++) {
                    tmp1[i * s + j] = a[i * s + j] + b[(m + i) * s + j];
                    tmp2[i * s + j] = a[(m + i) * s + j] + b[i * s + j];
                }
            }
            // std::vector<Cum_x4> tmp3(s * n);
            Cum_x4 tmp3[s * n];
            mul_many_aux<n / 2, s>(m, s, tmp1, tmp2, tmp3);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < s; j++) {
                    tmp3[i * s + j] -= out[i * s + j] + out[(i + n) * s + j];
                }
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < s; j++) {
                    out[(i + m) * s + j] += tmp3[i * s + j];
                }
            }
        }
    }

    void mul_many(int n, int m, std::span<Cum> a, std::span<Cum> b) {
        int k = n / m;
        int s = (k + 3) / 4;

        assert(m == 27);
        int m2 = 32;
        // int m2 = (m + 7) / 8

        std::vector<Cum_x4> a_T(m2 * s), b_T(m2 * s), c_T(2 * m2 * s);
        // constexpr int N = 1000;
        // Cum_x4 a_T[N], b_T[N], c_T[2 * N];
        // assert(m * s <= N);
        // std::fill(a_T, a_T + m * s, Cum_x4());
        // std::fill(b_T, b_T + m * s, Cum_x4());
        // std::fill(c_T, c_T + 2 * m * s, Cum_x4());
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < m; j++) {
                a_T[j * s + i / 4].a[i % 4] = a[i * m + j].a;
                a_T[j * s + i / 4].b[i % 4] = a[i * m + j].b;
                b_T[j * s + i / 4].a[i % 4] = b[i * m + j].a;
                b_T[j * s + i / 4].b[i % 4] = b[i * m + j].b;
            }
        }

        // for (int i = 0; i < m; i++) {
        //     for (int j = 0; j < m; j++) {
        //         for (int t = 0; t < s; t++) {
        //             c_T[(i + j) * s + t] += a_T[i * s + t] * b_T[j * s + t];
        //         }
        //     }
        // }
        assert(m == 27 && n == 27 * 27);
        mul_many_aux<32, 28 / 4>(m2, s, a_T, b_T, c_T);

        for (int j = 0; j < m; j++) {
            for (int i = 0; i < s; i++) {
                c_T[j * s + i] += c_T[(j + m) * s + i].mul_w();
            }
        }

        for (int i = 0; i < k; i++) {
            for (int j = 0; j < m; j++) {
                a[i * m + j].a = c_T[j * s + i / 4].a[i % 4];
                a[i * m + j].b = c_T[j * s + i / 4].b[i % 4];
            }
        }
    }

    int get_next_lg(int lg) {
        int lg2 = (lg + 1) / 2;
        return lg2 >= lg - 1 ? -1 : lg2;
    }

    // mod u^3^lg - x
    void convolve_aux(int lg, std::span<Cum> a, std::span<Cum> b) {
        int n = pow_fixed<3u>(lg);
        int lg2 = get_next_lg(lg);
        if (lg2 == -1) {
            std::vector<Cum> a_copy(a.begin(), a.end());
            mul_small(n, a_copy, b, a);
            return;
        } else {
            // std::cerr << lg << "->" << lg2 << " \n";

            int m = pow_fixed<3u>(lg2);
            std::span<Cum> data0_a = a, data0_b = b;
            std::vector<Cum> data1_a(a.begin(), a.end()), data1_b(b.begin(), b.end());

            for (auto& val : data1_a) val = val.conj();
            for (auto& val : data1_b) val = val.conj();

            transform<false>(lg, lg2, data0_a);
            transform<false>(lg, lg2, data0_b);
            transform<false>(lg, lg2, data1_a, true);
            transform<false>(lg, lg2, data1_b, true);

            if (get_next_lg(lg2) == -1 && 1) {
                mul_many(n, m, data0_b, data0_a);
                mul_many(n, m, data1_b, data1_a);
            } else {
                for (int i = 0; i < n; i += m) {
                    convolve_aux(lg2, std::span<Cum>(data0_b).subspan(i, m), std::span<Cum>(data0_a).subspan(i, m));
                    convolve_aux(lg2, std::span<Cum>(data1_b).subspan(i, m), std::span<Cum>(data1_a).subspan(i, m));
                }
            }

            transform<true>(lg, lg2, data0_b);
            transform<true>(lg, lg2, data1_b, true);

            std::fill(a.begin(), a.end(), Cum());
            for (int i = 0; i < n; i += m) {
                Cum f = Cum(1, 2) * Cum::inv3;
                for (int j = 0; j < m; j++) {
                    Cum A = data0_b[i + j] * f, B = data1_b[i + j].conj() * f;
                    Cum L = A.mul_w2() - B.mul_w();
                    Cum H = -A + B;

                    a[i + j] += L;

                    int ind = i + j + m;
                    if (ind < n) {
                        a[ind] += H;
                    } else {
                        a[ind - n] += H.mul_w();
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
        std::vector<Cum> A(n), B(n);
        std::copy(a.begin(), a.end(), A.begin()), std::copy(b.begin(), b.end(), B.begin());
        convolve_aux(lg, A, B);
        std::vector<u64> res(std::max(0, (int)a.size() + (int)b.size() - 1));
        for (int i = 0; i < res.size(); i++) {
            res[i] = i < n ? A[i].a : A[i - n].b;
        }
        return res;
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
        std::cout << c[i] << " \n"[i + 1 == c.size()];
    }

    return 0;
}
