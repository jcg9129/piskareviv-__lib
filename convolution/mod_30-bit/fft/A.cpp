#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;

namespace FFT {

struct complex {
    double real, imag;
    complex(double real = 0, double imag = 0) : real(real), imag(imag) { ; }
    friend complex operator+(const complex& a, const complex& b) { return complex(a.real + b.real, a.imag + b.imag); }
    friend complex operator-(const complex& a, const complex& b) { return complex(a.real - b.real, a.imag - b.imag); }
    friend complex operator*(const complex& a, const complex& b) { return complex(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real); }
    complex conj() const { return complex(real, -imag); }

    static complex polar(double ang) { return complex(std::cos(ang), std::sin(ang)); }
};

class FFT {
   private:
    std::vector<complex> w;

    void expand(int lg) {
        if (w.empty()) w = {1};
        while (0 < lg && w.size() < (1 << lg - 1)) {
            int k = __builtin_ctz(w.size());
            const double pi = std::acos(-1);
            for (int i = (1 << k); i < (1 << k + 1); i++) {
                int j = 0;
                for (int t = 0; t <= k; t++) j = j << 1 | i >> t & 1;
                w.push_back(complex::polar(j * (2 * pi / (1 << k + 2))));
            }
        }
    }

    template <bool inverse>
    static void butterfly_x2(complex& a, complex& b, complex w) {
        complex x = a, y = b;
        inverse ? (a = x + y, b = (x - y) * w) : (y = y * w, a = x + y, b = x - y);
    }

   public:
    template <bool inverse, bool fisting = false>
    void transform(int lg, complex* data) {
        expand(lg + 2 * fisting);
        // for (int k = inverse ? 0 : lg - 1; inverse ? k < lg : k >= 0; inverse ? k++ : k--) {
        //     for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
        //         complex wi = w[(1 << lg - k) * fisting + (i >> k + 1)];
        //         if (inverse) wi = wi.conj();
        //         for (int j = 0; j < (1 << k); j++) {
        //             butterfly_x2<inverse>(data[i + j], data[i + (1 << k) + j], wi);
        //         }
        //     }
        // }
        for (int i = 0; i + 2 <= (1 << lg); i += 2) {
            int h = __builtin_ctz(inverse ? i + 2 : i + (1 << lg));
            for (int k = inverse ? 0 : h - 1; inverse ? k < h : k >= 0; inverse ? k++ : k--) {
                int i0 = i - inverse * ((2 << k) - 2);
                complex wi = w[(1 << lg - k) * fisting + (i0 >> k + 1)];
                if (inverse) wi = wi.conj();
                for (int j = 0; j < (1 << k); j++) {
                    butterfly_x2<inverse>(data[i0 + j], data[i0 + (1 << k) + j], wi);
                }
            }
        }
        if (inverse) {
            for (int i = 0; i < (1 << lg); i++) data[i] = data[i] * (1.0 / (1 << lg));
        }
    }
};

FFT fft;

};  // namespace FFT

template <u32 mod>
struct Fisting {
    class mint {
        static_assert(mod < (1u << 31));

       private:
        u32 m_val;

        static u32 shrink(u32 val) { return val - mod * (val >= mod); }

       public:
        mint(int64_t val = 0) : m_val((val % mod + mod) % mod) {}

        mint& operator+=(const mint& other) { return m_val = shrink(m_val + other.m_val), *this; }
        mint& operator-=(const mint& other) { return m_val = shrink(m_val + (mod - other.m_val)), *this; }
        mint& operator*=(const mint& other) { return m_val = u64(m_val) * other.m_val % mod, *this; }

        friend mint operator+(mint a, const mint& b) { return mint(a += b); }
        friend mint operator-(mint a, const mint& b) { return mint(a -= b); }
        friend mint operator*(mint a, const mint& b) { return mint(a *= b); }

        mint power(u64 exp) const {
            mint r = 1, b = *this;
            for (; exp; exp >>= 1) {
                if (exp & 1) r *= b;
                b *= b;
            }
            return r;
        }
        mint inverse() const { return power(mod - 2); }

        friend bool operator==(const mint& a, const mint& b) { return a.m_val == b.m_val; }

        u32 get() const { return m_val; }

        friend std::istream& operator>>(std::istream& in, mint& val) {
            int64_t x;
            return in >> x, val = x, in;
        }
        friend std::ostream& operator<<(std::ostream& out, const mint& val) { return out << val.get(); }
    };

    struct poly : std::vector<mint> {
        using base = std::vector<mint>;
        using base::base;

        void remove_zeros() {
            while (this->size() && this->back() == 0) this->pop_back();
        }

        mint coeff(size_t ind) const { return ind < this->size() ? (*this)[ind] : mint(0); }
        poly& operator*=(const poly& other) { return *this = *this * other; }
        poly& operator+=(const poly& other) {
            resize(std::max(this->size(), other.size()));
            for (size_t i = 0; i < other.size(); i++) (*this)[i] += other[i];
            return *this;
        }
        poly& operator-=(const poly& other) {
            resize(std::max(this->size(), other.size()));
            for (size_t i = 0; i < other.size(); i++) (*this)[i] -= other[i];
            return *this;
        }

        friend poly operator+(poly a, const poly& b) { return a += b; }
        friend poly operator-(poly a, const poly& b) { return a -= b; }

        friend poly operator*(poly a, poly b) {
            if (a.empty() || b.empty()) {
                return {};
            }
            size_t n = a.size(), m = b.size();
            int lg = 0;
            while ((size_t(1) << lg + 1) < n + m - 1) lg++;

            if (n * m < 4 * (size_t(lg) << lg)) {
                poly c(n + m - 1);
                for (size_t i = 0; i < n; i++) {
                    for (size_t j = 0; j < m; j++) {
                        c[i + j] += a[i] * b[j];
                    }
                }
                return c;
            }

            using namespace FFT;
            std::vector<complex> al(size_t(1) << lg), ah(size_t(1) << lg), bl(size_t(1) << lg), bh(size_t(1) << lg);

            constexpr int B = [](u32 val) {
                u64 res = 0;
                while ((res + 1) * (res + 1) <= val) res++;
                return res;
            }(mod);

            static std::mt19937 rnd;
            mint f = rnd() % (mod - 1) + 1;
            for (auto [i, h] = std::pair{0, mint(1)}; i < std::max(n, m); i++, h *= f) {
                if (i < n) a[i] *= h;
                if (i < m) b[i] *= h;
            }

            auto get = [&](auto& vec, size_t ind) -> auto& {
                return ind < (size_t(1) << lg) ? vec[ind].real : vec[ind - (size_t(1) << lg)].imag;
            };
            auto func = [&](mint x) -> int {
                return int64_t(x.get()) - int64_t(mod) * (x.get() >= mod / 2);
            };
            for (size_t i = 0; i < n; i++) get(al, i) = func(a[i]) % B, get(ah, i) = func(a[i]) / B;
            for (size_t i = 0; i < m; i++) get(bl, i) = func(b[i]) % B, get(bh, i) = func(b[i]) / B;

            fft.transform<false, true>(lg, al.data());
            fft.transform<false, true>(lg, ah.data());
            fft.transform<false, true>(lg, bl.data());
            fft.transform<false, true>(lg, bh.data());

            std::vector<complex> cl(size_t(1) << lg), cm(size_t(1) << lg), ch(size_t(1) << lg);
            for (size_t i = 0; i < (1 << lg); i++) {
                cl[i] = al[i] * bl[i], cm[i] = al[i] * bh[i] + ah[i] * bl[i], ch[i] = ah[i] * bh[i];
            }

            fft.transform<true, true>(lg, cl.data());
            fft.transform<true, true>(lg, cm.data());
            fft.transform<true, true>(lg, ch.data());

            poly c(n + m - 1);
            mint fi = f.inverse();
            for (auto [i, h] = std::pair{0, mint(1)}; i < n + m - 1; i++, h *= fi) {
                mint lw = std::round(get(cl, i));
                mint md = std::round(get(cm, i));
                mint hg = std::round(get(ch, i));
                c[i] = (lw + B * (md + B * hg)) * h;
            }
            return c;
        }
    };
};

constexpr u32 mod = 1e9 + 7;

using mint = Fisting<mod>::mint;
using poly = Fisting<mod>::poly;

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    std::cin >> n >> m;
    poly a(n), b(m);
    for (auto& i : a) std::cin >> i;
    for (auto& i : b) std::cin >> i;

    poly c;
    c = a * b;
    clock_t beg = clock();
    for (int i = 0; i < 10; i++)
        c = a * b;
    std::cerr << (clock() - beg) * 1.0 / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;

    for (int i = 0; i < n + m - 1; i++) {
        std::cout << c.coeff(i) << " \n"[i + 1 == n + m - 1];
    }

    return 0;
}
