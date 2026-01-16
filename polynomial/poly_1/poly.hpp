#include <bits/stdc++.h>

// #include "ntt.hpp"
#include "ntt_simd.hpp"

using u32 = uint32_t;
using u64 = uint64_t;

namespace polynomial {
    template <u32 mod>
    class Poly : public std::vector<MintT<mod>> {
       public:
        using base = std::vector<MintT<mod>>;
        using base::base, base::size, base::resize;
        using mint = MintT<mod>;

       private:
        static const NTT<mod> ntt;

       public:
        mint coeff(size_t ind) const {
            return ind < this->size() ? this->operator[](ind) : mint();
        }

        int64_t deg() const {
            for (int64_t i = size() - 1; i >= 0; i--) {
                if (this->operator[](i) != 0) {
                    return i;
                }
            }
            return -1;
        }

        friend std::ostream& operator<<(std::ostream& out, const Poly& p) {
            out << "{";
            for (int i = 0; i < p.size(); i++) {
                if (i != 0) {
                    out << ", ";
                }
                out << p[i];
            }
            out << "}";
            return out;
        }

        void remove_zeros() {
            while (size() && this->back() == 0) {
                this->pop_back();
            }
        }

        friend Poly operator*(const Poly& a, const Poly& b) {
            int64_t n = a.deg(), m = b.deg();
            if (n == -1 || m == -1) {
                return {};
            }
            auto p = ntt.convolve(std::vector<mint>(a.begin(), a.begin() + n + 1), std::vector<mint>(b.begin(), b.begin() + m + 1));
            Poly c(p.begin(), p.end());
            c.remove_zeros();
            return c;
        }
        Poly& operator*=(const Poly& other) { return *this = *this * other; }

        Poly operator-() const {
            Poly a = *this;
            for (int i = 0; i < a.size(); i++) {
                a[i] = -a[i];
            }
            a.remove_zeros();
            return a;
        }

        Poly& operator+=(const Poly& b) {
            resize(std::max(size(), b.size()));
            for (int i = 0; i < b.size(); i++) {
                this->operator[](i) += b[i];
            }
            remove_zeros();
            return *this;
        }
        Poly& operator-=(const Poly& b) {
            resize(std::max(size(), b.size()));
            for (int i = 0; i < b.size(); i++) {
                this->operator[](i) -= b[i];
            }
            remove_zeros();
            return *this;
        }
        friend Poly operator+(Poly a, Poly b) { return a += b; }
        friend Poly operator-(Poly a, Poly b) { return a -= b; }

        // sub  x = ax
        Poly sub_ax(mint a) const {
            mint p = 1;
            Poly res = *this;
            for (int i = 0; i < size(); i++, p *= a) {
                res[i] *= p;
            }
            return res;
        }

        Poly div_xk(size_t k) const {
            return Poly(this->begin() + std::min(size(), k), this->end());
        }

        Poly mul_xk(size_t k) const {
            Poly a = *this;
            a.insert(a.begin(), k, 0);
            return a;
        }

        Poly mod_xk(size_t k) const {
            return Poly(this->begin(), this->begin() + std::min(size(), k));
        }

        Poly inv_series(int n) const {
            Poly a = *this;
            a.resize(n);
            Poly b = {a.coeff(0).inverse()};
            for (int k = 0; (1 << k) < n; k++) {
                int m = 1 << k;
                Poly c = a.mod_xk(2 * m);
                b.resize(4 * m);
                c.resize(4 * m);
                ntt.template transform<false>(k + 2, b.data());
                ntt.template transform<false>(k + 2, c.data());
                for (int i = 0; i < (4 * m); i++) {
                    b[i] *= mint(2) - b[i] * c[i];
                }
                ntt.template transform<true>(k + 2, b.data());
                b.resize(2 * m);
            }
            b.resize(n);
            return b;
        }

        Poly div(Poly b, Poly b_inv = {}) const {
            Poly a = *this;
            a.remove_zeros(), b.remove_zeros();
            assert(b.size());
            if (a.size() < b.size()) {
                return {{}, {}};
            }
            std::reverse(a.begin(), a.end()), std::reverse(b.begin(), b.end());
            size_t d = a.size() - b.size() + 1;

            if (b_inv.size() < d) {
                b_inv = b.inv_series(d);
            }

            Poly q = (a.mod_xk(d) * b_inv.mod_xk(d)).mod_xk(d);
            q.resize(d);
            std::reverse(q.begin(), q.end());
            return q;
        }
        std::pair<Poly, Poly> divmod(Poly b, const Poly& b_inv = {}) const {
            Poly q = this->div(b, b_inv);
            Poly r = *this - q * b;
            assert(r.size() < b.size());
            r.remove_zeros();
            return {q, r};
        }
        friend Poly operator/(Poly a, Poly b) { return a.div(b); }
        friend Poly operator%(const Poly& a, const Poly& b) { return a.divmod(b).second; }

        Poly power(u64 exp) const {
            if (exp == 0) {
                return Poly{1};
            } else if (exp & 1) {
                return power(exp - 1) * *this;
            } else {
                Poly a = power(exp >> 1);
                return a * a;
            }
        }
        Poly power_mod(u64 exp, const Poly& md, std::shared_ptr<Poly> md_inv = nullptr) const {
            if (exp == 0) {
                return Poly{1};
            }
            if (md_inv == nullptr || md_inv->size() < md.size()) {
                md_inv = std::make_shared<Poly>(Poly(md.rbegin(), md.rend()).inv_series(md.size()));
            }
            if (exp & 1) {
                return (power_mod(exp - 1, md, md_inv) * *this).divmod(md, *md_inv).second;
            } else {
                Poly a = power_mod(exp >> 1, md, md_inv);
                return (a * a).divmod(md, *md_inv).second;
            }
        }

        mint dot(const Poly& b) const {
            mint res = 0;
            for (size_t i = 0; i < std::min(size(), b.size()); i++) {
                res += this->operator[](i) * b[i];
            }
            return res;
        }

        Poly deriv() const {
            Poly res = *this;
            for (int i = 0; i < size(); i++) {
                res[i] *= i;
            }
            if (res.size()) {
                res.erase(res.begin());
            }
            res.remove_zeros();
            return res;
        }

        Poly integ() const {
            Poly res = *this;
            res.remove_zeros();
            mint val = 1;
            for (int i = 0; i < res.size(); i++) {
                res[i] *= val, val *= (i + 1);
            }
            val = val.inverse();
            for (int i = (int)res.size() - 1; i >= 0; i--) {
                res[i] *= val, val *= (i + 1);
            }
            res.insert(res.begin(), 0);
            res.remove_zeros();
            return res;
        }

        Poly ln(int n) const {
            if (n <= 1) {
                return Poly(n);
            }
            return (mod_xk(n).deriv() * inv_series(n - 1)).mod_xk(n - 1).integ();
        }

        Poly exp(int n) const {
            assert(coeff(0) == 0);
            Poly b = {1};
            for (int k = 0; (1 << k) < n; k++) {
                int m = 1 << k;
                // b = (b * (Poly{1} - b.ln(2 * m) + this->mod_xk(2 * m))).mod_xk(2 * m);
                Poly b2 = b;
                Poly c = Poly{1} - b.ln(2 * m) + this->mod_xk(2 * m);
                b2.resize(2 * m), c.resize(2 * m), b.resize(2 * m);
                ntt.convolve_cyclic(k + 1, b2.data(), c.data());
                for (int i = m; i < 2 * m; i++) {
                    b[i] = b2[i];
                }
            }
            b.resize(n);
            return b;
        }

        std::vector<mint> evaluate(const std::vector<mint>& pts) {
            if (pts.empty()) {
                return {};
            }
            int sz = 1;
            while (sz < pts.size()) sz *= 2;

            std::vector<Poly> data(2 * sz);
            for (int i = 0; i < pts.size(); i++) {
                data[sz + i] = Poly({-pts[i], 1});
            }
            for (int i = pts.size(); i < sz; i++) {
                data[sz + i] = Poly({1});
            }
            for (int i = sz - 1; i > 0; i--) {
                data[i] = data[2 * i] * data[2 * i + 1];
            }

            data[1] = *this % data[1];
            for (int i = 2; i < 2 * sz; i++) {
                data[i] = data[i >> 1] % data[i];
            }
            std::vector<mint> res(pts.size());
            for (int i = 0; i < pts.size(); i++) {
                res[i] = data[sz + i].coeff(0);
            }
            return res;
        }
    };
    template <u32 mod>
    const NTT<mod> Poly<mod>::ntt;

    template <u32 mod>
    Poly<mod> interpolate(const std::vector<MintT<mod>>& pts, const std::vector<MintT<mod>>& vals) {
        assert(pts.size() == vals.size());
        if (pts.empty()) {
            return Poly<mod>{};
        }

        int sz = 1;
        while (sz < pts.size()) sz *= 2;

        std::vector<Poly<mod>> data(2 * sz);
        for (int i = 0; i < pts.size(); i++) {
            data[sz + i] = Poly<mod>({-pts[i], 1});
        }
        for (int i = pts.size(); i < sz; i++) {
            data[sz + i] = Poly<mod>({1});
        }
        for (int i = sz - 1; i > 0; i--) {
            data[i] = data[2 * i] * data[2 * i + 1];
        }

        std::vector<MintT<mod>> d = data[1].deriv().evaluate(pts);
        d = MintT<mod>::bulk_inverse(d);

        auto rec = [&](auto rec, int i) -> Poly<mod> {
            if (i >= sz) {
                if (i - sz < vals.size()) {
                    return Poly<mod>{vals[i - sz] * d[i - sz]};
                } else {
                    return Poly<mod>{};
                }
            }
            Poly<mod> a = rec(rec, 2 * i);
            Poly<mod> b = rec(rec, 2 * i + 1);
            return a * data[2 * i + 1] + b * data[2 * i];
        };
        return rec(rec, 1);
    }

    // https://arxiv.org/abs/2008.08822
    template <u32 mod>
    MintT<mod> bostan_mori(u64 k, Poly<mod> p, Poly<mod> q) {
        assert(q.coeff(0) != 0);

        using mint = MintT<mod>;
        using poly = Poly<mod>;

        q.remove_zeros(), p.remove_zeros();
        int64_t n = q.deg();
        int lg = 1;
        while ((1 << lg) <= 2 * n) {
            lg++;
        }

        static const NTT<mod> ntt;

        q.resize(1 << lg);
        p.resize(1 << lg);
        poly r(1 << lg), t(1 << lg);

        if (n < k) {
            ntt.template transform<false>(lg, q.data());
            ntt.template transform<false>(lg, p.data());

            while (n < k) {
                if (lg < 3) {
                    for (int i = 0; i < (1 << lg); i += 2) {
                        mint a = p[i], b = p[i + 1], c = q[i], d = q[i + 1];
                        p[i] = a * d, p[i + 1] = b * c, q[i >> 1] = c * d;
                    }
                } else {
                    using mint_x8 = MintT_x8<mod>;
                    for (int i = 0; i < (1 << lg); i += 8) {
                        mint_x8 pi = mint_x8::load(&p[i]), qi = mint_x8::load(&q[i]);
                        mint_x8 qi_sw = qi.template shuffle<0b10'11'00'01>();
                        (pi * qi_sw).store(&p[i]), (qi * qi_sw).permute(_mm256_setr_epi32(0, 2, 4, 6, -1, -1, -1, -1)).store(&q[i >> 1]);
                    }
                }
                ntt.extract_cum(lg - 1, p.data(), k & 1);

                k >>= 1;
                if (n < k) {
                    ntt.expand_ntt(lg - 1, q.data());
                    ntt.expand_ntt(lg - 1, p.data());
                }
            }

            ntt.template transform<true>(lg - 1, q.data());
            ntt.template transform<true>(lg - 1, p.data());

            p.resize(k + 1), q.resize(k + 1);
            p.remove_zeros(), q.remove_zeros();
        }

        return (p * q.inv_series(k + 1)).coeff(k);
    }

    template <u32 mod>
    MintT<mod> kth_linear(u64 k, const Poly<mod>& gen, const Poly<mod>& ch) {
        // Poly<mod> r = Poly<mod>({0, 1}).power_mod(k, Poly<mod>(ch.rbegin(), ch.rend()));
        // return gen.dot(r);

        int64_t d = ch.deg();
        return bostan_mori(k, (gen * ch).mod_xk(d), ch);
    }
};  // namespace polynomial
