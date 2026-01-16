#include <bits/stdc++.h>

using u32 = uint32_t;
using u64 = uint64_t;

template <u32 mod>
struct MintT {
   private:
    u32 m_val;

    static u32 add(u32 a, u32 b) { return a + b - mod * (a + b >= mod); }
    static u32 mul(u32 a, u32 b) { return u64(a) * b % mod; }

   public:
    MintT() : m_val(0) { ; }
    MintT(int64_t x) : m_val((x % mod + mod) % mod) { ; }

    static MintT from_u32_unchecked(u32 val) {
        MintT m;
        m.m_val = val;
        return m;
    }

    MintT& operator+=(const MintT& other) { return m_val = add(m_val, other.m_val), *this; }
    MintT& operator-=(const MintT& other) { return m_val = add(m_val, mod - other.m_val), *this; }
    MintT& operator*=(const MintT& other) { return m_val = mul(m_val, other.m_val), *this; }

    MintT operator-() const { return MintT() - *this; }
    friend MintT operator+(MintT a, const MintT& b) { return MintT(a += b); }
    friend MintT operator-(MintT a, const MintT& b) { return MintT(a -= b); }
    friend MintT operator*(MintT a, const MintT& b) { return MintT(a *= b); }

    MintT power(u64 exp) const {
        MintT r = 1, b = *this;
        for (; exp; exp >>= 1) {
            if (exp & 1)
                r *= b;
            b *= b;
        }
        return r;
    }

    MintT inverse() const {
        assert(m_val != 0);
        return power(mod - 2);
    }
    static std::vector<MintT> bulk_inverse(const std::vector<MintT>& vec) {
        std::vector<MintT> res(vec.size(), 1);
        MintT val = 1;
        for (int i = 0; i < vec.size(); i++) {
            res[i] *= val, val *= vec[i];
        }
        val = val.inverse();
        for (int i = 0; i < vec.size(); i++) {
            res.rbegin()[i] *= val, val *= vec.rbegin()[i];
        }
        return res;
    }
    u32 get_value() const { return m_val; }

    friend bool operator!=(const MintT& a, const MintT& b) { return a.m_val != b.m_val; }
    friend bool operator==(const MintT& a, const MintT& b) { return a.m_val == b.m_val; }

    friend std::istream& operator>>(std::istream& in, MintT& x) {
        int64_t val;
        in >> val;
        x = MintT(val);
        return in;
    }
    friend std::ostream& operator<<(std::ostream& out, const MintT& x) { return out << x.get_value(); }
};

template <u32 mod>
class NTT {
   public:
    using mint = MintT<mod>;

   private:
    mint pr_root;
    std::vector<mint> wd, wrd, w_rt, wr_rt;

    static u32 find_pr_root() {
        std::vector<u32> vec;
        u32 val = mod - 1;
        for (u64 i = 2; i * i <= val; i++) {
            if (val % i == 0) {
                vec.push_back(i);
                do {
                    val /= i;
                } while (val % i == 0);
            }
        }
        if (val != 1) {
            vec.push_back(val);
        }
        for (u32 i = 2; i < mod; i++) {
            if (std::all_of(vec.begin(), vec.end(), [&](u32 q) { return mint(i).power((mod - 1) / q) != 1; })) {
                return i;
            }
        }
        assert(false && "pr_root not found");
    }

   public:
    NTT() : pr_root(find_pr_root()) {
        int lg = __builtin_ctz(mod - 1);
        wd.assign(lg, 0), wrd.assign(lg, 0);
        w_rt.assign(lg - 1, 0), wr_rt.assign(lg - 1, 0);
        for (int k = 0; k + 2 <= lg; k++) {
            mint a = pr_root.power(mod - 1 >> k + 2);
            mint b = pr_root.power((mod - 1 >> k + 2) * ((2 << k) - 2));
            w_rt[k] = a, wr_rt[k] = a.inverse();
            wd[k] = a * b.inverse(), wrd[k] = a.inverse() * b;
        }
    }

   private:
    template <bool inverse>
    static void butterfly_x2(mint& a, mint& b, mint w) {
        mint x = a, y = b;
        if (!inverse) {
            y *= w, a = x + y, b = x - y;
        } else {
            a = x + y, b = (x - y) * w;
        }
    }

   public:
    template <bool inverse, bool right_part = false>
    void transform(int lg, mint* data) const {
        for (int k = inverse ? 0 : lg - 1; inverse ? k < lg : k >= 0; inverse ? k++ : k--) {
            mint wi = right_part ? (inverse ? wr_rt : w_rt)[lg - k - 1] : mint(1);
            for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                for (int j = 0; j < (1 << k); j++) {
                    butterfly_x2<inverse>(data[i + j], data[i + (1 << k) + j], wi);
                }
                wi *= (inverse ? wrd : wd)[__builtin_ctz(~i >> k + 1)];
            }
        }
        if (inverse) {
            mint inv = mint(mod + 1 >> 1).power(lg);
            for (int i = 0; i < (1 << lg); i++) {
                data[i] *= inv;
            }
        }
    }

    void expand_ntt(int lg, mint* data) const {
        std::copy(data, data + (1 << lg), data + (1 << lg));
        transform<true>(lg, data + (1 << lg));
        transform<false, true>(lg, data + (1 << lg));
    }

    void extract_cum(int lg, mint* data, bool odd = false) const {
        const mint inv2 = mint(mod + 1 >> 1);
        if (!odd) {
            for (int i = 0; i < (1 << lg); i++) {
                data[i] = (data[2 * i] + data[2 * i + 1]) * inv2;
            }
        } else {
            mint wi = inv2;
            for (int i = 0; i < (1 << lg); i++) {
                data[i] = (data[2 * i] - data[2 * i + 1]) * wi;
                wi *= wrd[__builtin_ctz(~i)];
            }
        }
    }

    void convolve_cyclic(int lg, mint* a, mint* b) const {
        transform<false>(lg, a);
        transform<false>(lg, b);
        for (int i = 0; i < (1 << lg); i++) {
            a[i] *= b[i];
        }
        transform<true>(lg, a);
    }

    std::vector<mint> convolve(std::vector<mint> a, std::vector<mint> b) const {
        if (a.empty() || b.empty()) {
            return {};
        }
        int n = a.size(), m = b.size(), lg = (n == 1 && m == 1) ? 0 : 32 - __builtin_clz(n + m - 2);
        if (a.size() <= 2 || b.size() <= 2 || a.size() * b.size() <= int64_t(1 << lg) * lg * 2) {
            std::vector<mint> c(n + m - 1);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    c[i + j] += a[i] * b[j];
                }
            }
            return c;
        }
        if (lg > 0 && n + m - 1 == (1 << lg - 1) + 1) {
            mint p = a.back() * b.back();
            a.reserve((1 << lg - 1) + 1);
            a.resize(1 << lg - 1), b.resize(1 << lg - 1);
            convolve_cyclic(lg - 1, a.data(), b.data());
            a[0] -= p, a.push_back(p);
            return a;
        }
        a.resize(1 << lg), b.resize(1 << lg);
        convolve_cyclic(lg, a.data(), b.data());
        a.resize(n + m - 1);
        return a;
    }
};
