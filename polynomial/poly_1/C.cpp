#include <bits/stdc++.h>
#pragma GCC target("avx2")
#include <immintrin.h>

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
        std::cerr << s << ": " << elapsed() * 1000 << "ms" << std::endl;
    }

    ~cum_timer() {
        print();
    }
};

size_t ntt_sum_size = 0;

using u32 = uint32_t;
using u64 = uint64_t;
using i256 = __m256i;
using u32x8 = u32 __attribute__((vector_size(32)));
using u64x4 = u32 __attribute__((vector_size(32)));

template <u32 mod>
class MintT_x8;

template <u32 mod>
class MintT {
   private:
    static constexpr u32 compute_n_inv() {
        u32 res = 1;
        for (int i = 0; i < 5; i++) {
            res *= u32(2) + res * mod;
        }
        return res;
    }

    static constexpr u32 n_inv = compute_n_inv();
    static constexpr u32 mod2 = 2 * mod;
    static constexpr u32 r = (u64(1) << 32) % mod;
    static constexpr u32 r2 = (u64(r) * r) % mod;

    static_assert(mod % 2 == 1);
    static_assert(mod < (1 << 31));
    static_assert(-n_inv * mod == 1);

    u32 m_val;

    constexpr static u32 shrink(u32 val) { return val - mod * (val >= mod); }
    constexpr static u32 add(u32 a, u32 b) { return shrink(a + b); }
    constexpr static u64 reduce(u64 val) { return val + u32(val) * n_inv * u64(mod) >> 32; }
    constexpr static u32 mul(u32 a, u32 b) { return shrink(reduce(u64(a) * b)); }

   public:
    constexpr MintT() : m_val(0) { ; }
    constexpr MintT(int64_t x) : m_val(mul(x % int32_t(mod) + mod, r2)) { ; }

    constexpr MintT& operator+=(const MintT& other) { return m_val = add(m_val, other.m_val), *this; }
    constexpr MintT& operator-=(const MintT& other) { return m_val = add(m_val, mod - other.m_val), *this; }
    constexpr MintT& operator*=(const MintT& other) { return m_val = mul(m_val, other.m_val), *this; }

    constexpr friend MintT operator+(MintT a, const MintT& b) { return MintT(a += b); }
    constexpr friend MintT operator-(MintT a, const MintT& b) { return MintT(a -= b); }
    constexpr friend MintT operator*(MintT a, const MintT& b) { return MintT(a *= b); }
    constexpr MintT operator-() const { return MintT() - *this; }

    constexpr MintT power(u64 exp) const {
        MintT r = 1, b = *this;
        for (; exp; exp >>= 1) {
            if (exp & 1)
                r *= b;
            b *= b;
        }
        return r;
    }
    constexpr MintT inverse() const { return power(mod - 2); }
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

    constexpr u32 get_value() const { return shrink(reduce(m_val)); }

    constexpr friend bool operator!=(const MintT& a, const MintT& b) { return a.m_val != b.m_val; }
    constexpr friend bool operator==(const MintT& a, const MintT& b) { return a.m_val == b.m_val; }

    friend std::istream& operator>>(std::istream& in, MintT& x) {
        int64_t val;
        in >> val;
        x = MintT(val);
        return in;
    }
    friend std::ostream& operator<<(std::ostream& out, const MintT& x) { return out << x.get_value(); }

    friend class MintT_x8<mod>;
};

template <u32 mod>
class __attribute__((may_alias)) MintT_x8 {
    // class MintT_x8 {
   private:
    constexpr static MintT<mod> aux = 0;

    static constexpr i256 set1(u32 val) { return (i256)(u32x8{val, val, val, val, val, val, val, val}); }
    static constexpr i256 mod_x8 = set1(mod);
    static constexpr i256 ninv_x8 = set1(aux.n_inv);

    static i256 shrink(i256 val) {
        return _mm256_min_epu32(val, _mm256_sub_epi32(val, mod_x8));
    }

    static i256 reduce(i256 x0246, i256 x1357) {
        i256 x0246_res = _mm256_add_epi64(x0246, _mm256_mul_epu32(_mm256_mul_epu32(x0246, ninv_x8), mod_x8));
        i256 x1357_res = _mm256_add_epi64(x1357, _mm256_mul_epu32(_mm256_mul_epu32(x1357, ninv_x8), mod_x8));
        return shrink(_mm256_or_si256(_mm256_bsrli_epi128(x0246_res, 4), x1357_res));
    }
    static i256 reduce(i256 val) { return reduce(val, _mm256_bsrli_epi128(val, 4)); }

    i256 m_val;

    MintT_x8(i256 val) : m_val(val) { ; }

   public:
    MintT_x8() : m_val(_mm256_setzero_si256()) { ; }
    MintT_x8(MintT<mod> val) : m_val(_mm256_set1_epi32(val.m_val)) { ; }
    MintT_x8(MintT<mod> a0, MintT<mod> a1, MintT<mod> a2, MintT<mod> a3, MintT<mod> a4, MintT<mod> a5, MintT<mod> a6, MintT<mod> a7) : m_val(_mm256_setr_epi32(a0.m_val, a1.m_val, a2.m_val, a3.m_val, a4.m_val, a5.m_val, a6.m_val, a7.m_val)) { ; }
    MintT_x8(int64_t val) : m_val(_mm256_set1_epi32(MintT<mod>(val).m_val)) { ; }

    MintT_x8 operator+=(const MintT_x8& other) { return m_val = shrink(_mm256_add_epi32(m_val, other.m_val)), *this; }
    MintT_x8 operator-=(const MintT_x8& other) { return m_val = shrink(_mm256_add_epi32(m_val, _mm256_sub_epi32(mod_x8, other.m_val))), *this; }
    MintT_x8 operator*=(const MintT_x8& other) { return m_val = reduce(_mm256_mul_epu32(m_val, other.m_val), _mm256_mul_epu32(_mm256_bsrli_epi128(m_val, 4), _mm256_bsrli_epi128(other.m_val, 4))), *this; }

    constexpr friend MintT_x8 operator+(MintT_x8 a, const MintT_x8& b) { return MintT_x8(a += b); }
    constexpr friend MintT_x8 operator-(MintT_x8 a, const MintT_x8& b) { return MintT_x8(a -= b); }
    constexpr friend MintT_x8 operator*(MintT_x8 a, const MintT_x8& b) { return MintT_x8(a *= b); }
    MintT_x8 operator-() const { return MintT_x8() - *this; }

    constexpr MintT_x8 power(u64 exp) const {
        MintT_x8 r = 1, b = *this;
        for (; exp; exp >>= 1) {
            if (exp & 1)
                r *= b;
            b *= b;
        }
        return r;
    }
    constexpr MintT_x8 inverse() const { return power(mod - 2); }
    constexpr MintT_x8 div_by_2() const {
        i256 mask = _mm256_cmpeq_epi32(_mm256_and_si256(m_val, _mm256_set1_epi32(1)), _mm256_set1_epi32(1));
        return MintT_x8(_mm256_srli_epi32(_mm256_add_epi32(m_val, _mm256_and_si256(mask, mod_x8)), 1));
    }

    constexpr i256 get_value() const { return shrink(reduce(m_val)); }

    template <bool aligned = false>
    static MintT_x8 load(const MintT<mod>* ptr) { return MintT_x8(aligned ? _mm256_load_si256((const i256*)ptr) : _mm256_lddqu_si256((const i256*)ptr)); }
    template <bool aligned = false>
    void store(MintT<mod>* ptr) const { aligned ? _mm256_store_si256((i256*)ptr, m_val) : _mm256_storeu_si256((i256*)ptr, m_val); }

    template <int mask>
    MintT_x8 blend(const MintT_x8& other) const { return MintT_x8(_mm256_blend_epi32(m_val, other.m_val, mask)); }
    template <int mask>
    MintT_x8 shuffle() const { return MintT_x8(_mm256_shuffle_epi32(m_val, mask)); }
    MintT_x8 permute(i256 perm) const { return MintT_x8(_mm256_permutevar8x32_epi32(m_val, perm)); }

    MintT<mod>& operator[](size_t ind) { return ((MintT<mod>*)&m_val)[ind]; }
    const MintT<mod>& operator[](size_t ind) const { return ((const MintT<mod>*)&m_val)[ind]; }
};

template <u32 mod>
class NTT {
   public:
    using mint = MintT<mod>;
    using mint_x8 = MintT_x8<mod>;

   private:
    mint pr_root;
    std::vector<mint> wd, wrd, w_rt, wr_rt;
    std::vector<mint> wd_3, wrd_3;
    mint_x8 w_x8, wr_x8;

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
            mint a = pr_root.power(mod - 1 >> k + 2), b = pr_root.power((mod - 1 >> k + 2) * ((2 << k) - 2));
            w_rt[k] = a, wr_rt[k] = a.inverse();
            wd[k] = a * b.inverse(), wrd[k] = a.inverse() * b;
        }

        w_x8 = 0, wr_x8 = 0;
        if (lg >= 4) {
            mint w = pr_root.power(mod - 1 >> 4);
            for (int i = 0; i < 8; i++)
                w_x8[i] = w.power(std::array<int, 8>{0, 4, 2, 6, 1, 5, 3, 7}[i]);
            wr_x8 = w_x8.inverse();

            wd_3.assign(lg - 3, 0), wrd_3.assign(lg - 3, 0);
            for (int k = 0; k + 5 <= lg; k++) {
                mint a = pr_root.power(mod - 1 >> k + 5), b = pr_root.power((mod - 1 >> k + 5) * ((2 << k) - 2));
                wd_3[k] = a * b.inverse(), wrd_3[k] = a.inverse() * b;
            }
        }
    }

   private:
    template <bool inverse>
    static void butterfly_x2(mint& a, mint& b, mint w) {
        mint x = a, y = b;
        inverse ? (a = x + y, b = (x - y) * w) : (y *= w, a = x + y, b = x - y);
    }

    template <bool inverse>
    static void butterfly_x2(mint* a, mint* b, mint_x8 w) {
        mint_x8 x = mint_x8::load(a), y = mint_x8::load(b);
        inverse ? ((x + y).store(a), ((x - y) * w).store(b)) : (y *= w, (x + y).store(a), (x - y).store(b));
    }

    template <bool inverse, bool right_part, int k>
    // [[gnu::noinline]]
    void transform_lower_level(int lg, mint* data) const {
        static_assert(0 <= k && k <= 2);
        if constexpr (k == 2) {
            assert(lg >= 6);
            mint_x8 wi_x8 = (inverse ? wr_x8 : w_x8) * (right_part ? (inverse ? wr_rt : w_rt)[lg - k - 1] : mint(1));
            for (int i = 0; i < (1 << lg); i += 64) {
                for (int j = 0; j < 64; j += 8) {
                    mint_x8 wi_aux = mint_x8(wi_x8[j >> 3]).template blend<0x0F>(mint_x8(1)), a = mint_x8::load(&data[i + j]);
                    if constexpr (!inverse) {
                        a *= wi_aux, (a.permute(_mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3)) + a.template blend<0xF0>(-a)).store(&data[i + j]);
                    } else {
                        ((a.permute(_mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3)) + a.template blend<0xF0>(-a)) * wi_aux).store(&data[i + j]);
                    }
                }
                wi_x8 *= mint_x8((inverse ? wrd_3 : wd_3)[__builtin_ctz(~i >> 6)]);
            }
        } else if constexpr (k == 1) {
            assert(lg >= 5);
            mint_x8 wi_x8 = (inverse ? wr_x8 : w_x8) * (right_part ? (inverse ? wr_rt : w_rt)[lg - k - 1] : mint(1));
            for (int i = 0; i < (1 << lg); i += 32) {
                const i256 ar[4] = {_mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1), _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3), _mm256_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5), _mm256_setr_epi32(6, 6, 6, 6, 7, 7, 7, 7)};
                for (int j = 0; j < 32; j += 8) {
                    mint_x8 wi_aux = wi_x8.permute(ar[j >> 3]).template blend<0x33>(mint_x8(1)), a = mint_x8::load(&data[i + j]);
                    if constexpr (!inverse) {
                        a *= wi_aux, (a.template shuffle<0b01'00'11'10>() + a.template blend<0xCC>(-a)).store(&data[i + j]);
                    } else {
                        ((a.template shuffle<0b01'00'11'10>() + a.template blend<0xCC>(-a)) * wi_aux).store(&data[i + j]);
                    }
                }
                wi_x8 *= mint_x8((inverse ? wrd_3 : wd_3)[__builtin_ctz(~i >> 5)]);
            }
        } else if constexpr (k == 0) {
            assert(lg >= 4);
            mint_x8 wi_x8 = (inverse ? wr_x8 : w_x8) * (right_part ? (inverse ? wr_rt : w_rt)[lg - k - 1] : mint(1));
            for (int i = 0; i < (1 << lg); i += 16) {
                const i256 ar[2] = {_mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3), _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7)};
                for (int j = 0; j < 16; j += 8) {
                    mint_x8 wi_aux = wi_x8.permute(ar[j >> 3]).template blend<0x55>(mint_x8(1)), a = mint_x8::load(&data[i + j]);
                    if constexpr (!inverse) {
                        a *= wi_aux, (a.template shuffle<0b10'11'00'01>() + a.template blend<0xAA>(-a)).store(&data[i + j]);
                    } else {
                        ((a.template shuffle<0b10'11'00'01>() + a.template blend<0xAA>(-a)) * wi_aux).store(&data[i + j]);
                    }
                }
                wi_x8 *= mint_x8((inverse ? wrd_3 : wd_3)[__builtin_ctz(~i >> 4)]);
            }
        } else {
            static_assert(false);
        }
    }

    template <bool inverse, bool right_part>
    void transform_layer(int lg, int k, mint* data) const {
        if (k >= 3) {
            mint wi = right_part ? (inverse ? wr_rt : w_rt)[lg - k - 1] : mint(1);
            for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                mint_x8 wi_x8 = wi;
                for (int j = 0; j < (1 << k); j += 8) {
                    butterfly_x2<inverse>(&data[i + j], &data[i + (1 << k) + j], wi_x8);
                }
                wi *= (inverse ? wrd : wd)[__builtin_ctz(~i >> k + 1)];
            }
        } else if (k == 2 && lg >= 6) {
            transform_lower_level<inverse, right_part, 2>(lg, data);
        } else if (k == 1 && lg >= 5) {
            transform_lower_level<inverse, right_part, 1>(lg, data);
        } else if (k == 0 && lg >= 4) {
            transform_lower_level<inverse, right_part, 0>(lg, data);
        } else {
            mint wi = right_part ? (inverse ? wr_rt : w_rt)[lg - k - 1] : mint(1);
            for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                for (int j = 0; j < (1 << k); j++) {
                    butterfly_x2<inverse>(data[i + j], data[i + (1 << k) + j], wi);
                }
                wi *= (inverse ? wrd : wd)[__builtin_ctz(~i >> k + 1)];
            }
        }
    }

   public:
    static void dot_inplace(int n, const mint* a, const mint* b, mint* out) {
        int i = 0;
        for (; i + 8 <= n; i += 8) (mint_x8::load(a + i) * mint_x8::load(b + i)).store(out + i);
        for (; i < n; i++) out[i] = a[i] * b[i];
    }
    static void dot_inplace(int n, const mint* a, mint b, mint* out) {
        int i = 0;
        mint_x8 b_x8 = b;
        for (; i + 8 <= n; i += 8) (mint_x8::load(a + i) * b).store(out + i);
        for (; i < n; i++) out[i] = a[i] * b;
    }

   public:
    template <bool inverse, bool right_part = false>
    void transform(int lg, mint* data) const {
        ntt_sum_size += (1 << lg);

        for (int k = inverse ? 0 : lg - 1; inverse ? k < lg : k >= 0; inverse ? k++ : k--) {
            transform_layer<inverse, right_part>(lg, k, data);
        }
        if (inverse) {
            dot_inplace(1 << lg, data, mint(mod + 1 >> 1).power(lg), data);
        }
    }

    void expand_ntt(int lg, mint* data) const {
        std::copy(data, data + (1 << lg), data + (1 << lg));
        transform<true>(lg, data + (1 << lg));
        transform<false, true>(lg, data + (1 << lg));
    }

    void extract_cum(int lg, mint* data, bool odd = false) const {
        const mint inv2 = mint(mod + 1 >> 1);
        if (lg < 3) {
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
        } else {
            if (!odd) {
                for (int i = 0; i < (1 << lg); i += 8) {
                    mint_x8 a = mint_x8::load(&data[2 * i]), b = mint_x8::load(&data[2 * i + 8]);
                    mint_x8 c = a.template blend<0xAA>(b) + (a.template shuffle<0b10'11'00'01>()).template blend<0xAA>(b.template shuffle<0b10'11'00'01>());
                    c.permute(_mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)).div_by_2().store(&data[i]);
                }
            } else {
                mint_x8 wi_x8 = wr_x8 * inv2;
                for (int i = 0; i < (1 << lg); i += 8) {
                    mint_x8 a = mint_x8::load(&data[2 * i]), b = mint_x8::load(&data[2 * i + 8]);
                    mint_x8 c = a.template blend<0xAA>(b.template shuffle<0b10'11'00'01>()) - (a.template shuffle<0b10'11'00'01>()).template blend<0xAA>(b);
                    (c.permute(_mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)) * wi_x8).store(&data[i]);
                    wi_x8 *= mint_x8(wrd_3[__builtin_ctz(~i >> 3)]);
                }
            }
        }
    }

    void convolve_cyclic(int lg, mint* a, mint* b) const {
        transform<false>(lg, a);
        transform<false>(lg, b);
        dot_inplace(1 << lg, a, b, a);
        transform<true>(lg, a);
    }

    std::vector<mint> convolve(std::vector<mint> a, std::vector<mint> b) const {
        if (a.empty() || b.empty()) {
            return {};
        }
        int n = a.size(), m = b.size(), lg = (n == 1 && m == 1) ? 0 : 32 - __builtin_clz(n + m - 2);
        if (a.size() <= 2 || b.size() <= 2 || a.size() * b.size() <= int64_t(1 << lg) * lg) {
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

using u32 = uint32_t;
using u64 = uint64_t;

namespace polynomial {
    template <u32 mod>
    class Poly : public std::vector<MintT<mod>> {
       public:
        using base = std::vector<MintT<mod>>;
        using base::base, base::size, base::resize;
        using mint = MintT<mod>;
        using mint_x8 = MintT_x8<mod>;

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

        // std::vector<mint> evaluate(const std::vector<mint>& pts) {
        //     if (pts.empty()) {
        //         return {};
        //     }
        //     int sz = 1;
        //     while (sz < pts.size()) sz *= 2;

        //     std::vector<Poly> data(2 * sz);
        //     for (int i = 0; i < pts.size(); i++) {
        //         data[sz + i] = Poly({-pts[i], 1});
        //     }
        //     for (int i = pts.size(); i < sz; i++) {
        //         data[sz + i] = Poly({1});
        //     }
        //     for (int i = sz - 1; i > 0; i--) {
        //         data[i] = data[2 * i] * data[2 * i + 1];
        //     }

        //     data[1] = *this % data[1];
        //     for (int i = 2; i < 2 * sz; i++) {
        //         data[i] = data[i >> 1] % data[i];
        //     }
        //     std::vector<mint> res(pts.size());
        //     for (int i = 0; i < pts.size(); i++) {
        //         res[i] = data[sz + i].coeff(0);
        //     }
        //     return res;
        // }

        std::vector<mint> evaluate(std::vector<mint> points) const {
            Poly poly = *this;
            int res_sz = points.size();
            int n = std::max(poly.size(), points.size());
            int lg = std::__lg(std::max<int>(n - 1, 1)) + 1;  // * doesn't work for lg = 0
            // lg = std::max(lg, 3);

            poly.resize(1 << lg), points.resize(1 << lg);
            std::vector<Poly> data(lg + 1, Poly(1 << lg + 1));
            for (int i = 0; i < (1 << lg); i++) {
                data[0][2 * i] = 1 - points[i];
                data[0][2 * i + 1] = -1 - points[i];
            }
            for (int k = 0; k < lg; k++) {
                cum_timer tm(std::string("up  k = ") + std::to_string(k));
                for (int i = 0; i < (1 << lg + 1); i += 1 << k + 2) {
                    // for (int j = 0; j < (1 << k + 1); j++) {
                    //     data[k + 1][i + j] = data[k][i + j] * data[k][i + (1 << k + 1) + j];
                    // }
                    ntt.dot_inplace(1 << k + 1, &data[k][i], &data[k][i + (1 << k + 1)], &data[k + 1][i]);
                    if (k + 1 != lg) {
                        std::copy(data[k + 1].begin() + i, data[k + 1].begin() + i + (1 << k + 1), data[k + 1].begin() + i + (1 << k + 1));
                        ntt.template transform<true>(k + 1, data[k + 1].data() + i + (1 << k + 1));
                        data[k + 1][i + (1 << k + 1)] -= 2;
                        ntt.template transform<false, true>(k + 1, data[k + 1].data() + i + (1 << k + 1));
                    } else {
                        ntt.template transform<true>(k + 1, data[k + 1].data() + i);
                        data[k + 1][i] -= 1;
                        data[k + 1][i + (1 << k + 1)] += 1;
                    }
                }
            }

            Poly dt = std::move(data[lg]);

            {
                cum_timer tm("top");
                // Poly dt(data[lg].begin(), data[lg].end());

                std::reverse(dt.begin(), dt.begin() + (1 << lg) + 1);
                dt.resize(1 << lg);

                dt = dt.inv_series(dt.size());
                std::reverse(dt.begin(), dt.end());

                dt.resize(1 << lg + 1);
                ntt.template transform<false>(lg + 1, dt.data());

                poly.resize(1 << lg + 1);
                std::rotate(poly.begin(), poly.begin() + (1 << lg + 1) - 1, poly.end());
                ntt.template transform<false>(lg + 1, poly.data());
                // for (int i = 0; i < (1 << lg + 1); i++) {
                //     dt[i] *= poly[i];
                // }
                ntt.dot_inplace(1 << lg + 1, &dt[0], &poly[0], &dt[0]);
            }
            for (int k = lg - 1; k >= 0; k--) {
                cum_timer tm(std::string("down  k = ") + std::to_string(k));

                for (int i = 0; i < (1 << lg + 1); i += (1 << k + 2)) {
                    ntt.template transform<true, true>(k + 1, dt.data() + i + (1 << k + 1));
                    ntt.template transform<false>(k + 1, dt.data() + i + (1 << k + 1));

                    if ((1 << k + 1) < 8) {
                        for (int j = 0; j < (1 << k + 1); j++) {
                            mint val = dt[i + j] - dt[i + (1 << k + 1) + j];
                            dt[i + j] = val * data[k][i + (1 << k + 1) + j];
                            dt[i + (1 << k + 1) + j] = val * data[k][i + j];
                        }
                    } else {
                        for (int j = 0; j < (1 << k + 1); j += 8) {
                            mint_x8 val = mint_x8::load(&dt[i + j]) - mint_x8::load(&dt[i + (1 << k + 1) + j]);
                            (val * mint_x8::load(&data[k][i + (1 << k + 1) + j])).store(&dt[i + j]);
                            (val * mint_x8::load(&data[k][i + j])).store(&dt[i + (1 << k + 1) + j]);
                        }
                    }
                }
            }

            std::vector<mint> ans(1 << lg);
            mint fix = mint(mod + 1 >> 1).power(lg + 1);
            for (int i = 0; i < (1 << lg); i++) {
                ans[i] = dt[2 * i] - dt[2 * i + 1];
                ans[i] *= fix;
            }
            ans.resize(res_sz);
            return ans;
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

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    constexpr u32 mod = 998'244'353;
    using mint = MintT<mod>;
    using poly = polynomial::Poly<mod>;

    int n, m;
    n = m = 1 << 17;
    std::cin >> n >> m;
    poly pl(n);
    std::vector<mint> pt(m);
    for (int i = 0; i < n; i++) {
        pl[i] = i;
        std::cin >> pl[i];
    }
    for (int i = 0; i < m; i++) {
        pt[i] = i;
        std::cin >> pt[i];
    }

    auto beg = clock();
    std::vector<mint> values = pl.evaluate(pt);
    std::cerr << "work: " << (clock() - beg) * 1.0 / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;

    for (int i = 0; i < m; i++) {
        std::cout << values[i] << " \n"[i == m - 1];
    }

    std::cerr << "ntt size total: " << ntt_sum_size / 1e6 << "M" << std::endl;

    return 0;
}
