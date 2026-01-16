#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#pragma GCC target("avx2")

namespace cum {

template <bool cond, typename T>
__always_inline T apply_if(const T& val, auto&& f) { return cond ? f(val) : val; }

template <bool reverse>
void reverse_if() { ; }
template <bool reverse>  // __always_inline is critical
__always_inline void reverse_if(auto f1, auto... f) { reverse ? (reverse_if<reverse>(f...), f1()) : (f1(), (f(), ...)); }

using u32 = uint32_t;
using u64 = uint64_t;

struct Montgomery {
    u32 mod;
    u32 n_inv;
    u32 r, r2;

    Montgomery() { ; }
    Montgomery(u32 mod) : mod(mod) {
        assert(mod % 2 == 1);
        assert(mod < (1 << 30));

        n_inv = 1;
        for (int i = 0; i < 5; i++) {
            n_inv *= 2 + n_inv * mod;
        }
        r = (u64(1) << 32) % mod, r2 = u64(r) * r % mod;
    }

    u32 shrink(u32 val) const { return val < mod ? val : val - mod; }
    u32 dilate(u32 val) const { return val < mod ? val : val + mod; }
    u32 reduce(u64 val) const { return val + u32(val) * n_inv * u64(mod) >> 32; }

    template <bool strict = true>
    u32 mul(u32 a, u32 b) const {
        return apply_if<strict>(reduce(u64(a) * b), [&](u32 val) { return shrink(val); });
    }

    template <bool input_in_space, bool output_in_space>
    u32 power(u32 b, u64 e) const {
        b = input_in_space ? b : mul<0>(b, r2);
        u32 r = output_in_space ? this->r : 1;
        for (; e; e >>= 1) {
            if (e & 1)
                r = mul<0>(r, b);
            b = mul<0>(b, b);
        }
        return shrink(r);
    }
};

namespace simd {

using i256 = __m256i;
using u32x8 = u32 __attribute__((vector_size(32), may_alias, /* __alinged__(1) */));
using u64x4 = u64 __attribute__((vector_size(32), may_alias, /* __alinged__(1) */));

u64x4 load_x4(const u64* ptr) { return (u64x4)_mm256_load_si256((const i256*)ptr); }
u64x4 loadu_x4(const u64* ptr) { return (u64x4)_mm256_loadu_si256((const i256*)ptr); }
u32x8 load_x8(const u32* ptr) { return (u32x8)_mm256_load_si256((const i256*)ptr); }
u32x8 loadu_x8(const u32* ptr) { return (u32x8)_mm256_loadu_si256((const i256*)ptr); }
void store_x8(u32* ptr, u32x8 vec) { _mm256_store_si256((i256*)ptr, (i256)vec); }
void storeu_x8(u32* ptr, u32x8 vec) { _mm256_storeu_si256((i256*)ptr, (i256)vec); }

u32x8 set1_u32x8(u32 val) { return u32x8() + val; }
u64x4 mul_32x32_to_64(u64x4 a, u64x4 b) { return (u64x4)_mm256_mul_epu32((i256)a, (i256)b); }

template <int mask>
u32x8 blend(u32x8 a, u32x8 b) { return (u32x8)_mm256_blend_epi32((i256)a, (i256)b, mask); }
u32x8 permute(u32x8 vec, u32x8 perm) { return (u32x8)_mm256_permutevar8x32_epi32((i256)vec, (i256)perm); }

u32x8 swap_adjecent(u32x8 val) { return (u32x8)_mm256_shuffle_epi32((i256)val, 0b10'11'00'01); }
u32x8 swap_dist_2(u32x8 val) { return (u32x8)_mm256_shuffle_epi32((i256)val, 0b01'00'11'10); }
u32x8 swap_dist_4(u32x8 val) { return (u32x8)_mm256_permute2x128_si256((i256)val, (i256)val, 1); }

u32x8 min_u32x8(u32x8 a, u32x8 b) { return (u32x8)_mm256_min_epu32((i256)a, (i256)b); }

struct Montgomery_simd {
    u32x8 mod, mod2, n_inv, r, r2;

    Montgomery_simd() { ; }
    Montgomery_simd(u32 mod) {
        Montgomery mt(mod);
        this->mod = set1_u32x8(mod), this->mod2 = set1_u32x8(2 * mod), this->n_inv = set1_u32x8(mt.n_inv), this->r = set1_u32x8(mt.r), this->r2 = set1_u32x8(mt.r2);
    }

    u32x8 shrink(u32x8 val) const { return min_u32x8(val, val - mod); }
    u32x8 dilate(u32x8 val) const { return min_u32x8(val, val + mod); }

    u32x8 shrink2(u32x8 val) const { return min_u32x8(val, val - mod2); }
    u32x8 dilate2(u32x8 val) const { return min_u32x8(val, val + mod2); }

    u32x8 shrink_12(u32x8 val) const { return shrink(shrink2(val)); }

    u64x4 reduce_aux(u64x4 val) const { return val + mul_32x32_to_64(mul_32x32_to_64(val, (u64x4)n_inv), (u64x4)mod); }
    u32x8 reduce(u64x4 x0246, u64x4 x1357) const { return swap_adjecent((u32x8)reduce_aux(x0246)) | (u32x8)reduce_aux(x1357); }
    u32x8 fix_order(u32x8 vec) const { return (u32x8)_mm256_permutevar8x32_epi32((i256)vec, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)); }

    template <bool strict = true, bool b_use_only_even = false, bool use_hint = false>
    u32x8 mul_u32x8(u32x8 a, u32x8 b, u32x8 hint = u32x8()) const {
        if (b_use_only_even && use_hint) {
            u32x8 a2 = swap_adjecent(a);
            u64x4 r0 = mul_32x32_to_64((u64x4)a, (u64x4)b) + mul_32x32_to_64(mul_32x32_to_64((u64x4)hint, (u64x4)a), (u64x4)mod);
            u64x4 r1 = mul_32x32_to_64((u64x4)a2, (u64x4)b) + mul_32x32_to_64(mul_32x32_to_64((u64x4)hint, (u64x4)a2), (u64x4)mod);
            u32x8 res = swap_adjecent((u32x8)r0) | (u32x8)r1;
            return strict ? shrink(res) : res;
        }
        u32x8 res = reduce(mul_32x32_to_64((u64x4)a, (u64x4)b), mul_32x32_to_64((u64x4)swap_adjecent(a), (u64x4)(b_use_only_even ? b : swap_adjecent(b))));
        return strict ? shrink(res) : res;
    }

    template <bool strict = true, bool use_hint = false>
    u64x4 mul_u64x4(u64x4 a, u64x4 b, u64x4 hint = u64x4()) {
        u64x4 x0246 = mul_32x32_to_64(a, b);
        u64x4 x0246_ninv = use_hint ? mul_32x32_to_64(a, hint) : mul_32x32_to_64(x0246, (u64x4)n_inv);
        u64x4 res = (u64x4)swap_adjecent((u32x8)(mul_32x32_to_64(x0246_ninv, (u64x4)mod) + x0246));
        return strict ? (u64x4)shrink((u32x8)res) : res;
    }
};

}  // namespace simd

struct NTT {
    u32 mod, pr_root;
    Montgomery mt;
    simd::Montgomery_simd mts;

    u32 find_pr_root() const {
        u32 n = mod - 1;
        std::vector<u32> factors;
        for (u64 i = 2; i * i <= n; i++) {
            if (n % i == 0) {
                factors.push_back(i);
                while (n % i == 0) n /= i;
            }
        }
        if (n != 1) factors.push_back(n);
        for (u32 g = 2; g < mod; g++) {
            if (std::all_of(factors.begin(), factors.end(), [&](u32 f) { return mt.power<0, 0>(g, (mod - 1) / f) != 1; })) {
                return g;
            }
        }
        assert(false && "primitive root not found");
    }

    static constexpr int LG = 32;

    struct Info {
        std::vector<u32> wd, wh;

        simd::u32x8 wld_x8[LG][2], wlh_x8_init[LG][2];

        simd::u64x4 wd_x4[LG];

        simd::u32x8 w4_x8, w8_pw_x4_x2, w16_pw_x8;
        simd::u64x4 w8_pw_x4, w4_pw_x2_x2;

        Info() { ; }
        Info(u32 mod, u32 pr_root) {
            Montgomery mt(mod);
            simd::Montgomery_simd mts(mod);

            auto powers_x8 = [&](u32 val, std::array<u32, 8> pw = {0, 1, 2, 3, 4, 5, 6, 7}) {
                simd::u32x8 res;
                for (int i = 0; i < 8; i++) res[i] = mt.power<1, 1>(val, pw[i]);
                return res;
            };

            int lg = __builtin_ctz(mod - 1);

            wd.assign(lg, 0), wh.assign(lg, 0);
            memset(&wld_x8, 0, sizeof(wld_x8));

            for (int k = 0; k + 2 <= lg; k++) {
                u32 a = mt.power<0, 1>(pr_root, mod - 1 >> k + 2), b = mt.power<1, 1>(a, (2 << k) - 2);
                wh[k] = a;
                wd[k] = mt.mul(a, mt.power<1, 1>(b, mod - 2));

                if (k >= 3) {
                    auto set = [&](simd::u32x8* ar, u32 val) {
                        ar[0] = powers_x8(val), ar[1] = mts.mul_u32x8(ar[0], simd::set1_u32x8(mt.power<1, 1>(val, 8)));
                    };
                    set(wld_x8[k - 3], wd[k]);
                    set(wlh_x8_init[k - 3], wh[k]);
                }
            }

            w4_x8 = simd::set1_u32x8(mt.power<0, 1>(pr_root, mod - 1 >> 2));

            w4_pw_x2_x2 = (simd::u64x4)powers_x8(mt.power<0, 1>(pr_root, mod - 1 >> 2), {0, 0, 1, 1, 0, 0, 1, 1});
            w8_pw_x4 = (simd::u64x4)powers_x8(mt.power<0, 1>(pr_root, mod - 1 >> 3), {0, 0, 1, 1, 2, 2, 3, 3});

            w8_pw_x4_x2 = powers_x8(mt.power<0, 1>(pr_root, mod - 1 >> 3), {0, 1, 2, 3, 0, 1, 2, 3});
            w16_pw_x8 = powers_x8(mt.power<0, 1>(pr_root, mod - 1 >> 4), {0, 1, 2, 3, 4, 5, 6, 7});
        }
    };

    std::array<Info, 2> info;

    NTT() { ; }
    NTT(u32 mod) : mod(mod), mt(mod), mts(mod) {
        pr_root = find_pr_root();
        info[0] = Info(mod, pr_root), info[1] = Info(mod, mt.power<0, 0>(pr_root, mod - 2));
    }

    template <bool inverse>
    void butterfly_x2(u32& a, u32& b, u32 w) const {
        u32 x = a, y = b;
        inverse ? (a = mt.shrink(x + y), b = mt.mul(x + mod - y, w)) : (y = mt.mul(y, w), a = mt.shrink(x + y), b = mt.shrink(x + mod - y));
    }

    template <bool inverse, bool trivial = false>
    void butterfly_x2(u32* a, u32* b, simd::u32x8 w) const {
        simd::u32x8 x = simd::loadu_x8(a), y = simd::loadu_x8(b);
        inverse ? (simd::storeu_x8(a, mts.shrink2(x + y)), simd::storeu_x8(b, trivial ? mts.dilate2(x - y) : mts.mul_u32x8<false, true>(x + mts.mod2 - y, w)))
                : (x = mts.shrink2(x), (y = trivial ? mts.shrink2(y) : mts.mul_u32x8<false, true>(y, w)), simd::storeu_x8(a, x + y), simd::storeu_x8(b, x + mts.mod2 - y));
    }

    // void butterfly_x4(u32* a, u32* b, u32* c, u32* d, simd::u32x8 w, simd::u32x8 w1, simd::u32x8 w2) {
    //     ;
    // }

    // template <bool inverse>
    // void dft_x8_x2(simd::u32x8& a, simd::u32x8& b) const {
    //     using namespace simd;

    //     reverse_if<inverse>(
    //         [&] { fuck1(a); },
    //         [&] { fuck2(a); },
    //         [&] { fuck3(a); },
    //         [&] { fuck4(a); },
    //         [&] { fuck5(a); });

    //     reverse_if<inverse>(
    //         [&] { fuck1(b); },
    //         [&] { fuck2(b); },
    //         [&] { fuck3(b); },
    //         [&] { fuck4(b); },
    //         [&] { fuck5(b); });
    // }

    template <bool inverse, bool use_fix = false>
    // __always_inline
    void dft_x16(u32* data, simd::u32x8* fix = nullptr) const {
        // return;
        using namespace simd;

        auto fuck1 = [&](u32x8& a) {
            a = mts.shrink2(swap_dist_4(a) + blend<0xF0>(a, mts.mod2 - a));
        };
        auto fuck2 = [&](u32x8& a) {
            u32x8 b = (u32x8)mts.reduce_aux(mul_32x32_to_64((u64x4)permute(a, (u32x8)(u64x4{4, 5, 6, 7})), info[inverse].w8_pw_x4));
            a = blend<0xf0>(a, permute(b, u32x8{0, 0, 0, 0, 1, 3, 5, 7}));
        };
        auto fuck3 = [&](u32x8& a) {
            a = mts.shrink2(swap_dist_2(a) + blend<0b11'00'11'00>(a, mts.mod2 - a));
        };
        auto fuck4 = [&](u32x8& a) {
            u32x8 b = (u32x8)mts.reduce_aux(mul_32x32_to_64((u64x4)_mm256_shuffle_epi32((i256)a, 0x32), info[inverse].w4_pw_x2_x2));
            a = blend<0b11'00'11'00>(a, (u32x8)_mm256_shuffle_epi32((i256)b, 0b11'01'00'00));
        };
        auto fuck5 = [&](u32x8& a) {
            a = mts.shrink2(swap_adjecent(a) + blend<0b10'10'10'10>(a, mts.mod2 - a));
        };

        u32x8 a = loadu_x8(data), b = loadu_x8(data + 8);

        reverse_if<inverse>(
            [&] {
                if (use_fix) {
                    a = mts.mul_u32x8<false>(a, fix[0]), b = mts.mul_u32x8<false>(b, fix[1]);
                } else {
                    // assert(false);
                    a = mts.shrink2(a), b = mts.shrink2(b);
                }
            },
            [&] {
                reverse_if<inverse>(
                    [&] { u32x8 x = a, y = b; a = mts.shrink2(x + y), b = mts.dilate2(x - y); },
                    [&] { b = mts.mul_u32x8<false>(b, info[inverse].w16_pw_x8); });
            },

            // [&] {
            //     reverse_if<inverse>(
            //         [&] { fuck1(a); },
            //         [&] { fuck2(a); },
            //         [&] { fuck3(a); },
            //         [&] { fuck4(a); },
            //         [&] { fuck5(a); });
            // },

            // [&] {
            //     reverse_if<inverse>(
            //         [&] { fuck1(b); },
            //         [&] { fuck2(b); },
            //         [&] { fuck3(b); },
            //         [&] { fuck4(b); },
            //         [&] { fuck5(b); });
            // }

            [&] { fuck1(a); },
            [&] { fuck2(a); },
            [&] { fuck3(a); },
            [&] { fuck4(a); },
            [&] { fuck5(a); },

            [&] { fuck1(b); },
            [&] { fuck2(b); },
            [&] { fuck3(b); },
            [&] { fuck4(b); },
            [&] { fuck5(b); }

            // [&] { fuck1(a); },
            // [&] { fuck1(b); },

            // [&] { fuck3(a); },
            // [&] { fuck3(b); },

            // [&] { fuck2(b); },
            // [&] { fuck2(a); },

            // [&] { fuck4(a); },
            // [&] { fuck4(b); },

            // [&] { fuck5(a); },
            // [&] { fuck5(b); }

        );

        a = mts.shrink(a), b = mts.shrink(b);

        storeu_x8(data, a), storeu_x8(data + 8, b);
    }

    // template <bool inverse, bool trivial = false>
    // void transform_aux_radix2(int lg, u32* data, int k, int i, u32& wi) const {
    //     if (!trivial && wi == mt.r) {
    //         return transform_aux_radix2<inverse, true>(lg, data, k, i, wi);
    //     } else {
    //         simd::u32x8 wi_x8 = simd::set1_u32x8(wi);
    //         for (int j = 0; j < (1 << k); j += 8) {
    //             butterfly_x2<inverse, trivial>(&data[i + j], &data[i + (1 << k) + j], wi_x8);
    //         }
    //     }
    //     wi = mt.mul(wi, info[inverse].wd[__builtin_ctz(~i >> k + 1)]);
    // }

    template <bool inverse, bool trivial = false>
    void transform_aux_radix4(int lg, u32* data, int k, int i, u64x4& wi) const {
        if (!trivial && wi == mt.r) {
            return transform_aux_radix4<inverse, true>(lg, data, k, i, wi);
        } else {
            simd::u32x8 wi_x8 = simd::set1_u32x8(wi);
            for (int j = 0; j < (1 << k); j += 8) {
                butterfly_x2<inverse, trivial>(&data[i + j], &data[i + (1 << k) + j], wi_x8);
            }
        }
        wi = mt.mul(wi, info[inverse].wd[__builtin_ctz(~i >> k + 1)]);
    }

    template <bool inverse, bool right_part = false, bool remove_factor = false>
    void transform(int lg, u32* data) const {
        const auto mt = this->mt;
        const auto mts = this->mts;

        // if (lg == 3 && !right_part) {
        //     simd::u32x8 a = simd::loadu_x8(data);
        //     dft_x8<inverse>(a);
        //     if (inverse) {
        //         a = mts.mul_u32x8(a, simd::set1_u32x8(mt.power<0, 1>(mod + 1 >> 1, lg)));
        //     }
        //     simd::storeu_x8(data, a);
        //     return;
        // }

        if (lg <= 4) {
            for (int k = inverse ? 0 : lg - 1; inverse ? k < lg : k >= 0; inverse ? k++ : k--) {
                u32 wi = right_part ? info[inverse].wh[lg - 1 - k] : mt.r;
                for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                    for (int j = 0; j < (1 << k); j++) {
                        butterfly_x2<inverse>(data[i + j], data[i + (1 << k) + j], wi);
                    }
                    wi = mt.mul(wi, info[inverse].wd[__builtin_ctz(~i >> k + 1)]);
                }
            }
            if (inverse) {
                u32 f = mt.power<0, 1>(mod + 1 >> 1, lg);
                if (remove_factor) f = mt.mul(f, mt.r2);
                for (int i = 0; i < (1 << lg); i++) {
                    data[i] = mt.mul(data[i], f);
                }
            }
            return;
        }

        // std::vector<u32> wi(lg);
        std::array<u32, LG> wi;
        for (int k = 0; k < lg; k++) {
            wi[k] = right_part ? info[inverse].wh[lg - 1 - k] : mt.r;
        }

        simd::u32x8 wi_l[2] = {mts.r, mts.r};
        if (right_part) wi_l[0] = info[inverse].wlh_x8_init[lg - 4][0], wi_l[1] = info[inverse].wlh_x8_init[lg - 4][1];

        for (int i = 0; i < (1 << lg); i += 32) {
            int h = __builtin_ctz(inverse ? i + 32 : (1 << lg) + i);
            reverse_if<inverse>(
                [&] {
                    for (int k = inverse ? 4 : h - 1; inverse ? k < h : k >= 4; inverse ? k++ : k--) {
                        transform_aux_radix2<inverse>(lg, data, k, inverse ? i + 32 - (2 << k) : i, wi[k]);
                    }
                },
                [&] {
                    for (int j = 0; j < 32; j += 16) {
#ifdef TEST
                        dft_x16<inverse, true>(data + i + j, wi_l);
                        int t = __builtin_ctz(~(i + j) >> 4);
                        wi_l[0] = mts.mul_u32x8(wi_l[0], info[inverse].wld_x8[t][0]), wi_l[1] = mts.mul_u32x8(wi_l[1], info[inverse].wld_x8[t][1]);
#endif
                    }
                });
        }

        if (inverse) {
            u32 f = mt.power<0, 1>(mod + 1 >> 1, lg);
            if (remove_factor) f = mt.mul(f, mt.r2);
            simd::u32x8 f_x8 = simd::set1_u32x8(f);
            for (int i = 0; i < (1 << lg); i += 8) {
                simd::storeu_x8(&data[i], mts.mul_u32x8(simd::loadu_x8(&data[i]), f_x8));
            }
        }
    }

    template <bool negacyclic = false>
    void convolve_cyclic(int lg, u32* a, u32* b) const {
        // if (lg <= 3) {
        //     alignas(64) u64 c[8];
        //     std::fill(c, c + 8, 0);
        //     u64 mod_sq = u64(mod) * mod;
        //     for (int i = 0; i < (1 << lg); i++) {
        //         for (int j = 0; j < (1 << lg) - i; j++) {
        //             c[i + j] += u64(a[i]) * b[j];
        //         }
        //         for (int j = (1 << lg) - i; j < (1 << lg); j++) {
        //             c[i + j - (1 << lg)] += negacyclic ? mod_sq - u64(a[i]) * b[j] : u64(a[i]) * b[j];
        //         }
        //     }
        //     alignas(32) u32 d[8];
        //     simd::store_x8(d, mts.mul_u32x8(mts.r2, mts.fix_order(mts.reduce(simd::load_x4(c), simd::load_x4(c + 4)))));
        //     std::copy(d, d + (1 << lg), a);
        //     return;
        // }

        transform<false, negacyclic>(lg, a);
        transform<false, negacyclic>(lg, b);
        if (lg < 3) {
            for (int i = 0; i < (1 << lg); i++) {
                a[i] = mt.mul(a[i], b[i]);
            }
        } else {
            for (int i = 0; i < (1 << lg); i += 8) {
                // simd::storeu_x8(&a[i], mts.mul_u32x8(mts.mul_u32x8(simd::loadu_x8(&a[i]), simd::loadu_x8(&b[i])), mts.r2));
                simd::storeu_x8(&a[i], mts.mul_u32x8(simd::loadu_x8(&a[i]), simd::loadu_x8(&b[i])));
            }
        }
        transform<true, negacyclic, true>(lg, a);
    }
};
};  // namespace cum
