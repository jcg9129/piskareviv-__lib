#include <iostream>

#pragma GCC target("avx2")
#include <immintrin.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

using i256 = __m256i;
using u64 = uint64_t;
using u64x4 = u64 __attribute__((vector_size(32)));

u64x4 set1_u64x4(u64 val) {
    return (u64x4)_mm256_set1_epi64x(val);
}

u64x4 mul_u64x4_naive(u64x4 a, u64x4 b) {
    return a * b;
}

u64x4 mul_u64x4_avx512(u64x4 a_, u64x4 b_) {
    i256 a = (i256)a_, b = (i256)b_;
    return (u64x4)_mm256_mullo_epi64(a, b);
}

u64x4 mul_u64x4_0(u64x4 a_, u64x4 b_) {
    i256 a = (i256)a_, b = (i256)b_;
    i256 a_sh = _mm256_shuffle_epi32(a, 0b10'11'00'01), b_sh = _mm256_shuffle_epi32(b, 0b10'11'00'01);

    i256 l = _mm256_mul_epu32(a, b);
    i256 tmp1 = _mm256_add_epi64(_mm256_mul_epu32(a_sh, b), _mm256_mul_epu32(a, b_sh));
    const __m128i shuffle_mask = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, 8, 9, 10, 11);
    i256 m = _mm256_shuffle_epi8(tmp1, _mm256_set_m128i(shuffle_mask, shuffle_mask));

    return (u64x4)_mm256_add_epi64(l, m);
}
u64x4 mul_u64x4_1(u64x4 a_, u64x4 b_) {
    i256 a = (i256)a_, b = (i256)b_;

    i256 l = _mm256_mul_epu32(a, b);
    i256 tmp1 = _mm256_mullo_epi32(a, _mm256_shuffle_epi32(b, 0b10'11'00'01));
    i256 tmp2 = _mm256_hadd_epi32(tmp1, tmp1);
    const __m128i shuffle_mask = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6, 7);
    i256 m = _mm256_shuffle_epi8(tmp2, _mm256_set_m128i(shuffle_mask, shuffle_mask));

    return (u64x4)_mm256_add_epi64(l, m);
}

u64x4 mul_u64x4_2(u64x4 a_, u64x4 b_) {
    i256 a = (i256)a_, b = (i256)b_;

    i256 l = _mm256_mul_epu32(a, b);
    i256 tmp1 = _mm256_mullo_epi32(a, _mm256_shuffle_epi32(b, 0b10'11'00'01));
    i256 tmp2 = _mm256_shuffle_epi32(tmp1, 0b10'11'00'01);
    i256 m = _mm256_and_si256(_mm256_add_epi32(tmp1, tmp2), _mm256_setr_epi32(0, -1, 0, -1, 0, -1, 0, -1));

    return (u64x4)_mm256_add_epi64(l, m);
}

// https://stackoverflow.com/questions/37296289/fastest-way-to-multiply-an-array-of-int64-t
u64x4 mul_u64x4_3(u64x4 aa, u64x4 bb) {
    i256 a = (i256)aa, b = (i256)bb;
    // There is no vpmullq until AVX-512. Split into 32-bit multiplies
    // Given a and b composed of high<<32 | low  32-bit halves
    // a*b = a_low*(u64)b_low  + (u64)(a_high*b_low + a_low*b_high)<<32;  // same for signed or unsigned a,b since we aren't widening to 128
    // the a_high * b_high product isn't needed for non-widening; its place value is entirely outside the low 64 bits.

    __m256i b_swap = _mm256_shuffle_epi32(b, _MM_SHUFFLE(2, 3, 0, 1));  // swap H<->L
    __m256i crossprod = _mm256_mullo_epi32(a, b_swap);                  // 32-bit L*H and H*L cross-products

    __m256i prodlh = _mm256_slli_epi64(crossprod, 32);                                     // bring the low half up to the top of each 64-bit chunk
    __m256i prodhl = _mm256_and_si256(crossprod, _mm256_set1_epi64x(0xFFFFFFFF00000000));  // isolate the other, also into the high half were it needs to eventually be
    __m256i sumcross = _mm256_add_epi32(prodlh, prodhl);                                   // the sum of the cross products, with the low half of each u64 being 0.

    __m256i prodll = _mm256_mul_epu32(a, b);            // widening 32x32 => 64-bit  low x low products
    __m256i prod = _mm256_add_epi32(prodll, sumcross);  // add the cross products into the high half of the result
    return (u64x4)prod;
}

// void test(auto &&f) {
#define test(f)                                                                                                             \
    {                                                                                                                       \
        u64x4 sum = set1_u64x4(0);                                                                                          \
        u64x4 val0 = set1_u64x4(0), val1 = set1_u64x4(0);                                                                   \
        u64x4 dlt0 = set1_u64x4(123456789876543), dlt1 = set1_u64x4(2551873683612321);                                      \
                                                                                                                            \
        clock_t beg = clock();                                                                                              \
        int n = 1e8;                                                                                                        \
        for (int i = 0; i < n; i++) {                                                                                       \
            sum += f(val0, val1);                                                                                           \
            val0 += dlt0, val1 += dlt1;                                                                                     \
        }                                                                                                                   \
        std::string name = #f;                                                                                              \
        name.resize(20, ' ');                                                                                               \
        std::cout << name << "  ";                                                                                          \
        std::cout << "time: " << std::fixed << (clock() - beg) * 1.0 / CLOCKS_PER_SEC / n * 1e9 << "ns  " << " checksum: "; \
                                                                                                                            \
        for (int i = 0; i < 4; i++) {                                                                                       \
            std::cout << std::hex << sum[i] << " \n"[i + 1 == 4];                                                           \
        }                                                                                                                   \
    }

int main() {
    test(mul_u64x4_naive);
    test(mul_u64x4_avx512);
    test(mul_u64x4_0);
    test(mul_u64x4_1);
    test(mul_u64x4_2);
    test(mul_u64x4_3);

    return 0;
}