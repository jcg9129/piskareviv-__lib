#pragma once

#include <immintrin.h>

#include <cstdint>

#include "simd_types.hpp"

template <typename Ext_type, typename Enable = void>
class Simd_Compare;

template <typename Ext_type>
class Simd_Compare<Ext_type, typename std::enable_if<std::is_base_of<Sse2_tag, Ext_type>::value>::type> {
   public:
    template <typename T, std::enable_if<sizeof(T) == 1 && std::is_integral<T>::value && std::is_signed<T>::value, bool>::type = 0>
    static Register_Type<Ext_type, T> compare_greater_than(Register_Type<Ext_type, T> a, Register_Type<Ext_type, T> b) {
        return _mm_cmpgt_epi8(a, b);
    }

    template <typename T, std::enable_if<sizeof(T) == 1 && std::is_integral<T>::value && std::is_unsigned<T>::value, bool>::type = 0>
    static Register_Type<Ext_type, T> compare_greater_than(Register_Type<Ext_type, T> a, Register_Type<Ext_type, T> b) {
        const __m128i mask = _mm_set1_epi16(1U << 7);
        return _mm_cmpgt_epi8(_mm_xor_si128(a, mask), _mm_xor_si128(b, mask));
    }

    template <typename T, std::enable_if<sizeof(T) == 2 && std::is_integral<T>::value && std::is_signed<T>::value, bool>::type = 0>
    static Register_Type<Ext_type, T> compare_greater_than(Register_Type<Ext_type, T> a, Register_Type<Ext_type, T> b) {
        return _mm_cmpgt_epi16(a, b);
    }

    template <typename T, std::enable_if<sizeof(T) == 2 && std::is_integral<T>::value && std::is_unsigned<T>::value, bool>::type = 0>
    static Register_Type<Ext_type, T> compare_greater_than(Register_Type<Ext_type, T> a, Register_Type<Ext_type, T> b) {
        const __m128i mask = _mm_set1_epi16(1U << 15);
        return _mm_cmpgt_epi16(_mm_xor_si128(a, mask), _mm_xor_si128(b, mask));
    }

    template <typename T, std::enable_if<sizeof(T) == 4 && std::is_integral<T>::value && std::is_signed<T>::value, bool>::type = 0>
    static Register_Type<Ext_type, T> compare_greater_than(Register_Type<Ext_type, T> a, Register_Type<Ext_type, T> b) {
        return _mm_cmpgt_epi32(a, b);
    }

    template <typename T, std::enable_if<sizeof(T) == 4 && std::is_integral<T>::value && std::is_unsigned<T>::value, bool>::type = 0>
    static Register_Type<Ext_type, T> compare_greater_than(Register_Type<Ext_type, T> a, Register_Type<Ext_type, T> b) {
        const __m128i mask = _mm_set1_epi32(1U << 31);
        return _mm_cmpgt_epi32(_mm_xor_si128(a, mask), _mm_xor_si128(b, mask));
    }

    template <typename T, std::enable_if<sizeof(T) == 8 && std::is_integral<T>::value && std::is_signed<T>::value, bool>::type = 0>
    static Register_Type<Ext_type, T> compare_greater_than(Register_Type<Ext_type, T> a, Register_Type<Ext_type, T> b) {
        return _mm_cmpgt_epi64(a, b);
    }

    template <typename T, std::enable_if<sizeof(T) == 8 && std::is_integral<T>::value && std::is_unsigned<T>::value, bool>::type = 0>
    static Register_Type<Ext_type, T> compare_greater_than(Register_Type<Ext_type, T> a, Register_Type<Ext_type, T> b) {
        const __m128i mask = _mm_set1_epi64x(1ULL << 63);
        return _mm_cmpgt_epi64(_mm_xor_si128(a, mask), _mm_xor_si128(b, mask));
    }

    template <typename T, typename std::enable_if<sizeof(T) == 4 && std::is_floating_point<T>::value, bool>::type = 0>
    static Register_Type<Ext_type, T> compare_greater_than(Register_Type<Ext_type, T> a, Register_Type<Ext_type, T> b) {
        return _mm_cmpgt_ps(a, b);
    }

    template <typename T, typename std::enable_if<sizeof(T) == 8 && std::is_floating_point<T>::value, bool>::type = 0>
    static Register_Type<Ext_type, T> compare_greater_than(Register_Type<Ext_type, T> a, Register_Type<Ext_type, T> b) {
        return _mm_cmpgt_pd(a, b);
    }
};

auto c1 = Simd_Compare<Sse2_tag>::compare_greater_than<int>(__m128i(), __m128i());
auto c3 = Simd_Compare<Sse2_tag>::compare_greater_than<float>(__m128(), __m128());
auto c4 = Simd_Compare<Sse2_tag>::compare_greater_than<double>(__m128d(), __m128d());

auto cmp1 = Simd_Compare<Sse2_tag>::compare_greater_than<uint32_t>(__m128i(), __m128i());
auto cmp2 = Simd_Compare<Sse2_tag>::compare_greater_than<unsigned>(__m128i(), __m128i());
auto cmp3 = Simd_Compare<Sse2_tag>::compare_greater_than<unsigned int>(__m128i(), __m128i());

auto cmp4 = Simd_Compare<Sse2_tag>::compare_greater_than<short>(__m128i(), __m128i());
auto cmp5 = Simd_Compare<Sse2_tag>::compare_greater_than<uint16_t>(__m128i(), __m128i());
auto cmp6 = Simd_Compare<Sse2_tag>::compare_greater_than<char>(__m128i(), __m128i());
auto cmp7 = Simd_Compare<Sse2_tag>::compare_greater_than<unsigned char>(__m128i(), __m128i());
