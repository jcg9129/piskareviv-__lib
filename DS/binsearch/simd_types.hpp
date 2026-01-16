#pragma once

#include <immintrin.h>

#include <type_traits>

struct Null_Ext_tag {};

struct Sse_tag {};
struct Sse2_tag : public Sse_tag {};
struct Sse_41_tag : public Sse2_tag {};
struct Sse_42_tag : public Sse_41_tag {};

struct Avx_tag {};
struct Avx2_tag : public Avx_tag {};

struct Avx512_tag {};
struct Avx512f_tag : public Avx512_tag {};

template <typename Extension_Tag, typename T, typename Enable = void>
struct Register_Type_Aux;

template <typename T>
struct Register_Type_Aux<Null_Ext_tag, T> {
    using type = T;
};

template <typename Ext_Tag, typename T>
struct Register_Type_Aux<Ext_Tag, T, typename std::enable_if<std::is_base_of<Sse_tag, Ext_Tag>::value && sizeof(T) == 4 && std::is_floating_point<T>::value>::type> {
    using type = __m128;
};
template <typename Ext_Tag, typename T>
struct Register_Type_Aux<Ext_Tag, T, typename std::enable_if<std::is_base_of<Sse_tag, Ext_Tag>::value && sizeof(T) == 8 && std::is_floating_point<T>::value>::type> {
    using type = __m128d;
};
template <typename Ext_Tag, typename T>
struct Register_Type_Aux<Ext_Tag, T, typename std::enable_if<std::is_base_of<Sse_tag, Ext_Tag>::value && std::is_integral<T>::value>::type> {
    using type = __m128i;
};

template <typename Ext_Tag, typename T>
struct Register_Type_Aux<Ext_Tag, T, typename std::enable_if<std::is_base_of<Avx_tag, Ext_Tag>::value && sizeof(T) == 4 && std::is_floating_point<T>::value>::type> {
    using type = __m256;
};
template <typename Ext_Tag, typename T>
struct Register_Type_Aux<Ext_Tag, T, typename std::enable_if<std::is_base_of<Avx_tag, Ext_Tag>::value && sizeof(T) == 8 && std::is_floating_point<T>::value>::type> {
    using type = __m256d;
};
template <typename Ext_Tag, typename T>
struct Register_Type_Aux<Ext_Tag, T, typename std::enable_if<std::is_base_of<Avx_tag, Ext_Tag>::value && std::is_integral<T>::value>::type> {
    using type = __m256i;
};

template <typename Ext_Tag, typename T>
struct Register_Type_Aux<Ext_Tag, T, typename std::enable_if<std::is_base_of<Avx512_tag, Ext_Tag>::value && sizeof(T) == 4 && std::is_floating_point<T>::value>::type> {
    using type = __m512;
};
template <typename Ext_Tag, typename T>
struct Register_Type_Aux<Ext_Tag, T, typename std::enable_if<std::is_base_of<Avx512_tag, Ext_Tag>::value && sizeof(T) == 8 && std::is_floating_point<T>::value>::type> {
    using type = __m512d;
};
template <typename Ext_Tag, typename T>
struct Register_Type_Aux<Ext_Tag, T, typename std::enable_if<std::is_base_of<Avx512_tag, Ext_Tag>::value && std::is_integral<T>::value>::type> {
    using type = __m512i;
};

template <typename Extenstion_Tag, typename T>
using Register_Type = Register_Type_Aux<Extenstion_Tag, T>::type;

template <typename T>
int func(T val) {
    if constexpr (std::is_integral<T>::value) {
        return 1;
    } else {
        return 2;
    }
}
