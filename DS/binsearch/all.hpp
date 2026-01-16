#pragma once

#include <immintrin.h>
#include <stddef.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <memory>
#include <type_traits>
#include <vector>

#include "simd_cmp.hpp"
#include "simd_types.hpp"

template <typename T>
class Data {
   public:
    Data() = default;
    Data(const Data& other) = delete;
    Data operator=(const Data& other) = delete;

    virtual size_t lower_bound(T) const = 0;
};

template <typename T>
class Binsearch_Std : public Data<T> {
   private:
    size_t n;
    const T* data;

   public:
    Binsearch_Std() = default;
    Binsearch_Std(size_t n, const T* ptr) : n(n), data(ptr) { ; }

    size_t lower_bound(T val) const override {
        return std::lower_bound(data, data + n, val) - data;
    }
};

template <typename T>
class Binsearch_Branchless : public Data<T> {
   private:
    size_t n;
    const T* data;

   public:
    Binsearch_Branchless() = default;
    Binsearch_Branchless(size_t n, const T* ptr) : n(n), data(ptr) { ; }

    size_t lower_bound(T val) const override {
        size_t len = n;
        return len;
    }
};

template <typename T, size_t K>
class B_Tree : public Data<T> {
   protected:
    size_t n, lg;
    std::vector<const T*> vec;

   private:
    static size_t div_up(size_t a, size_t b) {
        assert(a > 0);
        return (a - 1) / b + 1;
    }

   public:
    B_Tree() = default;

    B_Tree(size_t n, const T* ptr) : n(n), lg(0) {
        static_assert(K > 0);
        static_assert(std::has_single_bit(K));

        if (n == 0) {
            return;
        }

        for (size_t m = n + 1; m > 1; m = div_up(m, K + 1)) {
            lg++;
        }
        vec.assign(lg, nullptr);
        vec[lg - 1] = ptr;

        for (size_t k = lg - 1, m = div_up(n + 1, K + 1), f = K + 1; k-- > 0; f *= K + 1) {
            size_t block_cnt = div_up(m, K + 1);
            vec[k] = _mm_malloc(sizeof(T) * K * block_cnt, sizeof(T) * K);
            for (size_t i = 0; i < block_cnt; i++) {
                for (size_t j = 0; j < K; j++) {
                    // index of maximum in subtree
                    size_t ind = i * f * (K + 1) + f * j + (f - 1);
                    vec[k][i] = ind < n ? ptr[m] : std::numeric_limits<T>::max();
                }
            }
            m = block_cnt;
        }
    }

    // size_t lower_bound(T val)  const override {
    //     ;
    //     return -1;
    // }
};

template <typename T, size_t K, class Extension_Tag>
class B_Tree_Simd : public B_Tree<T, K> {
   public:
    size_t lower_bound(T val) const override {
        return -1;
    }
};