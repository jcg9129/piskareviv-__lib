#pragma once

#include <immintrin.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <random>

namespace Field_64 {

using u64 = uint64_t;
using u128 = __uint128_t;

__m128i clmul_vec(u64 a, u64 b) {
    __m128i tmp = _mm_clmulepi64_si128(_mm_cvtsi64_si128(a), _mm_cvtsi64_si128(b), 0);
    return tmp;
}

u128 clmul(u64 a, u64 b) {
    __m128i tmp = clmul_vec(a, b);
    u128 res;
    memcpy(&res, &tmp, 16);
    return res;
}

constexpr u128 clmul_constexpr(u64 a, u64 b) {
    u128 res = 0;
    for (int i = 0; i < 64; i++) {
        if (a >> i & 1) {
            res ^= u128(b) << i;
        }
    }
    return res;
}

int lg_u128(u128 val) {
    u64 a = val, b = val >> 64;
    return b ? 64 + (63 - __builtin_clzll(b)) : (a ? 63 - __builtin_clzll(a) : -1);
}

u128 take_mod(u128 val, u128 mod) {
    int lg = lg_u128(mod);
    for (int i = lg_u128(val); i >= lg; i = lg_u128(val)) {
        val ^= mod << i - lg;
    }
    return val;
}

u128 pow_mod(u128 b, u128 exp, u128 mod) {
    assert(lg_u128(mod) <= 64);
    u128 r = 1;
    for (; exp; exp >>= 1) {
        if (exp & 1) {
            r = take_mod(clmul(r, b), mod);
        }
        b = take_mod(clmul(b, b), mod);
    }
    return r;
}

u128 poly_gcd(u128 a, u128 b) {
    while (b) {
        a = take_mod(a, b), std::swap(a, b);
    }
    return a;
}

bool is_irreducible_naive(u128 mod) {
    int lg = lg_u128(mod);
    for (u128 i = 2; i < (u128(1) << lg / 2 + 1) && i < mod; i++) {
        if (take_mod(mod, i) == 0) {
            return false;
        }
    }
    return true;
}

bool is_irreducible(u128 mod) {
    const int n = lg_u128(mod);
    auto get = [&](int k) {
        return take_mod(pow_mod(2, u128(1) << k, mod) ^ 2, mod);
    };
    bool result = true;
    if (poly_gcd(get(n), mod) != mod) {
        result = false;
    } else {
        for (int i = 1; i < n; i++) {
            if (n % i == 0 && poly_gcd(get(i), mod) != 1) {
                result = false;
                break;
            }
        }
    }
    // if (result != is_irreducible_naive(mod)) {
    //     std::cerr << result << " " << (u64)mod << "\n";
    // }
    // assert(result == is_irreducible_naive(mod));
    return result;
}

// std::mt19937_64 rnd_64;

u128 find_irreducible_poly(int deg) {
    // while (true) {
    //     u128 mod = rnd_64() | u128(rnd_64()) << 64;
    //     mod >>= 127 - deg;
    //     mod |= u128(1) << deg;
    //     if (is_irreducible(mod)) {
    //         return mod;
    //     }
    // }
    for (u128 mod = u128(1) << deg; (mod >> deg + 1) == 0; mod++) {
        if (is_irreducible(mod)) {
            return mod;
        }
    }
    assert(false);
}

struct Field {
   public:
    static constexpr u128 mod = 0b11011 | u128(1) << 64;

   private:
    static constexpr u64 inv = [] {
        auto clmul = clmul_constexpr;
        u64 a = 1;
        for (int i = 0; i < 6; i++) {
            a = clmul(a, clmul(a, (u64)mod));
        }
        return a;
    }();
    static constexpr auto pow = [](int exp) {
        u128 r = 1;
        for (int i = 0; i < exp; i++) {
            r <<= 1;
            if (r >> 64 & 1) {
                r ^= mod;
            }
        }
        return (u64)r;
    };

    static constexpr u64 r = pow(64);
    static constexpr u64 r2 = pow(128);

    static_assert((u64)clmul_constexpr(u64(mod), inv) == 1);
    
   public:

    static u64 reduce(u128 val) {
        u64 f = clmul(val, inv);
        // return (val ^ clmul(f, (u64)mod) ^ u128(f) << 64) >> 64;
        return val >> 64 ^ clmul(f, (u64)mod) >> 64 ^ f;
    }

   // private:
   public:
    u64 val;
    
   public:

    __always_inline Field(u64 val, int) : val(val) { ; }

   public:
    __always_inline explicit Field() : val(0) { ; }
    __always_inline explicit Field(u64 val) : val(reduce(clmul(val, r2))) { ; }

    static Field n(int64_t n) { return Field(n & 1); }

    Field operator-() const { return Field(val, 0); }
    Field operator+(const Field& other) const { return Field(val ^ other.val, 0); }
    Field operator-(const Field& other) const { return Field(val ^ other.val, 0); }

    // Field operator*(const Field& other) const { return Field(take_mod(clmul(val, other.val), mod)); }
    Field operator*(const Field& other) const {
        return Field(reduce(clmul(val, other.val)), 0);
    }

    Field& operator+=(const Field& other) { return *this = *this + other; }
    Field& operator-=(const Field& other) { return *this = *this - other; }
    Field& operator*=(const Field& other) { return *this = *this * other; }

    // u64 get() const { return val; }
    u64 get() const { return reduce(val); }

    bool operator==(const Field& other) const { return val == other.val; }

    // friend std::ostream& operator<<(std::ostream& out, const Field& val) { return out << val.val; }
};

};  // namespace Field_64
