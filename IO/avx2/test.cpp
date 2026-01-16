
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cassert>
#include <iostream>

#pragma GCC target("sse4.2,bmi")
#include <immintrin.h>

using s64 = int64_t;
using s32 = int32_t;
using u32 = uint32_t;
using u64 = uint64_t;

using u8x16 = uint8_t __attribute__((vector_size(16)));
using u16x8 = uint16_t __attribute__((vector_size(16)));
using u32x4 = uint32_t __attribute__((vector_size(16)));

struct Cum_Input {
    char *begin, *end;
    char *cur;
    size_t sz;

    __m128i ar[12];

    Cum_Input(int fd) {
        struct stat st;
        fstat(fd, &st);
        sz = st.st_size;
        begin = (char *)mmap(0, sz + 1, PROT_READ, MAP_PRIVATE, fd, 0);
        cur = begin;
        end = begin + sz;
        madvise(begin, sz + 1, MADV_SEQUENTIAL);

        {
            for (int i = 0; i < 12; i++) {
                using u8x16 = char __attribute__((vector_size(16)));
                ar[i] = _mm_set1_epi8(-1);
                for (int j = 0; j < i; j++) {
                    ((u8x16 &)(ar[i]))[j] = i - j - 1;
                }
            }
        }
    }

    void skip_s() {
        for (; *cur <= ' '; cur++)
            ;
    }

    // [[gnu::noinline]]
    //
    u32
    fuck(__m128i vec) {
        __asm volatile("# LLVM-MCA-BEGIN fuck");

        uint16_t mask = _mm_movemask_epi8(_mm_cmpgt_epi8(_mm_set1_epi8(' ' + 1), vec));
        vec = _mm_sub_epi8(vec, _mm_set1_epi8('0'));
        int d = __builtin_ctz(mask);

        vec = _mm_shuffle_epi8(vec, ar[d]);
        vec = _mm_maddubs_epi16(_mm_setr_epi8(1, 10, 1, 10,
                                              25, 250, 25, 250,
                                              1, 10, 0, 0,
                                              0, 0, 0, 0),
                                vec);

        __m128i vec2 = _mm_madd_epi16(vec, _mm_setr_epi16(1, 100, 100, 10000,
                                                          0, 0, 0, 0));
        vec2 = _mm_add_epi32(vec2, _mm_srli_epi64(vec2, 30));

        u32 x = _mm_cvtsi128_si32(vec2) + _mm_cvtsi128_si32(_mm_bsrli_si128(vec, 8)) * u32(1e8);

        __asm volatile("# LLVM-MCA-END fuck");

        return x;
    }

#pragma GCC target("avx2")

    u32 read_u32() {
        skip_s();
        u32 val = 0;
        for (; *cur > ' '; cur++) {
            val = val * 10 + (*cur - '0');
        }
        return val;
    }

    u32 *read_all_u32(int cnt) {
        const int N = (end - cur) / 32;

        u32 *data = new u32[cnt];
        int cnt_read = 0;

        u64 sum = 0;
        for (int i = 0; i < N; i++) {
            __m256i val = _mm256_loadu_si256((__m256i *)(cur + i * 32));
            __m256i cmp = _mm256_cmpgt_epi8(_mm256_set1_epi8(' ' + 1), val);
            u32 mask = _mm256_movemask_epi8(cmp);

            while (mask) {
                // __asm volatile("# LLVM-MCA-BEGIN read_all");
                int t = __builtin_ctz(mask);
                mask ^= (1 << t);
                __m128i vec = _mm_loadu_si128((__m128i *)(cur + i * 32 + t + 1));
                u32 val = fuck(vec);
                data[cnt_read++] = val;
                // __asm volatile("# LLVM-MCA-END read_all");
            }
        }

        return data;
    }
};

Cum_Input input(open("input.txt", O_RDONLY));
// Cum_Input input(fileno(stdin));

int32_t main() {
    int n = input.read_u32();
    clock_t beg = clock();

    uint64_t hash = 0;

    // // #pragma GCC unroll 4
    // for (int i = 0; i < n; i++) {
    //     hash += input.read_u32();
    // }

    auto data = input.read_all_u32(n);
    for (int i = 0; i < n; i++) {
        hash += data[i];
    }

    std::cerr.precision(5);
    std::cerr << std::fixed;
    std::cerr << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << " " << n << " " << hash << " " << hash * 1.0 / n << "\n";

    return 0;
}
