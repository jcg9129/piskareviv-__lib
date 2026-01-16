#include <iostream>
#pragma GCC optimize("O3")
#pragma GCC target("avx2,lzcnt,bmi,bmi2")
#include <bits/stdc++.h>
#include <immintrin.h>

template <typename u_tp = uint64_t>
class TreeBitset {
   private:
    static constexpr size_t B = sizeof(u_tp) * 8;

    std::vector<u_tp> my_data;
    std::vector<u_tp*> data;
    size_t n, lg;

   public:
    TreeBitset(size_t n = 0) {
        assign(n);
    }

    void assign(size_t n) {
        this->n = n;
        size_t m = n + 2;
        std::vector<size_t> vec;
        while (m > 1) {
            m = (m - 1) / B + 1;
            vec.push_back(m);
        }
        std::reverse(vec.begin(), vec.end());

        lg = vec.size();
        data.resize(vec.size());
        size_t sum = std::accumulate(vec.begin(), vec.end(), size_t(0));
        my_data.assign(sum, 0);
        for (size_t i = 0, s = 0; i < lg; s += vec[i], i++) {
            data[i] = my_data.data() + s;
        }

        for (size_t i = 0, k = lg; k--; i /= B) {
            data[k][i / B] |= u_tp(1) << i % B;
        }
        for (size_t i = n + 1, k = lg; k--; i /= B) {
            data[k][i / B] |= u_tp(1) << i % B;
        }
    }

    size_t size() const {
        return n;
    }

    void clear() {
        my_data.assign(my_data.size(), 0);
        for (size_t i = 0, k = lg; k--; i /= B) {
            data[k][i / B] |= u_tp(1) << i % B;
        }
        for (size_t i = n + 1, k = lg; k--; i /= B) {
            data[k][i / B] |= u_tp(1) << i % B;
        }
    }

    // i must be in [0, n)
    bool insert(size_t i) {
        i++;
        if ((data[lg - 1][i / B] >> i % B) & 1) {
            return false;
        }
        for (size_t k = lg; k--; i /= B) {
            data[k][i / B] |= u_tp(1) << i % B;
        }
        return true;
    }

    // i must be in [0, n)
    bool erase(size_t i) {
        i++;
        if (!((data[lg - 1][i / B] >> i % B) & 1)) {
            return false;
        }
        data[lg - 1][i / B] ^= u_tp(1) << i % B;
        i /= B;
        for (size_t k = lg - 1; k > 0 && !data[k][i]; k--, i /= B) {
            data[k - 1][i / B] ^= u_tp(1) << i % B;
        }
        return true;
    }

    // i must be in [0, n)
    bool contains(size_t i) const {
        i++;
        return (data[lg - 1][i / B] >> i % B) & 1;
    }

    // i must be in [0, n]
    // smallest element greater than or equal to i, n if doesn't exist
    size_t find_next(size_t i) const {
        i++;
        size_t k = lg - 1;

        for (; !u_tp(data[k][i / B] >> i % B); k--) {
            i = i / B + 1;
        }

        for (; k < lg; k++) {
            u_tp mask = u_tp(data[k][i / B] >> i % B) << i % B;
            size_t ind = std::countr_zero(mask);
            i = (i / B * B + ind) * B;
        }
        i /= B;
        return i - 1;
    }

    // i must be in [0, n)
    // largest element less than or equal to i, n if doesn't exist
    size_t find_prev(size_t i) const {
        i++;
        size_t k = lg - 1;
        for (; !u_tp(data[k][i / B] << (B - i % B - 1)); k--) {
            i = i / B - 1;
        }

        for (; k < lg; k++) {
            u_tp mask = u_tp(data[k][i / B] << (B - i % B - 1)) >> (B - i % B - 1);
            assert(mask);
            size_t ind = B - 1 - std::countl_zero(mask);
            i = (i / B * B + ind) * B + (B - 1);
        }
        i /= B;
        if (i == 0) {
            return n;
        }
        return i - 1;
    }
};

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, q;
    std::cin >> n >> q;
    std::vector<int> input(n);
    for (auto& i : input) {
        std::cin >> i;
        i--;
    }
    std::vector<std::pair<int64_t, std::array<int, 3>>> quer(q);
    for (int i = 0; i < q; i++) {
        auto& [ord, ar] = quer[i];
        auto& [l, r, id] = ar;
        std::cin >> l >> r;
        l--;
        id = i;

        // constexpr int H_LG = 19, H_MAX = 1 << H_LG;
        // auto hilbert = [&](int x, int y) -> int64_t {
        //     long long d = 0;
        //     for (int s = H_MAX >> 1; s; s >>= 1) {
        //         bool rx = x & s, ry = y & s;
        //         d = d << 2 | rx * 3 ^ static_cast<int>(ry);
        //         if (!ry) {
        //             if (rx) x = H_MAX - x, y = H_MAX - y;
        //             std::swap(x, y);
        //         }
        //     }
        //     return d;
        // };
        // ord = hilbert(l, r);

        const uint64_t mask = 0x5555'5555'5555'5555;
        ord = _pdep_u64(l, mask) | (_pdep_u64(r, mask) << 1);
    }

    TreeBitset set1(n);
    TreeBitset set2(n);
    std::vector<int> cnt(n);
    auto add_cum = [&](int val, int dlt) {
        cnt[val] += dlt;
        if (dlt == +1) {
            if (cnt[val] == 1) {
                set2.insert(val);
            }
        } else {
            if (cnt[val] == 0) {
                set2.erase(val);
            }
        }
    };

    auto add = [&](int val, int dlt) {
        if (dlt == +1) {
            set1.insert(val);
        } else {
            set1.erase(val);
        }
        int it1 = n, it2 = n;
        if (val != 0) {
            it1 = set1.find_prev(val - 1);
            if (it1 != n) {
                add_cum(val - it1, dlt);
            }
        }
        {
            it2 = set1.find_next(val + 1);
            if (it2 != n) {
                add_cum(it2 - val, dlt);
            }
        }
        if (it1 != n && it2 != n) {
            add_cum(it2 - it1, -dlt);
        }
    };

    std::sort(quer.begin(), quer.end(), [&](auto a, auto b) {
        return a.first < b.first;
    });

    std::vector<int> ans(q);
    int L = 0, R = 0;
    for (const auto& [ord, ar] : quer) {
        const auto& [l, r, id] = ar;
        while (R < r) {
            add(input[R++], +1);
        }
        while (l < L) {
            add(input[--L], +1);
        }
        while (r < R) {
            add(input[--R], -1);
        }
        while (L < l) {
            add(input[L++], -1);
        }
        int res = set2.find_next(0);
        ans[id] = res;
    }

    for (int i = 0; i < q; i++) {
        std::cout << ans[i] << " \n"[i == q - 1];
    }
    return 0;
}
