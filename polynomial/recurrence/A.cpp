#include <bits/stdc++.h>

using u32 = uint32_t;
using u64 = uint64_t;

constexpr u32 mod = 998'244'353;
constexpr u32 pr_root = 3;

u32 mul(u32 a, u32 b) {
    return u64(a) * b % mod;
}
u32 add(u32 a, u32 b) {
    return a + b - mod * (a + b >= mod);
}
void add_to(u32& a, u32 b) {
    a = add(a, b);
}
u32 power(u32 b, u32 e) {
    u32 r = 1;
    for (; e; e >>= 1) {
        if (e & 1) {
            r = mul(r, b);
        }
        b = mul(b, b);
    }
    return r;
}

std::vector<u32> fisting_ass(const std::vector<u32>& a) {
    int n = a.size();

    std::vector<u32> rec;

    int last_ind = -1;  // uh?
    std::vector<u32> last_rec;

    for (int i = 0; i < n; i++) {
        u32 val = 0;
        for (int j = 0; j < rec.size(); j++) {
            add_to(val, mul(rec[j], a[i - j - 1]));
        }
        if (val != a[i]) {
            if (last_ind == -1) {
                rec.resize(i + 1);
                last_ind = i;
                last_rec = {};
                continue;
            }

            int new_last_ind = i;
            std::vector<u32> new_last_rec = rec;

            u32 val2 = 0;
            for (int j = 0; j < last_rec.size(); j++) {
                add_to(val2, mul(last_rec[j], a[last_ind - j - 1]));
            }
            add_to(val2, mod - a[last_ind]);
            assert(val2 != 0);
            u32 f = mul(power(val2, mod - 2), add(a[i], mod - val));

            int dt = i - last_ind;
            rec.resize(std::max((int)rec.size(), (int)last_rec.size() + dt));

            add_to(rec[dt - 1], mod - f);
            for (int j = 0; j < last_rec.size(); j++) {
                add_to(rec[dt + j], mul(last_rec[j], f));
            }

            if (last_ind - (int)last_rec.size() < new_last_ind - (int)new_last_rec.size()) {
                last_ind = new_last_ind, last_rec = new_last_rec;
            }
        }
    }

    return rec;
}

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n;
    std::cin >> n;
    std::vector<u32> input(n);
    for (auto& i : input) {
        std::cin >> i;
    }

    std::vector<u32> rec = fisting_ass(input);
    std::cout << rec.size() << "\n";
    for (int i = 0; i < rec.size(); i++) {
        std::cout << rec[i] << " \n"[i + 1 == rec.size()];
    }

    return 0;
}
