#include "poly.hpp"

constexpr u32 mod = 998'244'353;
using mint = MintT<mod>;
using poly = polynomial::Poly<mod>;

int32_t main() {
#ifdef LOCAL
    freopen("test.in", "r", stdin);
#endif
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int d;
    int64_t k;
    d = 1e5, k = 1e18;
    std::cin >> d >> k;
    poly a(d, 1), b(d, 1);
    for (auto& i : a) {
        std::cin >> i;
    }
    for (auto& i : b) {
        std::cin >> i;
    }
    b.insert(b.begin(), -1);

    std::cout << polynomial::kth_linear(k, a, b) << std::endl;

    return 0;
}
