#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

int64_t primecount(int64_t n) {
    if (n <= 0) {
        return 0;
    }

    int64_t K = 1;
    while (K * K <= n) {
        K++;
    }

    std::vector<int64_t> primes;
    std::vector<bool> is_prime(K, true);
    for (int64_t i = 2; i < K; i++) {
        if (is_prime[i]) {
            primes.push_back(i);
            for (int64_t j = i * i; j < K; j += i) {
                is_prime[j] = false;
            }
        }
    }

    std::vector<int64_t> fuck;
    for (int64_t i = 1; i <= n; i = n / (n / i) + 1) {
        fuck.push_back(i);
    }

    int m = fuck.size();
    fuck.push_back(n + 1);
    std::vector<int64_t> dp(m + 1);
    std::vector<int> dlt(m + 1);
    for (int i = 0, j = primes.size(); i <= m; i++) {
        dp[i] = n / fuck[i];
        while (j > 0 && dp[i] < primes[j - 1]) {
            j--;
        }
        dlt[i] = j + (dp[i] != 0);
    }

    int64_t total = 0;
    std::vector<int> ptr(m);
    for (int k = 0; k < primes.size(); k++) {
        for (int i = 0, j = 0; i < m; i++) {
            j = std::max(j, ptr[i]);
            while (j + 1 <= m && fuck[i] * primes[k] >= fuck[j + 1]) {
                j += 1;
            }
            ptr[i] = j;
            int64_t dt = dp[j] - std::min<int64_t>(k + 1, dlt[j]);
            if (dt == 0) {
                break;
            }
            dp[i] -= dt;
        }
    }
    return dp[0] - 1;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // int64_t n;
    long double n;
    std::cin >> n;
    std::cout << primecount(n) << std::endl;

    return 0;
}
