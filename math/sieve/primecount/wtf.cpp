// src: https://codeforces.com/blog/entry/91632?#comment-1060256

#include <bits/stdc++.h>

using namespace std;
using ll = long long;

ll countprimes(ll n) {
    vector<ll> primes;

    // if (!primes.empty() && primes.back() > n) {
    //     auto i = upper_bound(primes.begin(), primes.end(), n);
    //     return distance(primes.begin(), i);
    // }

    ll v = sqrtl(n);
    vector<ll> higher(v + 2, 0), lower(v + 2, 0);
    vector<bool> used(v + 2, false);

    ll result = n - 1;
    for (ll p = 2; p <= v; p++) {
        lower[p] = p - 1;
        higher[p] = n / p - 1;
    }

    for (ll p = 2; p <= v; p++) {
        if (lower[p] == lower[p - 1]) continue;
        if (primes.empty() || p > primes.back()) primes.push_back(p);

        ll temp = lower[p - 1];
        result -= higher[p] - temp;
        ll psq = p * p;
        ll end = min(v, n / psq);
        ll j = 1 + (p & 1);

        for (ll i = p + j; i <= end + 1; i += j) {
            if (used[i]) continue;
            ll d = i * p;
            if (d <= v)
                higher[i] -= higher[d] - temp;
            else
                higher[i] -= lower[n / d] - temp;
        }

        for (ll i = v; i >= psq; i--) lower[i] -= lower[i / p] - temp;
        for (ll i = psq; i <= end; i += p * j) used[i] = true;
    }
    return result;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // int64_t n;
    long double n;
    std::cin >> n;
    std::cout << countprimes(n) << std::endl;

    return 0;
}
