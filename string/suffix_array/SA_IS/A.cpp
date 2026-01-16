#include <bits/stdc++.h>

template <typename T>
void count_sort(int n, auto&& f, const std::vector<T>& src, std::vector<T>& dest) {
    std::vector<int> cnt(n);
    for (auto& val : src) {
        cnt[f(val)] += 1;
    }
    std::exclusive_scan(cnt.begin(), cnt.end(), cnt.begin(), 0);
    dest.resize(src.size());
    for (auto& val : src) {
        dest[cnt[f(val)]++] = val;
    }
}

template <typename T>
void count_sort(int n, auto&& f, std::vector<T>& vec) {
    std::vector<T> vec2;
    count_sort(n, f, vec, vec2), vec.swap(vec2);
}

std::vector<int> suffix_array(std::vector<int> input) {
    if (input.size() <= 1) {
        return std::vector<int>(input.size(), 0);
    }

    int mx = *std::max_element(input.begin(), input.end()) + 1;
    int n = input.size();

    std::cerr << "n: " << n << "  mx: " << mx << "\n";

    std::vector<int> bck(mx + 1);
    for (int val : input) {
        bck[val] += 1;
    }
    std::exclusive_scan(bck.begin(), bck.end(), bck.begin(), 1);

    std::vector<bool> cum(n + 1);
    cum[n] = true, cum[n - 1] = false;
    for (int i = n - 2; i >= 0; i--) {
        cum[i] = input[i] == input[i + 1] ? cum[i + 1] : (input[i] < input[i + 1]);
    }

    std::vector<int> ind;
    for (int i = 0; i + 1 <= n; i++) {
        if (!cum[i] && cum[i + 1]) {
            ind.push_back(i + 1);
        }
    }
    int m = ind.size();
    ind.push_back(n);

    std::vector<std::array<int, 3>> all;
    for (int i = 0; i < m; i++) {
        for (int j = ind[i]; j < ind[i + 1]; j++) {
            all.push_back({i, j, input[j]});
        }
        if (ind[i + 1] == n) {
            all.push_back({i, n, 0});
        } else {
            all.push_back({i, ind[i + 1], input[ind[i + 1]]});
        }
    }

    auto pos = [&](const std::array<int, 3>& ar) { return ar[1] - ind[ar[0]]; };

    count_sort(mx, [&](auto x) { return x[2]; }, all);
    count_sort(n, [&](auto x) { return pos(x); }, all);

    for (int i = 0, j = 0; i < all.size(); i = j) {
        while (j < all.size() && pos(all[i]) == pos(all[j])) j++;
        for (int k = i, last = all[i][2], val = 0; k < j; k++) {
            val += last != all[k][2], last = all[k][2], all[k][2] = val;
        }
    }
    std::reverse(all.begin(), all.end());
    std::vector<int> ord, sym(m, -1);
    for (int i = 0, j = 0; i < all.size(); i = j) {
        while (j < all.size() && pos(all[i]) == pos(all[j])) j++;
        // std::reverse(ord.begin(), ord.end());
        for (int k = i, last = all[i][2], val = 0; k < j; k++) {
            if (sym[all[k][0]] == -1) {
                ord.push_back(all[k][0]);
            }
            sym[all[k][0]] = all[k][2];
        }
        // std::reverse(ord.begin(), ord.end());
        count_sort(ord.size(), [&](int ind) { return sym[ind]; }, ord);
    }
    // ord.insert(ord.begin(), m - 1);

    std::vector<int> cock(m);
    for (int i = 1; i < m; i++) {
        int a = ord[i - 1], b = ord[i];
        int dlt = 0;
        if (int l1 = (ind[a + 1] - ind[a]) + 1, l2 = (ind[b + 1] - ind[b]) + 1; l1 != l2) {
            dlt = 1;
        } else {
            for (int j = 0; j < l1; j++) {
                if ((ind[a] + j != n) != (ind[b] + j != n)) {
                    dlt = 1;
                    break;
                }
                if (input[ind[a] + j] != input[ind[b] + j]) {
                    dlt = 1;
                    break;
                }
            }
        }
        cock[ord[i]] = cock[ord[i - 1]] + dlt;
    }

    std::vector<int> fuck = suffix_array(cock);

    std::vector<int> suf(n + 1, -1);

    std::vector<int> bck2 = bck;
    suf[0] = n;
    for (int i = 0; i < m - 1; i++) {
        int p = ind[fuck.rbegin()[i]];
        suf[--bck2[input[p] + 1]] = p;
    }

    std::vector<int> bck3 = bck;
    for (int i = 0; i <= n; i++) {
        int p = suf[i];
        if (p != -1 && p > 0 && !cum[p - 1]) {
            suf[bck3[input[p - 1]]++] = p - 1;
        }
    }
    for (int i = 0; i <= n; i++) {
        if (suf[i] != -1 && cum[suf[i]]) {
            suf[i] = -1;
        }
    }

    std::vector<int> bck4 = bck;
    for (int i = n; i > 0; i--) {
        int p = suf[i];
        if (p != -1 && p > 0 && cum[p - 1]) {
            suf[--bck4[input[p - 1] + 1]] = p - 1;
        }
    }
    suf[0] = n;

    suf.erase(suf.begin());

    // for (int i = 0; i + 1 < n; i++) {
    //     auto a = std::vector<int>(input.begin() + suf[i], input.end());
    //     auto b = std::vector<int>(input.begin() + suf[i + 1], input.end());
    //     assert(a < b);
    // }
    return suf;
}

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string s;
    std::cin >> s;
    s += '$';

    clock_t beg = clock();
    std::vector<int> suf = suffix_array(std::vector<int>(s.begin(), s.end()));
    std::cerr << "work: " << (clock() - beg) * 1.0 / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;
    // suf.erase(suf.begin());

    // std::cerr << std::endl;
    // for (int i = 0; i < s.size(); i++) {
    //     std::cerr << s.substr(suf[i]) << std::endl;
    //     if (i > 0) {
    //         assert(s.substr(suf[i - 1]) < s.substr(suf[i]));
    //     }
    // }

    for (int i = 0; i < suf.size(); i++) {
        std::cout << suf[i] + 1 << " \n"[i + 1 == suf.size()];
    }

    return 0;
}
