#include <bits/stdc++.h>

std::vector<int> sa_is(const std::vector<int>& input) {
    if (input.size() <= 1) {
        return std::vector<int>(input.size(), 0);
    }

    int n = input.size();
    int mx = *std::max_element(input.begin(), input.end()) + 1;

    std::cerr << "n: " << n << "   mx: " << mx << "\n";

    // assert(mx <= n);
    assert(input.back() < *std::min_element(input.begin(), input.end() - 1));

    std::vector<bool> tp(n + 1);
    tp[n] = true;
    for (int i = n - 2; i >= 0; i--) {
        tp[i] = input[i] == input[i + 1] ? tp[i + 1] : (input[i] < input[i + 1]);
    }

    std::vector<int> suf(n + 1, -1);
    std::vector<int> bck(mx + 1);

    for (int val : input) bck[val] += 1;
    std::exclusive_scan(bck.begin(), bck.end(), bck.begin(), 1);

    std::vector<int> ptr(mx + 1);

    std::copy(bck.begin(), bck.end(), ptr.begin());
    suf[0] = n;
    for (int i = 1; i < n; i++) {
        if (!tp[i - 1] && tp[i]) {
            suf[--ptr[input[i] + 1]] = i;
        }
    }

    std::copy(bck.begin(), bck.end(), ptr.begin());
    for (int i = 0; i <= n; i++) {
        int p = suf[i];
        if (p > 0 && !tp[p - 1]) {
            suf[ptr[input[p - 1]]++] = p - 1;
        }
    }

    std::copy(bck.begin(), bck.end(), ptr.begin());
    for (int i = n; i >= 0; i--) {
        int p = suf[i];
        if (p > 0 && tp[p - 1]) {
            suf[--ptr[input[p - 1] + 1]] = p - 1;
        }
    }

    int m = 0;
    std::vector<int> lms_pos(n + 1, -1);
    for (int i = 1; i <= n; i++) {
        if (!tp[i - 1] && tp[i]) {
            lms_pos[i] = m++;
        }
    }

    std::vector<int> input2(m);
    for (int i = 0, cnt = 0, last = -1; i <= n; i++) {
        int p = suf[i];
        if (p > 0 && !tp[p - 1] && tp[p]) {
            if (last != -1) {
                if (last == n) {
                    cnt++;
                } else {
                    for (int j = 0;; j++) {
                        if (p + j == n || last + j == n || input[p + j] != input[last + j]) {
                            cnt++;
                            break;
                        }
                        if (j != 0 && !tp[p + j - 1] && tp[p + j]) {
                            break;
                        }
                    }
                }
            }
            last = p;
            input2[lms_pos[p]] = cnt;
        }
    }

    std::vector<int> suf2;
    if (*std::max_element(input2.begin(), input2.end()) == m - 1) {
        suf2.assign(m, -1);
        for (int i = 0; i < m; i++) {
            suf2[input2[i]] = i;
        }
    } else {
        suf2 = sa_is(input2);
    }

    std::vector<int> ind;
    ind.swap(lms_pos), ind.clear();
    for (int i = 1; i <= n; i++) {
        if (!tp[i - 1] && tp[i]) {
            ind.push_back(i);
        }
    }
    ind.push_back(n);

    std::fill(suf.begin(), suf.end(), -1);

    std::copy(bck.begin(), bck.end(), ptr.begin());
    suf[0] = n;
    for (int i = m - 1; i > 0; i--) {
        int p = ind[suf2[i]];
        suf[--ptr[input[p] + 1]] = p;
    }

    std::copy(bck.begin(), bck.end(), ptr.begin());
    for (int i = 0; i <= n; i++) {
        int p = suf[i];
        if (p > 0 && !tp[p - 1]) {
            suf[ptr[input[p - 1]]++] = p - 1;
        }
    }

    std::copy(bck.begin(), bck.end(), ptr.begin());
    for (int i = n; i >= 0; i--) {
        int p = suf[i];
        if (p > 0 && tp[p - 1]) {
            suf[--ptr[input[p - 1] + 1]] = p - 1;
        }
    }

    suf.erase(suf.begin());

    return suf;
}

std::vector<int> suffix_array(std::vector<int> input) {
    if (input.size() <= 1) {
        return std::vector<int>(input.size(), 0);
    }

    int n = input.size();
    int mx = *std::max_element(input.begin(), input.end()) + 1;
    if (mx > n) {
        std::vector<int> fuck(mx, -1);
        for (int i = 0; i < n; i++) {
            fuck[input[i]] = 0;
        }
        int mx2 = 0;
        for (int i = 0; i < mx; i++) {
            if (fuck[i] != -1) {
                fuck[i] = mx2++;
            }
        }
        for (int i = 0; i < n; i++) {
            input[i] = fuck[input[i]];
        }
        mx = mx2;
    }

    if (std::all_of(input.begin(), input.end() - 1, [&](int val) { return val > input.back(); })) {
        return sa_is(input);
    } else {
        for (auto& i : input) {
            i++;
        }
        input.push_back(0);
        auto suf = sa_is(input);
        suf.erase(suf.begin());
        return suf;
    }
}

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string s;
    std::cin >> s;

    clock_t beg = clock();
    std::vector<int> suf = suffix_array(std::vector<int>(s.begin(), s.end()));
    std::cerr << "work: " << (clock() - beg) * 1.0 / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;

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
