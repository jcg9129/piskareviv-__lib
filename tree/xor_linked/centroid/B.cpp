#include <bits/stdc++.h>

int32_t main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n;
    std::cin >> n;
    std::vector<std::pair<int, int>> edg(n - 1);
    for (auto& [u, v] : edg) {
        std::cin >> u >> v;
        u--, v--;
    }

    std::vector<int> ctr_depth(n);

    auto build = [&](auto build, std::vector<int> vtx, std::vector<std::pair<int, int>> edg) -> void {
        {
            for (int i = 0; i < order.size(); i++) {
                int v = order[i];
                max_dist[v] = -1, last_occ[v] = i;
            }
            for (int i = 0; i < order.size(); i++) {
                int v = order[i];
                int l = last_occ[v];
                int d = (l < i) ? (i - l) : ((int)order.size() - (l - i));
                max_dist[v] = std::max(max_dist[v], d);
                last_occ[v] = i;
            }
            int it = std::min_element(order.begin(), order.end(), [&](int a, int b) { return max_dist[a] < max_dist[b]; }) - order.begin();
            v = order[it];
            std::rotate(order.begin(), order.begin() + it + 1, order.end());
        }

        ctr_depth[v] = f == -1 ? 0 : ctr_depth[f] + 1;

        for (int i = 0, l = 0; i < order.size(); i++) {
            if (order[i] == v) {
                if (l != i) {
                    build(build, order[l], order.subspan(l + 1, i - (l + 1)), v);
                }
                l = i + 1;
            }
        }
    };

    build(build, 0, std::span<int>(dfs_order.data(), 2 * n - 2), -1);

    for (int i = 0; i < n; i++) {
        std::cout << char('A' + ctr_depth[i]) << " \n"[i + 1 == n];
    }

    return 0;
}
