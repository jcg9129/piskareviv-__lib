#include <algorithm>
#include <array>
#include <cassert>
#include <span>
#include <vector>

namespace conv {

template <typename R>
std::vector<R> convolve_naive(const std::vector<R>& a, const std::vector<R>& b) {
    if (a.empty() || b.empty()) {
        return {};
    }
    std::vector<R> c(a.size() + b.size() - 1);
    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < b.size(); j++) {
            c[i + j] += a[i] * b[j];
        }
    }
    return c;
}

static constexpr size_t pow3(int k) {
    constexpr auto ar = [&] {
        std::array<size_t, 32> ar;
        ar[0] = 1;
        for (size_t i = 1; i < ar.size(); i++) {
            ar[i] = ar[i - 1] * 3;
        }
        return ar;
    }();
    return ar[k];
}

template <typename R>
struct Meow {
    // template <typename R>
    struct L {
        R x, y;

        L(R x = R(), R y = R()) : x(x), y(y) { ; }

        L operator-() const { return L(-x, -y); }
        friend L operator+(const L& a, const L& b) { return L(a.x + b.x, a.y + b.y); }
        friend L operator-(const L& a, const L& b) { return L(a.x - b.x, a.y - b.y); }
        friend L operator*(const L& a, const L& b) {
            R yy = a.y * b.y;
            return L(a.x * b.x - yy, a.x * b.y + a.y * b.x - yy);
        }

        L& operator+=(const L& other) { return *this = *this + other; }
        L& operator-=(const L& other) { return *this = *this - other; }
        L& operator*=(const L& other) { return *this = *this * other; }

        L mul_by_w() const { return L(-y, x - y); }
        L mul_by_w2() const { return L(y - x, -x); }

        L conj() const { return L(x - y, -y); }

        bool operator==(const L& other) const { return x == other.x && y == other.y; }

        // friend std::ostream& operator<<(std::ostream& out, const L& val) { return out << "(" << val.x << "," << val.y << ")"; }
    };

    using K = L;
    using W = int;

    void mul_by_w(std::span<K> s, int w) {
        if (w == 1) {
            for (K& i : s) {
                i = i.mul_by_w();
            }
        } else if (w == 2) {
            for (K& i : s) {
                i = i.mul_by_w2();
            }
        } else {
            assert(w == 0);
        }
    }

    template <bool inverse>
    void butterfly_x3(int lg, W w, std::span<K> a, std::span<K> b, std::span<K> c) {
        int sh1 = w % pow3(lg);
        int tw1 = w / pow3(lg);
        int sh2 = (2 * w) % pow3(lg);
        int tw2 = (2 * w) / pow3(lg) % 3;

        if (!inverse) {
            std::rotate(b.begin(), b.end() - sh1, b.end());
            std::rotate(c.begin(), c.end() - sh2, c.end());

            mul_by_w(b.subspan(0, sh1), (1 + tw1) % 3);
            mul_by_w(c.subspan(0, sh2), (1 + tw2) % 3);
            mul_by_w(b.subspan(sh1), tw1);
            mul_by_w(c.subspan(sh2), tw2);

            for (int i = 0; i < pow3(lg); i++) {
                K x = a[i], y = b[i], z = c[i];
                a[i] = x + y + z;
                b[i] = x + y.mul_by_w() + z.mul_by_w2();
                c[i] = x + y.mul_by_w2() + z.mul_by_w();
            }
        } else {
            for (int i = 0; i < pow3(lg); i++) {
                K x = a[i], y = b[i], z = c[i];
                a[i] = x + y + z;
                b[i] = x + y.mul_by_w2() + z.mul_by_w();
                c[i] = x + y.mul_by_w() + z.mul_by_w2();
            }

            mul_by_w(b.subspan(0, sh1), (5 - tw1) % 3);
            mul_by_w(c.subspan(0, sh2), (5 - tw2) % 3);
            mul_by_w(b.subspan(sh1), (3 - tw1) % 3);
            mul_by_w(c.subspan(sh2), (3 - tw2) % 3);

            std::rotate(b.begin(), b.begin() + sh1, b.end());
            std::rotate(c.begin(), c.begin() + sh2, c.end());
        }
    }

    template <bool inverse>
    void aux_transform(int lg, int lg2, std::span<K> a, bool conj = false) {
        auto rec = [&](auto rec, int k, int i, int w) -> void {
            size_t n = pow3(k);
            size_t m = pow3(lg2);
            auto recurse = [&] {
                for (int j = 0; j < 3; j++) {
                    rec(rec, k - 1, i + n * j, (w / 3 + pow3(lg2) * j) % pow3(lg2 + 1));
                }
            };
            if (inverse && k >= lg2 + 1) {
                recurse();
            }
            for (int j = 0; j < n; j += m) {
                butterfly_x3<inverse>(lg2, w / 3, a.subspan(i + j, m), a.subspan(i + j + n, m), a.subspan(i + j + 2 * n, m));
            }
            if (!inverse && k >= lg2 + 1) {
                recurse();
            }
        };
        rec(rec, lg - 1, 0, (1 + conj) * pow3(lg2));
    }

    std::vector<std::vector<K>> vec;
    
    std::array<std::span<K>, 2> get(int lg) {
        while (vec.size() <= lg) {
            int k = vec.size();
            vec.emplace_back(2 * pow3(k));
        }
        std::span<K> sp = vec[lg];
        std::fill(sp.begin(), sp.end(), K());
        return {sp.subspan(0, pow3(lg)), sp.subspan(pow3(lg))};
    }

    void convolve_aux(std::span<K> a, std::span<K> b) {
        int lg = 0;
        size_t n = 1;
        while (n < a.size()) n *= 3, lg++;
        assert(n == a.size());

        constexpr int LG = 2;
        if (lg <= LG) {
            // std::array<K, pow3(LG)> out;
            // out.fill(K());
            K out[pow3(LG)];
            std::fill(out, out + n, K());
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    K p = a[i] * b[j];
                    if (i + j < n) {
                        out[i + j] += p;
                    } else {
                        out[i + j - n] += p.mul_by_w();
                    }
                }
            }
            std::copy(out, out + n, a.begin());
            return;
        }

        int lg2 = (lg + 1) / 2;
        size_t m = pow3(lg2);

        std::span<K> xa = a, xb = b;
        //std::vector<K> ya(n), yb(n);
        auto [ya, yb] = get(lg);
        for (size_t i = 0; i < n; i++) {
            ya[i] = a[i].conj(), yb[i] = b[i].conj();
        }

        aux_transform<false>(lg, lg2, xa);
        aux_transform<false>(lg, lg2, xb);
        aux_transform<false>(lg, lg2, ya, true);
        aux_transform<false>(lg, lg2, yb, true);

        for (int i = 0; i < n; i += m) {
            convolve_aux(std::span(xb).subspan(i, m), std::span(xa).subspan(i, m));
            convolve_aux(std::span(yb).subspan(i, m), std::span(ya).subspan(i, m));
        }

        aux_transform<true>(lg, lg2, xb);
        aux_transform<true>(lg, lg2, yb, true);

        for (K& k : yb) {
            k = k.conj();
        }

        std::fill(a.begin(), a.end(), K());

        for (int i = 0; i < n; i += m) {
            for (int j = 0; j < m; j++) {
                // X == A  (mod x^m - w  )
                // X == B  (mod x^m - w^2)
                // X = A / (w - w^2) * (x^m - w^2)
                //   + B / (w^2 - w) * (x^m - w  )
                //
                // 1 / (a + bw) =
                // (a + b * conj(w)) / ((a + bw) * (a + b * conj(w))) =
                // (a - b - bw) / (a^2 - ab + b^2)
                //
                // w - w^2 = 1 + 2w = a + bw
                // a = 1, b = 2   ->  1 / (a + bw) = (-1 - 2w) / (1 - 2 + 4) = (-1 - 2w) / 3

                K x1 = xb[i + j].mul_by_w2() + yb[i + j].mul_by_w();
                K x2 = xb[i + j] + yb[i + j];

                a[i + j] += x1;
                if (i + j + m < n) {
                    a[i + j + m] += x2;
                } else {
                    a[i + j + m - n] += x2.mul_by_w();
                }
            }
        }
    }
};


template <typename R>
std::vector<R> convolve(const std::vector<R>& a, const std::vector<R>& b) {
    if (a.empty() || b.empty()) {
        return {};
    }
    int lg = 0;
    while (2 * pow3(lg) < a.size() + b.size() - 1) lg++;
    size_t n = pow3(lg);
    
    if (a.size() * b.size() <= n * lg * 10) {
        return convolve_naive(a, b);
    }
  
    Meow<R> mw;
    using K = Meow<R>::K;

    std::vector<K> f(n), g(n);
    for (int i = 0; i < a.size() && i < 2 * n; i++) {
        (i < n ? f[i].x : f[i - n].y) = a[i];
    }
    for (int i = 0; i < b.size() && i < 2 * n; i++) {
        (i < n ? g[i].x : g[i - n].y) = b[i];
    }

    mw.convolve_aux(f, g);

    std::vector<R> ans(a.size() + b.size() - 1);
    for (size_t i = 0; i < ans.size() && i < 2 * n; i++) {
        ans[i] = i < n ? f[i].x : f[i - n].y;
    }
    return ans;
}

template <typename R>
std::vector<R> convolve_transposed(const std::vector<R>& a, const std::vector<R>& b) {
    size_t n = a.size(), m = b.size();
    assert(n != 0);
    if (m < n) {
        return {};
    }
    size_t d = m - n + 1;
    
    if (0) {
        std::vector<R> res(d);
        for (int i = 0; i < d; i++) {
            R r = R();
            for (int j = 0; j < n; j++) {
                r += a[j] * b[i + j];
            }
            res[i] = r; 
        }
        return res;
    }
    
    std::vector<R> p = convolve(std::vector<R>(a.rbegin(), a.rend()), b);
    p.erase(p.begin(), p.begin() + (n - 1)); 
    p.resize(d);
    return p; 
}

};  // namespace conv
