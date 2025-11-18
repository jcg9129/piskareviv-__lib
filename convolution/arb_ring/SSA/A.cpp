#include <bits/allocator.h>
#pragma GCC target("pclmul")
#include <algorithm>
#include <cassert>
#include <iostream>

#include "conv.hpp"
#include "field_64.hpp"

size_t total = 0;

namespace meow {

using R = Field_64::Field;
using poly = std::vector<R>;

using conv::convolve, conv::convolve_transposed;

void remove_zeros(poly& p) {
    while (p.size() && p.back() == R()) p.pop_back();
}
R coeff(const poly& p, size_t ind) {
    return ind < p.size() ? p[ind] : R();
}

poly operator*(const poly& a, const poly& b) {
    size_t sz = a.size() + b.size();
    total += sz;
    if (sz >= 1000) {
        std::cerr << sz << " ";
    }
    poly res = convolve(a, b);
    remove_zeros(res);
    return res;
}
poly operator+(const poly& a, const poly& b) {
    poly c(std::max(a.size(), b.size()));
    for (size_t i = 0; i < a.size(); i++) c[i] += a[i];
    for (size_t i = 0; i < b.size(); i++) c[i] += b[i];
    remove_zeros(c);
    return c;
}
poly operator-(const poly& a) {
    poly c(a.size());
    for (size_t i = 0; i < a.size(); i++) c[i] = -a[i];
    remove_zeros(c);
    return c;
}
poly operator-(const poly& a, const poly& b) {
    poly c(std::max(a.size(), b.size()));
    for (size_t i = 0; i < a.size(); i++) c[i] += a[i];
    for (size_t i = 0; i < b.size(); i++) c[i] -= b[i];
    remove_zeros(c);
    return c;
}

poly mod_xk(poly a, size_t k) {
    if (a.size() > k) a.resize(k);
    return a;
}

poly inv_series(const poly& a, size_t sz) {
    poly b = {a[0]};
    if (sz > 1000) {
        std::cerr << "(" << sz << ") ";
    }
    auto get = [&](int a, int b) {
        int r = (b + a - 1) / a;
        return std::__lg(r - 1) + 1;
    };
    while (b.size() < sz) {
        size_t n = b.size();
        // while (n > 1 && get(n, sz) == get(n - 1, sz)) {
        //     b.pop_back(), n--;
        // }
        // b = b * (poly{R::n(2)} - mod_xk(a, std::min(sz, 2 * n)) * b);
        b = b * b * mod_xk(a, std::min(sz, 2 * n));
        b.resize(2 * n);
    }
    b.resize(sz);
    return b;
}

std::pair<poly, poly> divmod(const poly& a, const poly& b) {
    size_t n = a.size(), m = b.size();
    if (n < m) {
        return {poly{}, a};
    }
    size_t d = n - m + 1;
    poly ra(a.rbegin(), a.rbegin() + d);
    poly rb(b.rbegin(), b.rbegin() + std::min(d, m));
    poly q = mod_xk(ra * inv_series(rb, d), d);
    std::reverse(q.begin(), q.end());
    poly r = a - b * q;
    assert(r.size() < m);
    return {q, r};
}

poly operator%(const poly& a, const poly& b) {
    return divmod(a, b).second;
}

std::vector<R> evaluate_naive(const poly& a, const std::vector<R>& pts) {
    size_t m = pts.size();
    std::vector<poly> vec(2 * m);
    for (int i = 0; i < m; i++) {
        vec[m + i] = poly{-pts[i], R::n(1)};
    }
    for (int i = m - 1; i > 0; i--) {
        vec[i] = vec[2 * i] * vec[2 * i + 1];
    }
    vec[1] = a % vec[1];
    for (int i = 2; i < 2 * m; i++) {
        vec[i] = vec[i / 2] % vec[i];
    }
    std::vector<R> res(m);
    for (int i = 0; i < m; i++) {
        res[i] = coeff(vec[m + i], 0);
    }
    return res;
}

std::vector<R> evaluate(const poly& a, const std::vector<R>& pts) {
    // return evaluate_naive(a, pts);
    
    size_t n = a.size(), m = pts.size();
    std::vector<poly> vec(2 * m);
    for (int i = 0; i < m; i++) {
        vec[m + i] = poly{R::n(1), -pts[i]};
    }
    for (int i = m - 1; i > 0; i--) {
        vec[i] = vec[2 * i] * vec[2 * i + 1];
    }

    poly a2 = a;
    a2.resize(n + m - 1);
    vec[1] = convolve_transposed(inv_series(vec[1], n), a2);
     
    for (int i = 1; i < n; i++) {
        poly tmp = vec[2 * i];
        vec[2 * i] = convolve_transposed(vec[2 * i + 1], vec[i]);
        vec[2 * i + 1] = convolve_transposed(tmp, vec[i]);
    }
    std::vector<R> res(n);
    for (int i = 0; i < m; i++) {
        res[i] = coeff(vec[m + i], 0);
    }
    return res;
}

R eval(const poly& a, R pt) {
    R res = R();
    for (size_t i = 0; i < a.size(); i++) {
        res = res * pt + a.rbegin()[i];
    }
    return res;
}

};  // namespace meow

using meow::R, meow::poly;

void test_eval() {
    std::mt19937_64 rnd;

    int n = 1e5;

    poly p(n);
    std::vector<R> pts(n);
    for (auto& i : p) i = R(rnd());
    for (auto& i : pts) i = R(rnd());

    std::vector<R> vals = meow::evaluate(p, pts);

    for (int i = 0; i < 10; i++) {
        int ind = rnd() % n;
        assert(vals[ind] == meow::eval(p, pts[ind]));
    }
}

int main() {
    test_eval();
    std::cerr << total / 1e6 << "M" << std::endl;
}
