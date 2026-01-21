#ifndef NUMERIC_H_DEFINED
#define NUMERIC_H_DEFINED

#include <numeric>
#include <ranges>

namespace ax {

namespace ranges {

template<std::ranges::range Rn_, class Tp_>
constexpr auto accumulate(const Rn_& range, Tp_ x0) {
    return std::accumulate(range.begin(), range.end(), x0);
}

template<std::ranges::range Rn_, class Tp_, class Fn_>
constexpr auto accumulate(const Rn_& range, Tp_ x0, Fn_ op) {
    return std::accumulate(range.begin(), range.end(), x0, op);
}

template<std::ranges::range Rn_>
constexpr auto sum(const Rn_& range) {
    using Tp_ = std::ranges::range_value_t<Rn_>;
    return accumulate(range, Tp_{0}, std::plus());
}

template<std::ranges::range Rn_>
constexpr auto product(const Rn_& range) {
    using Tp_ = std::ranges::range_value_t<Rn_>;
    return accumulate(range, Tp_{1}, std::multiplies());
}

} // namespace ranges

} // namespace ax

#endif /* NUMERIC_H_DEFINED */
