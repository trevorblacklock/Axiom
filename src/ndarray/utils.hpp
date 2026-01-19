#ifndef NDARRAY_UTILS_H_DEFINED
#define NDARRAY_UTILS_H_DEFINED

#include <cmath>

#include "core.hpp"

namespace ax {

template<class Tp_>
constexpr auto diagonal(const std::vector<Tp_>& data) {
    auto size = data.size();
    auto array = ndarray<Tp_>(size, size);
    for (const auto [i, x] : data | std::views::enumerate) array[i, i] = x;
    return array;
}

template<class Tp_>
constexpr auto diagonal(const std::initializer_list<Tp_>& data) {
    return diagonal(std::vector(data));
}

template<class Tp_>
constexpr auto eye(std::size_t size) {
    auto array = ndarray<Tp_>(size, size);
    for (std::size_t i = 0; i < size; ++i) array[i, i] = static_cast<Tp_>(1);
    return array;
}

template<class Tp_>
constexpr auto linspace(Tp_ start, Tp_ end, std::size_t num) {
    auto array = ndarray<Tp_>(num);
    auto step = (end - start) / (num - 1);
    for (std::size_t i = 0; i < num; ++i) {
        array[i] = start;
        start += step;
    }
    return array;
}

template<class Tp_>
constexpr auto arange(Tp_ start, Tp_ end, Tp_ step) {
    std::size_t num = std::ceil(std::abs(end - start) / step);
    auto array = ndarray<Tp_>(num);
    for (std::size_t i = 0; i < num; ++i) {
        array[i] = start;
        start += step;
    }
    return array;
}

} // namespace ax

#endif /* NDARRAY_UTILS_H_DEFINED */
