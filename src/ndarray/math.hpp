#ifndef NDARRAY_MATH_H_DEFINED
#define NDARRAY_MATH_H_DEFINED

#include <cmath>

#include "core.hpp"

namespace ax {

template<class Tp_>
constexpr auto max(const ndarray<Tp_>& array) {
    ax_assert(array.size() > 0, "Cannot find max of array of size 0!");
    auto ptr = array.data();
    auto max = *ptr;
    for (std::size_t i = 1; i < array.size(); ++i) {
        auto value = ptr[i];
        if (value > max) max = ptr[i];
    }
    return max;
}

template<class Tp_>
constexpr auto min(const ndarray<Tp_>& array) {
    ax_assert(array.size() > 0, "Cannot find max of array of size 0!");
    auto ptr = array.data();
    auto min = *ptr;
    for (std::size_t i = 1; i < array.size(); ++i) {
        auto value = ptr[i];
        if (value < min) min = value;
    }
    return min;
}

template<class Tp_>
constexpr auto minmax(const ndarray<Tp_>& array) {
    ax_assert(array.size() > 0, "Cannot find max of array of size 0!");
    auto ptr = array.data();
    auto min = *ptr; 
    auto max = *ptr;
    for (std::size_t i = 1; i < array.size(); ++i) {
        auto value = ptr[i];
        if (value < min) min = value;
        else if (value > max) max = value;
    }
    return std::make_pair(min, max);
}

template<class Tp_>
constexpr auto sin(const ndarray<Tp_>& array) {
    return array.apply(static_cast<double (*)(Tp_)>(std::sin));
}

template<class Tp_>
constexpr auto cos(const ndarray<Tp_>& array) {
    return array.apply(static_cast<double (*)(Tp_)>(std::cos));
}

template<class Tp_>
constexpr auto tan(const ndarray<Tp_>& array) {
    return array.apply(static_cast<double (*)(Tp_)>(std::tan));
}

template<class Tp_>
constexpr auto abs(const ndarray<Tp_>& array) {
    return array.apply(static_cast<Tp_ (*)(Tp_)>(std::abs));
}

template<class Tp_>
constexpr auto floor(const ndarray<Tp_>& array) {
    return array.apply(static_cast<double (*)(Tp_)>(std::floor));
}

template<class Tp_>
constexpr auto ceil(const ndarray<Tp_>& array) {
    return array.apply(static_cast<double (*)(Tp_)>(std::ceil));
}

template<class Tp_>
constexpr auto rint(const ndarray<Tp_>& array) {
    return array.apply(static_cast<double (*)(Tp_)>(std::rint));
}

template<class Tp_>
constexpr auto sum(const ndarray<Tp_>& array, int axis = 0) {
    auto& old_shape = array.extents().shape();
    auto new_shape = std::vector<std::size_t>(array.rank() - 1);
    for (auto&& [i, x] : 
        std::views::zip(
        std::views::iota(0ul, array.rank()) | 
        std::views::drop(axis),
        new_shape))
        x = old_shape[i];
    auto new_array = ndarray<Tp_>(new_shape);
    if (axis != 0) {
        auto range = std::views::iota(0ul, array.rank());
        auto axes = std::vector(range.begin(), range.end());
        std::swap(axes[axis], axes[0]);
        array = array.transpose(axes);
    }
    for (auto i : std::views::iota(0ul, old_shape[axis])) {
        new_array += array[i];
    }
    return new_array;
}

} // namespace ax

#endif /* NDARRAY_MATH_H_DEFINED */
