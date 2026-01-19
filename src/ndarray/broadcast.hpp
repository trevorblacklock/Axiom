#ifndef BROADCAST_H_DEFINED
#define BROADCAST_H_DEFINED

#include "../core.hpp"
#include "extents.hpp"

#include "immintrin.h"

#include <algorithm>
#include <vector>

namespace ax {

template<class Tp_>
class ndarray;

namespace detail {

template<
    class Tp1_, 
    class Tp2_,
    class Fn_,
    class Tp3_ = std::invoke_result_t<Fn_, Tp1_, Tp2_>>
requires (std::invocable<Fn_, Tp1_, Tp2_>)
constexpr void broadcast_apply(
    const Tp1_* data1,
    const Tp2_* data2,
    Tp3_* data3,
    Fn_&& func,
    const std::size_t* shape,
    const std::size_t* strides1,
    const std::size_t* strides2,
    std::size_t n_rank,
    std::size_t& idx3,
    std::size_t idx1 = 0,
    std::size_t idx2 = 0) {
    if (n_rank == 0) {
        data3[idx3++] = func(data1[idx1], data2[idx2]);
    }
    else {
        const auto offset1 = *strides1;
        const auto offset2 = *strides2;
        const auto rank = *shape;
        shape++;
        strides1++;
        strides2++;
        n_rank--;
        broadcast_apply(data1, data2, data3, func,
                shape, strides1, strides2, n_rank, 
                idx3, idx1, idx2);
        for (std::size_t i = 1; i < rank; ++i) {
            idx1 += offset1;
            idx2 += offset2;
            broadcast_apply(data1, data2, data3, func,
                shape, strides1, strides2, n_rank, 
                idx3, idx1, idx2);
        }
    } 
}

} // namespace detail

template<
    class Tp1_, 
    class Tp2_,
    class Fn_,
    class Tp3_ = std::invoke_result_t<Fn_, Tp1_, Tp2_>>
requires (std::invocable<Fn_, Tp1_, Tp2_>)
constexpr auto broadcast(
    const ndarray<Tp1_>& arr1,
    const ndarray<Tp2_>& arr2,
    Fn_&& func) {

    auto rank1 = arr1.rank();
    auto rank2 = arr2.rank();
    auto [rmin, rmax] = std::minmax(rank1, rank2);

    auto& extents1 = arr1.extents();
    auto& extents2 = arr2.extents();
    auto& shape1 = extents1.shape();
    auto& shape2 = extents2.shape();
    auto& strides1 = extents1.strides();
    auto& strides2 = extents2.strides();

    std::vector<std::size_t> shape3(rmax);
    std::copy(shape1.begin(), shape1.begin() + rmax - rmin, shape3.begin());

    auto ptr = &shape3.back();
    for (std::size_t i = 0; i < rmin; ++i) {
        auto x1 = shape1[rank1 - i - 1];
        auto x2 = shape2[rank2 - i - 1];
        ax_assert(x1 == x2 || x1 == 1 || x2 == 1,
            "Cannot broadcast arrays of incompatible shape!");
        *(ptr--) = std::max(x1, x2);
    }

    std::size_t idx3 = 0;
    auto arr3 = ndarray<Tp3_>(shape3);

    detail::broadcast_apply(arr1.data(), arr2.data(), arr3.data(), func,
        shape1.data(), strides1.data(), strides2.data(), arr3.rank(), idx3);

    return arr3;
}

} // namespace ax

#endif /* BROADCAST_H_DEFINED */
