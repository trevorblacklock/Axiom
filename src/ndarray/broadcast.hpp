#ifndef BROADCAST_H_DEFINED
#define BROADCAST_H_DEFINED

#include "../core.hpp"
#include "concepts.hpp"
#include "extents.hpp"

#include <algorithm>
#include <functional>
#include <type_traits>
#include <vector>

namespace ax {

namespace detail {

template<class Tp2_, class Fn_, class Tp1_ = std::invoke_result_t<Fn_, Tp2_>>
inline void linear_walk(Tp1_* const       data1,
                        const Tp2_* const data2,
                        Fn_&&             func,
                        std::size_t       size) {
#pragma omp simd
    for (std::size_t i = 0; i < size; ++i)
        data1[i] = func(data2[i]);
}

template<class Tp2_,
         class Tp3_,
         class Fn_,
         class Tp1_ = std::invoke_result_t<Fn_, Tp2_, Tp3_>>
inline void linear_walk(Tp1_* const       data1,
                        const Tp2_* const data2,
                        const Tp3_* const data3,
                        Fn_&&             func,
                        std::size_t       size) {
#pragma omp simd
    for (std::size_t i = 0; i < size; ++i)
        data1[i] = func(data2[i], data3[i]);
}

template<class Tp2_, class Fn_, class Tp1_ = std::invoke_result_t<Fn_, Tp2_>>
inline void strided_walk(Tp1_* const        data1,
                         const Tp2_* const  data2,
                         Fn_&&              func,
                         const std::size_t* shape,
                         const std::size_t* strides,
                         std::size_t        rank,
                         std::size_t&       idx1,
                         std::size_t        idx2 = 0) {
    auto stride = *strides;
    auto dim    = *shape;
    if (rank == 1) {
#pragma omp simd
        for (std::size_t i = 0; i < dim; ++i) {
            data1[idx1++] = func(data2[idx2]);
            idx2 += stride;
        }
    } else {
        shape++;
        strides++;
        rank--;
        for (std::size_t i = 0; i < dim; ++i) {
            strided_walk(data1, data2, func, shape, strides, rank, idx1, idx2);
            idx2 += stride;
        }
    }
}

template<class Tp2_,
         class Tp3_,
         class Fn_,
         class Tp1_ = std::invoke_result_t<Fn_, Tp2_, Tp3_>>
inline void strided_walk(Tp1_* const        data1,
                         const Tp2_* const  data2,
                         const Tp3_* const  data3,
                         Fn_&&              func,
                         const std::size_t* shape,
                         const std::size_t* strides,
                         std::size_t        rank,
                         std::size_t&       idx12,
                         std::size_t        idx3 = 0) {
    auto stride = *strides;
    auto dim    = *shape;
    if (rank == 1) {
#pragma omp simd
        for (std::size_t i = 0; i < dim; ++i) {
            data1[idx12] = func(data2[idx12], data3[idx3]);
            idx3 += stride;
            idx12++;
        }
    } else {
        shape++;
        strides++;
        rank--;
        for (std::size_t i = 0; i < dim; ++i) {
            strided_walk(data1, data2, func, shape, strides, rank, idx12, idx3);
            idx3 += stride;
        }
    }
}

template<class Tp2_,
         class Tp3_,
         class Fn_,
         class Tp1_ = std::invoke_result_t<Fn_, Tp2_, Tp3_>>
inline void strided_walk(Tp1_* const        data1,
                         const Tp2_* const  data2,
                         const Tp3_* const  data3,
                         Fn_&&              func,
                         const std::size_t* shape,
                         const std::size_t* strides2,
                         const std::size_t* strides3,
                         std::size_t        rank,
                         std::size_t&       idx1,
                         std::size_t        idx2 = 0,
                         std::size_t        idx3 = 0) {
    auto stride2 = *strides2;
    auto stride3 = *strides3;
    auto dim     = *shape;
    if (rank == 1) {
#pragma omp simd
        for (std::size_t i = 0; i < dim; ++i) {
            data1[idx1++] = func(data2[idx2], data3[idx3]);
            idx2 += stride2;
            idx3 += stride3;
        }
    } else {
        shape++;
        strides2++;
        strides3++;
        rank--;
        for (std::size_t i = 0; i < dim; ++i) {
            strided_walk(data1, data2, data3, func, shape, strides2, strides3,
                         rank, idx1, idx2, idx3);
            idx2 += stride2;
            idx3 += stride3;
        }
    }
}

constexpr void check_broadcastable(const std::vector<std::size_t>& shape1,
                                   const std::vector<std::size_t>& shape2) {
    auto rank1 = shape1.size();
    auto rank2 = shape2.size();
    auto rmin  = std::min(rank1, rank2);

    for (std::size_t i = 0; i < rmin; ++i) {
        auto x1 = shape1[rank1 - i - 1];
        auto x2 = shape2[rank2 - i - 1];
        ax_assert(x1 == x2 || x1 == 1 || x2 == 1,
                  "Cannot broadcast arrays of incompatible shape!");
    }
}

constexpr auto check_broadcastable_return_shape(
    const std::vector<std::size_t>& shape1,
    const std::vector<std::size_t>& shape2) {
    auto rank1        = shape1.size();
    auto rank2        = shape2.size();
    auto [rmin, rmax] = std::minmax(rank1, rank2);
    auto shape3       = std::vector<std::size_t>(rmax);

    auto& ishape = rank1 > rank2 ? shape1 : shape2;
    for (std::size_t i = 0; i < rmax - rmin; ++i)
        shape3[i] = ishape[i];

    auto ptr = &shape3.back();
    for (std::size_t i = 0; i < rmin; ++i) {
        auto x1 = shape1[rank1 - i - 1];
        auto x2 = shape2[rank2 - i - 1];
        ax_assert(x1 == x2 || x1 == 1 || x2 == 1,
                  "Cannot broadcast arrays of incompatible shape!");
        *(ptr--) = std::max(x1, x2);
    }

    return shape3;
}

constexpr auto make_new_shape_strides(
    const std::vector<std::size_t>& shape1,
    const std::vector<std::size_t>& shape2,
    const std::vector<std::size_t>& strides1,
    const std::vector<std::size_t>& strides2) {
    auto size1  = shape1.size();
    auto size2  = shape2.size();
    auto result = std::vector<std::size_t>(2 * size1 + 2 * size2);

    std::size_t rank1 = 0;
    std::size_t rank2 = 0;
    std::size_t idx   = 0;
    for (auto&& [dim, stride] : std::views::zip(shape1, strides1)) {
        if (dim == 1)
            continue;
        result[idx]         = dim;
        result[idx + size1] = stride;
        rank1++;
        idx++;
    }
    idx = 2 * size1;
    for (auto&& [dim, stride] : std::views::zip(shape2, strides2)) {
        if (dim == 1)
            continue;
        result[idx]         = dim;
        result[idx + size2] = stride;
        rank2++;
        idx++;
    }
    return std::make_tuple(result, rank1, rank2);
}

template<class Tp1_, class Tp2_, class Tp3_, class Fn_>
constexpr void broadcast_helper(Tp1_* const        data1,
                                const Tp2_* const  data2,
                                const Tp3_* const  data3,
                                Fn_&&              func,
                                const std::size_t* shape2,
                                const std::size_t* shape3,
                                const std::size_t* strides2,
                                const std::size_t* strides3,
                                std::size_t        rank2,
                                std::size_t        rank3,
                                std::size_t        size2,
                                std::size_t        size3) {
    auto [rmin, rmax] = std::minmax(rank2, rank3);
    auto rdiff        = rmax - rmin;

    auto is_strided_same
        = detail::is_strided_same(strides2, strides3, rank2, rank3);
    auto is_same_size = size2 == size3;

    if (is_strided_same && is_same_size) {
        detail::linear_walk(data1, data2, data3, func, size2);
    } else if (is_same_size) {
        std::size_t idx = 0;
        detail::strided_walk(data1, data2, data3, func, shape2, strides2,
                             strides3, rank2, idx);
    } else if (is_strided_same) {
        auto [stride, n, tsize, idata, bdata]
            = rank2 > rank3
                ? std::make_tuple(strides2[rdiff - 1], size2 / size3, size3,
                                  data2, data3)
                : std::make_tuple(strides3[rdiff - 1], size3 / size2, size2,
                                  data3, data2);
        std::size_t idx = 0;
        for (std::size_t i = 0; i < n * stride; i += stride) {
            detail::linear_walk(&data1[tsize], idata + i, bdata, func, tsize);
            idx += tsize;
        }
    } else {
        auto [stride, n, tsize, idata, bdata, istrides, bstrides, bshape, brank]
            = rank2 > rank3
                ? std::make_tuple(strides2[rdiff - 1], size2 / size3, size3,
                                  data2, data3, strides2 + rdiff, strides3,
                                  shape3, rank3)
                : std::make_tuple(strides3[rdiff - 1], size3 / size2, size2,
                                  data3, data2, strides3 + rdiff, strides2,
                                  shape2, rank2);
        std::size_t idx = 0;
        for (std::size_t i = 0; i < n * stride; i += stride)
            detail::strided_walk(data1, idata + i, bdata, func, bshape,
                                 istrides, bstrides, brank, idx);
    }
}

} // namespace detail

template<class Tp1_,
         ndarray_like Tp2_,
         class Fn_,
         class Dt1_ = Tp1_,
         class Dt2_ = Tp2_::data_type,
         class Dt3_ = std::invoke_result_t<Fn_, Dt1_, Dt2_>,
         class Tp3_ = ndarray<Dt3_>>
    requires(!ndarray_like<Tp1_>) // Ensure no ndarray as scalar
constexpr void broadcast(Tp1_ scalar, Tp2_&& arr1, Fn_&& func) {
    detail::linear_walk(arr1.data(), arr1.data(), std::bind_front(func, scalar),
                        arr1.size());
}

template<class Tp1_,
         ndarray_like Tp2_,
         class Fn_,
         class Dt1_ = Tp1_,
         class Dt2_ = Tp2_::data_type,
         class Dt3_ = std::invoke_result_t<Fn_, Dt1_, Dt2_>,
         class Tp3_ = ndarray<Dt3_>>
    requires(!ndarray_like<Tp1_>) // Ensure no ndarray as scalar
constexpr auto broadcast(Tp1_ scalar, const Tp2_& arr1, Fn_&& func) {
    auto arr2 = Tp3_(arr1.shape());
    if (arr1.is_contiguous()) {
        detail::linear_walk(arr2.data(), arr1.data(),
                            std::bind_front(func, scalar), arr2.size());
    } else {
        auto&       strides = arr1.strides();
        auto&       shape   = arr1.shape();
        std::size_t idx     = 0;
        detail::strided_walk(arr2.data(), arr1.data(),
                             std::bind_front(func, scalar), shape.data(),
                             strides.data(), arr2.rank(), idx);
    }
    return arr2;
}

template<ndarray_like Tp1_,
         ndarray_like Tp2_,
         class Fn_,
         class Dt1_ = Tp1_::data_type,
         class Dt2_ = Tp2_::data_type,
         class Dt3_ = std::invoke_result_t<Fn_, Dt1_, Dt2_>,
         class Tp3_ = ndarray<Dt3_>>
constexpr void broadcast(Tp1_&& arr1, const Tp2_& arr2, Fn_&& func) {
    ax_assert(arr1.size() >= arr2.size(),
              "Cannot self broadcast array of larger size onto smaller array!");
    // Check for scalar broadcasting
    if (arr2.size() == 1) {
        broadcast(arr2.data()[0], std::move(arr1), func);
        return;
    }

    detail::check_broadcastable(arr1.shape(), arr2.shape());

    // Process the shape and strides to remove any dimension and corresponding
    // stride that equals 1. If the corresponding rank is zero it means the
    // array only contains a single value.
    auto [vec, rank1, rank2] = detail::make_new_shape_strides(
        arr1.shape(), arr2.shape(), arr1.strides(), arr2.strides());

    auto shape1 = &vec[0];
    auto shape2 = &vec[2 * arr1.rank()];

    auto strides1 = &vec[arr1.rank()];
    auto strides2 = &vec[2 * arr1.rank() + arr2.rank()];

    detail::broadcast_helper(arr1.data(), arr1.data(), arr2.data(), func,
                             shape1, shape2, strides1, strides2, rank1, rank2,
                             arr1.size(), arr2.size());
}

template<ndarray_like Tp1_,
         ndarray_like Tp2_,
         class Fn_,
         class Dt1_ = Tp1_::data_type,
         class Dt2_ = Tp2_::data_type,
         class Dt3_ = std::invoke_result_t<Fn_, Dt1_, Dt2_>,
         class Tp3_ = ndarray<Dt3_>>
constexpr auto broadcast(const Tp1_& arr1, const Tp2_& arr2, Fn_&& func) {
    // Check for scalar broadcasting
    if (arr1.size() == 1)
        return broadcast(arr1.data()[0], arr2, func);
    else if (arr2.size() == 1)
        return broadcast(arr2.data()[0], arr1, func);

    auto shape3
        = detail::check_broadcastable_return_shape(arr1.shape(), arr2.shape());

    // Process the shape and strides to remove any dimension and corresponding
    // stride that equals 1. If the corresponding rank is zero it means the
    // array only contains a single value.
    auto [vec2, rank1, rank2] = detail::make_new_shape_strides(
        arr1.shape(), arr2.shape(), arr1.strides(), arr2.strides());

    auto shape1 = &vec2[0];
    auto shape2 = &vec2[2 * arr1.rank()];

    auto strides1 = &vec2[arr1.rank()];
    auto strides2 = &vec2[2 * arr1.rank() + arr2.rank()];

    auto arr3 = Tp3_(shape3);

    detail::broadcast_helper(arr3.data(), arr1.data(), arr2.data(), func,
                             shape1, shape2, strides1, strides2, rank1, rank2,
                             arr1.size(), arr2.size());
    return arr3;
}

} // namespace ax

#endif /* BROADCAST_H_DEFINED */
