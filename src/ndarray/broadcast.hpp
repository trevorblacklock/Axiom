#ifndef BROADCAST_H_DEFINED
#define BROADCAST_H_DEFINED

#include "../core.hpp"
#include "extents.hpp"

#include <algorithm>
#include <type_traits>
#include <vector>

namespace ax {

template<class Tp_>
class ndarray;

namespace detail {

// TODO: FIX THIS!!
constexpr auto is_similar_strides(
    const std::vector<std::size_t>& strides1,
    const std::vector<std::size_t>& strides2) {
    std::size_t i1 = strides1.size() - 1;
    for (std::size_t i2 = strides2.size() - 1; i2-- > 0;) {
        auto x1 = strides1[i1];
        auto x2 = strides2[i2];
        if (x1 == x2 || x2 == 1) continue;
        else if (x1 == 1) { i1--; i2++; continue; }
        else return false;
    }
    return true;
}

template<
    class Tp1_,
    class Tp2_,
    class Fn_,
    class Tp3_ = std::invoke_result_t<Fn_, Tp1_, Tp2_>>
requires (std::invocable<Fn_, Tp1_, Tp2_>)
constexpr void broadcast_apply_contiguous(
    const Tp1_* const data1,
    const Tp2_* const data2,
    Tp3_* const data3,
    Fn_&& func,
    std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        data3[i] = func(data1[i], data2[i]);
    }
}

template<
    class Tp1_, 
    class Tp2_,
    class Fn_,
    class Tp3_ = std::invoke_result_t<Fn_, Tp1_, Tp2_>>
requires (std::invocable<Fn_, Tp1_, Tp2_>)
constexpr void broadcast_apply_general(
    const Tp1_* const data1,
    const Tp2_* const data2,
    Tp3_* const data3,
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
        broadcast_apply_general(data1, data2, data3, func,
                shape, strides1, strides2, n_rank, 
                idx3, idx1, idx2);
        for (std::size_t i = 1; i < rank; ++i) {
            idx1 += offset1;
            idx2 += offset2;
            broadcast_apply_general(data1, data2, data3, func,
                shape, strides1, strides2, n_rank, 
                idx3, idx1, idx2);
        }
    } 
}

template<
    bool Sf_,
    class Tp1_, 
    class Tp2_,
    class Fn_,
    class Tp3_ = std::invoke_result_t<Fn_, Tp1_, Tp2_>>
requires (std::invocable<Fn_, Tp1_, Tp2_>)
constexpr void broadcast_apply_scalar(
    const Tp1_ scalar,
    const Tp2_* const data1,
    Tp3_* data2,
    Fn_&& func,
    std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        if constexpr (Sf_ == true)
            data2[i] = func(scalar, data1[i]);
        else
            data2[i] = func(data1[i], scalar);
    }
}

} // namespace detail

template<
    class Tp1_,
    class Tp2_,
    class Fn_>
requires (std::invocable<Fn_, Tp1_, Tp2_>
    && std::is_convertible_v<std::invoke_result_t<Fn_, Tp1_, Tp2_>, Tp2_>)
constexpr void broadcast_scalar_self(
    const Tp1_& scalar,
    Tp2_& arr,
    Fn_&& func) {
    detail::broadcast_apply_scalar<false>(scalar,
        arr.data(), arr.data(), func, arr.size());
}

template<
    bool Sf_,
    class Tp1_,
    class Tp2_,
    class Fn_,
    class Tp3_ = std::invoke_result_t<Fn_, Tp1_, Tp2_>>
requires (std::invocable<Fn_, Tp1_, Tp2_>)
constexpr auto broadcast_scalar(
    const Tp1_& scalar,
    const ndarray<Tp2_>& arr1,
    Fn_&& func) {
    ndarray<Tp3_> arr2(arr1.extents());
    detail::broadcast_apply_scalar<Sf_>(scalar, 
        arr1.data(), arr2.data(), func, arr2.size());
    return arr2;
}

template<
    class Tp1_,
    class Fn_>
requires (std::invocable<Fn_, Tp1_, Tp1_>)
constexpr void broadcast_self(
    const ndarray<Tp1_>& arr1,
    const ndarray<Tp1_>& arr2,
    Fn_&& func) {
    // Check for scalar operations
    if (arr2.size() == 1)
        broadcast_scalar_self(*arr2.data(), arr1, func);

    auto& shape1 = arr1.shape();
    auto& shape2 = arr2.shape();
    auto& strides1 = arr1.extents().strides();
    auto& strides2 = arr2.extents().strides();
    
    auto rank1 = arr1.rank();
    auto rank2 = arr2.rank();
    auto [rmin, rmax] = std::minmax(rank1, rank2);

    for (std::size_t i = 0; i < rmin; ++i) {
        auto x1 = shape1[rank1 - i - 1];
        auto x2 = shape2[rank2 - i - 1];
        ax_assert(x1 == x2 || x1 == 1 || x2 == 1,
            "Cannot broadcast arrays of incompatible shape!");
    }

    if (arr1.is_contiguous() && arr2.is_contiguous()) {
        if (arr1.size() == arr2.size()) {
            detail::broadcast_apply_contiguous(arr1.data(), arr2.data(),
                arr1.data(), func, arr1.size());
        } 
        else {
            auto stride_idx = rmax - rmin - 1;
            if (rank1 > rank2) {
                auto stride = strides1[stride_idx];
                for (std::size_t i = 0; i < arr1.size(); i += stride) {
                    detail::broadcast_apply_contiguous(arr1.data() + i, 
                        arr2.data(), arr1.data() + i, func, arr2.size());
                }          
            }
            else {
                auto stride = strides2[stride_idx];
                for (std::size_t i = 0; i < arr2.size(); i += stride) {
                    detail::broadcast_apply_contiguous(arr1.data(),
                        arr2.data() + i, arr1.data() + i, func, arr1.size());
                }
            }
        }
    }
    else {
        std::size_t idx3 = 0;
        if (arr1.size() == arr2.size()) {
            detail::broadcast_apply_general(arr1.data(), arr2.data(), 
                arr1.data(), func, shape1.data(), strides1.data(), 
                strides2.data(), arr1.rank(), idx3);
        }
        else {  
            auto rdiff = rmax - rmin;
            auto stride_idx = rdiff - 1;
            auto offset_shape = shape1.data() + rdiff;
            if (rank1 > rank2) {
                auto stride = strides1[stride_idx];
                auto offset_strides = strides1.data() + rdiff;
                for (std::size_t i = 0; i < arr1.size(); i += stride) {
                    detail::broadcast_apply_general(arr1.data() + i,
                        arr2.data(), arr1.data(), func, offset_shape,
                        offset_strides, strides2.data(), arr2.rank(), idx3);
                }
            }
            else {
                auto stride = strides2[stride_idx];
                auto offset_strides = strides2.data() + rdiff;
                for (std::size_t i = 0; i < arr2.size(); i += stride) {
                    detail::broadcast_apply_general(arr1.data(), 
                    arr2.data() + i, arr1.data(), func, offset_shape,
                        strides1.data(), offset_strides, arr1.rank(), idx3);
                }
            }
        }
    }
}

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
    // Quickly check for scalar operations
    if (arr1.size() == 1) 
        return broadcast_scalar<true>(*arr1.data(), arr2, func);
    else if (arr2.size() == 1) 
        return broadcast_scalar<false>(*arr2.data(), arr1, func);

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
    for (std::size_t i = 0; i < rmax - rmin; ++i) {
        shape3[i] = std::max(shape1[i], shape2[i]);
    }

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

    if (arr1.is_contiguous() && arr2.is_contiguous()) {
        if (arr1.size() == arr2.size()) {
            detail::broadcast_apply_contiguous(arr1.data(), arr2.data(),
                arr3.data(), func, arr1.size());
        } 
        else {
            auto stride_idx = rmax - rmin - 1;
            if (rank1 > rank2) {
                auto stride = strides1[stride_idx];
                for (std::size_t i = 0; i < arr1.size(); i += stride) {
                    detail::broadcast_apply_contiguous(arr1.data() + i, 
                        arr2.data(), arr3.data() + i, func, arr2.size());
                }          
            }
            else {
                auto stride = strides2[stride_idx];
                for (std::size_t i = 0; i < arr2.size(); i += stride) {
                    detail::broadcast_apply_contiguous(arr1.data(),
                        arr2.data() + i, arr3.data() + i, func, arr1.size());
                }
            }
        }
    }
    else {
        if (arr1.size() == arr2.size()) {
            detail::broadcast_apply_general(arr1.data(), arr2.data(), 
                arr3.data(), func, shape1.data(), strides1.data(), 
                strides2.data(), arr3.rank(), idx3);
        }
        else {  
            auto rdiff = rmax - rmin;
            auto stride_idx = rdiff - 1;
            auto offset_shape = shape3.data() + rdiff;
            if (rank1 > rank2) {
                auto stride = strides1[stride_idx];
                auto offset_strides = strides1.data() + rdiff;
                for (std::size_t i = 0; i < arr1.size(); i += stride) {
                    detail::broadcast_apply_general(arr1.data() + i,
                        arr2.data(), arr3.data(), func, offset_shape,
                        offset_strides, strides2.data(), arr2.rank(), idx3);
                }
            }
            else {
                auto stride = strides2[stride_idx];
                auto offset_strides = strides2.data() + rdiff;
                for (std::size_t i = 0; i < arr2.size(); i += stride) {
                    detail::broadcast_apply_general(arr1.data(), 
                    arr2.data() + i, arr3.data(), func, offset_shape,
                        strides1.data(), offset_strides, arr1.rank(), idx3);
                }
            }
        }
    }
    return arr3;
}

} // namespace ax

#endif /* BROADCAST_H_DEFINED */
