#ifndef NDARRAY_CORE_H_DEFINED
#define NDARRAY_CORE_H_DEFINED

#include <concepts>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <vector>

#include "broadcast.hpp"
#include "../core.hpp"
#include "extents.hpp"
#include "iterator.hpp"

namespace ax {

namespace detail {

template<class T_, std::size_t N_>
struct nested_init_list_impl {
    using subtype = typename nested_init_list_impl<T_, N_ - 1>::type;
    using type = std::initializer_list<subtype>;
};

template<class T_>
struct nested_init_list_impl<T_, 1> {
    using subtype = T_;
    using type = std::initializer_list<T_>;
};

template<class Tp_, std::size_t N_>
using nested_init_list = typename detail::nested_init_list_impl<Tp_, N_>::type;

template<class T_, std::size_t N_>
inline void data_from_nested_init_list_impl(
        const nested_init_list<T_, N_>& list, T_* ptr, 
        const std::vector<std::size_t>& shape,
        std::size_t& idx) {
    ax_assert(shape[shape.size() - N_] == list.size(),
            "Cannot represent non-rectangular data as ndarray!");
    for (const auto& x : list) {
        if constexpr (N_ == 1) ptr[idx++] = x;
        else data_from_nested_init_list_impl<T_, N_ - 1>(x, ptr, shape, idx);
    }
}

template<class T_, std::size_t N_>
inline void shape_from_nested_init_list_impl(
        const nested_init_list<T_, N_>& list,
        std::vector<std::size_t>& shape,
        std::size_t& size) {
    size *= list.size();
    shape.push_back(list.size());
    if constexpr (N_ >= 2) 
        shape_from_nested_init_list_impl<T_, N_ - 1>(*list.begin(), shape, size);
}

template<class T_, std::size_t N_>
inline void data_from_nested_init_list(
        const nested_init_list<T_, N_>& list, T_* ptr,
        const std::vector<std::size_t>& shape) {
    ax_assert(ptr != nullptr, "Trying to fill unallocated pointer");
    std::size_t idx = 0;
    detail::data_from_nested_init_list_impl<T_, N_>(list, ptr, shape, idx);
}

template<class T_, std::size_t N_>
inline void shape_from_nested_init_list(
        const nested_init_list<T_, N_>& list,
        std::vector<std::size_t>& shape,
        std::size_t& size) {
    size = 1; // Size has to be 1 when recursively multiplying
    detail::shape_from_nested_init_list_impl<T_, N_>(list, shape, size);
}

template<std::integral It_, std::integral... Its_>
inline void verify_indices(const std::size_t* shape, It_ idx, Its_... idxs) {
    ax_assert(static_cast<std::size_t>(idx) < *shape, "Index out of bounds!");
    if constexpr (sizeof...(Its_) > 0) verify_indices(++shape, idxs...);
}

constexpr void transpose_helper(
    const std::size_t* old_shape, 
    const std::size_t* old_strides,
    std::size_t* new_shape, 
    std::size_t* new_strides,
    std::size_t idx,
    const std::vector<std::size_t>& axes) {
    for (const auto& axis : axes) {
        new_shape[idx] = old_shape[axis];
        new_strides[idx++] = old_strides[axis];
    }
}

} // namespace detail

template<class Tp_>
class ndarray {
public:
    using data_type = std::remove_cv_t<Tp_>;
    using extent_type = ndarray_extents<stride_type::row_major>;

    constexpr auto extent(std::size_t rank = 0) const { 
        return extents_->extent(rank); 
    }

    constexpr auto size() const noexcept { 
        return extents_->size(); 
    }

    constexpr auto rank() const noexcept {
        return extents_->rank();
    }

    constexpr auto& accessor() const noexcept {
        return data_;
    }

    constexpr auto data() const noexcept {
        return data_.get();
    }

    constexpr auto& extents() const noexcept {
        return *extents_;
    }

    constexpr auto is_contiguous() const noexcept {
        return extents_->is_contiguous();
    }

    constexpr auto is_unique() const noexcept {
        return data_.unique();
    }

    constexpr auto& shape() const noexcept {
        return extents_->shape();
    }

    constexpr void fill(Tp_ value) {
        std::fill(data_.get(), data_.get() + size(), value);
    }

    constexpr auto begin() {
        return ndarray_iterator(this, 0);
    }

    constexpr auto begin() const {
        return ndarray_iterator(this, 0);
    }

    constexpr auto end() {
        return ndarray_iterator(this, extent());
    }

    constexpr auto end() const {
        return ndarray_iterator(this, extent());
    }

    template<
        class Fn_,
        class Tp2_ = std::invoke_result_t<Fn_, Tp_>>
    requires (std::invocable<Fn_, data_type>)
    constexpr auto apply(Fn_&& func) const {
        auto array = ndarray<Tp2_>(this->extents_->shape());
        auto new_data = array.data();
        auto old_data = this->data();
        for (std::size_t i = 0; i < array.size(); ++i)
            new_data[i] = func(old_data[i]);
        return array;
    }

    constexpr auto reshape(const std::vector<std::size_t>& shape) const {
        ax_assert(ranges::product(shape) == size(),
            "New shape does not match size of data!");
        if (is_contiguous()) return ndarray(data_, shape);
        // TODO: REDO THIS!!!
        auto& strides = extents_->strides();
        ndarray<data_type> array(data_type{}, shape);
        std::vector<std::size_t> index(this->rank());

        auto new_ptr = array.data();
        auto old_ptr = this->data();

        for (std::size_t i = 0; i < this->size(); ++i) {
            // Map 1d index to nd
            auto k = i;
            for (std::size_t j = 0; j < this->rank(); ++j) {
                auto idx = this->rank() - j - 1;
                if (k == 0) {
                    index[idx] = 0;
                    break;
                }
                else {
                    index[idx] = k / strides[idx];
                    k %= strides[idx];
                }      
            }
            new_ptr[i] = old_ptr[this->extents_->index(index)];
        }
        return array;
    }

    constexpr auto flatten() const {
        return reshape({ size() });
    }

    constexpr auto transpose(const std::vector<std::size_t>& axes) const {
        ax_assert(rank() >= 2, "Cannot transpose array less than rank 2!");
        ax_assert(axes.size() == rank(),
            "Number of axes does not match rank of array!");
        for (auto& x : axes) ax_assert(x < rank(),
            "Axis cannot exceed rank of array!");
        auto old_shape = extents_->shape().data();
        auto old_strides = extents_->strides().data();
        auto new_shape = std::vector<std::size_t>(rank());
        auto new_strides = std::vector<std::size_t>(rank());
        detail::transpose_helper(old_shape, old_strides, 
            new_shape.data(), new_strides.data(), 0, axes);
        auto new_extents = extent_type(new_shape, new_strides, 
            size(), false);
        auto array = ndarray(data_, new_extents);
        return array;
    }

    constexpr auto transpose() const {
        ax_assert(rank() >= 2, "Cannot transpose array less than rank 2!");
        auto idx = rank();
        auto shape = extents_->shape();
        auto strides = extents_->strides();
        std::swap(shape[idx - 1], shape[idx - 2]);
        std::swap(strides[idx - 1], strides[idx - 2]);
        auto new_extents = extent_type(shape, strides, 
            size(), false);
        auto array = ndarray(data_, new_extents);
        return array;
    }

    template<std::integral... Its_>
    requires (sizeof...(Its_) >= 1)
    constexpr auto view(Its_... idxs) const {
        auto flat_idx = extents_->index(idxs...);
        auto data_ptr = std::shared_ptr<data_type[]>(data_, &data_[flat_idx]);
        auto& old_shape = extents_->shape();
        auto& old_strides = extents_->strides();
        auto new_shape = std::vector(
            old_shape.begin() + sizeof...(Its_), 
            old_shape.end());
        auto new_strides = std::vector(
            old_strides.begin() + sizeof...(Its_),
            old_strides.end());
        auto new_extents = extent_type(new_shape, 
            new_strides, ranges::product(new_shape), is_contiguous());
        return ndarray(data_ptr, new_extents);
    }

    ndarray() = default;

    ndarray(const ndarray<Tp_>& other) { *this = other; }
    ndarray(ndarray<Tp_>&& other) noexcept { *this = std::move(other); }

    explicit ndarray(const Tp_* ptr, const std::vector<std::size_t>& shape) :
        extents_(std::make_unique<extent_type>(shape)),
        data_(std::shared_ptr<Tp_[]>(new Tp_[extents_->size()])) {
        std::copy(ptr, ptr + extents_->size(), data_.get());
    }

    explicit ndarray(const std::vector<std::size_t>& shape) :
        extents_(std::make_unique<extent_type>(shape)),
        data_(std::shared_ptr<Tp_[]>(new Tp_[extents_->size()])) {}

    explicit ndarray(Tp_ value, const std::vector<std::size_t>& shape) :
        extents_(std::make_unique<extent_type>(shape)),
        data_(std::shared_ptr<Tp_[]>(new Tp_[extents_->size()])) {
        fill(value);
    }

    explicit ndarray(const extent_type& extents) :
        extents_(std::make_unique<extent_type>(extents)),
        data_(std::shared_ptr<Tp_[]>(new Tp_[extents.size()])) {}

    template<std::size_t N_>
    using Nl_ = detail::nested_init_list<data_type, N_>;

    ndarray(const Nl_<1>& data) { init_from_nl<1>(data); }
    ndarray(const Nl_<2>& data) { init_from_nl<2>(data); }
    ndarray(const Nl_<3>& data) { init_from_nl<3>(data); }
    ndarray(const Nl_<4>& data) { init_from_nl<4>(data); }
    ndarray(const Nl_<5>& data) { init_from_nl<5>(data); }

    template<std::integral... Its_>
    requires (sizeof...(Its_) >= 1)
    constexpr auto& operator[](Its_... idxs) const {
        ax_assert(sizeof...(Its_) == rank(),
            "Incorrect number of indices!");
        detail::verify_indices(extents_->shape().data(), idxs...);
        auto flat_idx = extents_->index(idxs...);
        return data_[flat_idx];
    }

    constexpr auto& operator=(const ndarray<Tp_>& other) {
        auto size = other.size();
        extents_ = std::make_unique<extent_type>(other.extents());
        data_ = std::shared_ptr<Tp_[]>(new Tp_[size]);
        std::copy(other.data(), other.data() + size, data_.get());
        return *this;
    }
    
    constexpr auto& operator=(ndarray<data_type>&& other) noexcept {
        extents_ = std::make_unique<extent_type>(other.extents());
        data_ = other.accessor();
        return *this;
    }

    template<class Tp2_>
    constexpr auto operator+(const ndarray<Tp2_>& arr) {
        return broadcast(*this, arr, std::plus());
    }

    template<class Tp2_>
    constexpr auto operator+(Tp2_ scalar) {
        return broadcast_scalar<false>(scalar, *this, std::plus());
    }

    template<class Tp2_>
    constexpr auto operator-(const ndarray<Tp2_>& arr) {
        return broadcast(*this, arr, std::minus());
    }

    template<class Tp2_>
    constexpr auto operator-(Tp2_ scalar) {
        return broadcast_scalar<false>(scalar, *this, std::minus());
    }

    template<class Tp2_>
    constexpr auto operator*(const ndarray<Tp2_>& arr) {
        return broadcast(*this, arr, std::multiplies());
    }

    template<class Tp2_>
    constexpr auto operator*(Tp2_ scalar) {
        return broadcast_scalar<false>(scalar, *this, std::multiplies());
    }

    template<class Tp2_>
    constexpr auto operator/(const ndarray<Tp2_>& arr) {
        return broadcast(*this, arr, std::divides());
    }

    template<class Tp2_>
    constexpr auto operator/(Tp2_ scalar) {
        return broadcast_scalar<false>(scalar, *this, std::divides());
    }

    template<class Tp2_>
    constexpr auto operator+=(const ndarray<Tp2_>& arr) {
        broadcast_self(*this, arr, std::plus());
        return *this;
    }

    template<class Tp2_>
    constexpr auto operator+=(Tp2_ scalar) {
        broadcast_scalar_self(scalar, *this, std::plus());
        return *this;
    }

    template<class Tp2_>
    constexpr auto operator-=(const ndarray<Tp2_>& arr) {
        broadcast_self(*this, arr, std::minus());
        return *this;
    }

    template<class Tp2_>
    constexpr auto operator-=(Tp2_ scalar) {
        broadcast_scalar_self(scalar, *this, std::minus());
        return *this;
    }

    template<class Tp2_>
    constexpr auto operator*=(const ndarray<Tp2_>& arr) {
        broadcast_self(*this, arr, std::multiplies());
        return *this;
    }

    template<class Tp2_>
    constexpr auto operator*=(Tp2_ scalar) {
        broadcast_scalar_self(scalar, *this, std::multiplies());
        return *this;
    }

    template<class Tp2_>
    constexpr auto operator/=(const ndarray<Tp2_>& arr) {
        broadcast_self(*this, arr, std::divides());
        return *this;
    }

    template<class Tp2_>
    constexpr auto operator/=(Tp2_ scalar) {
        broadcast_scalar_self(scalar, *this, std::divides());
        return *this;
    }

private:
    std::unique_ptr<extent_type> extents_;
    std::shared_ptr<data_type[]> data_;

    template<std::size_t N_>
    constexpr void init_from_nl(const Nl_<N_>& data) {
        std::vector<std::size_t> shape; std::size_t size;
        detail::shape_from_nested_init_list<data_type, N_>(
            data, shape, size);
        data_ = std::shared_ptr<Tp_[]>(new Tp_[size]);
        extents_ = std::make_unique<extent_type>(shape, size);
        detail::data_from_nested_init_list<data_type, N_>(
            data, data_.get(), shape);
    }

    explicit ndarray(const std::shared_ptr<data_type[]>& data_ptr,
        const std::vector<std::size_t>& shape) : 
        extents_(std::make_unique<extent_type>(shape)), data_(data_ptr) {}

    explicit ndarray(const std::shared_ptr<data_type[]>& data_ptr,
        const extent_type& extents) : 
        extents_(std::make_unique<extent_type>(extents)), data_(data_ptr) {}
};

template<class Tp1_, class Tp2_>
constexpr auto operator+(Tp1_ scalar, const ndarray<Tp2_>& arr) {
    return broadcast_scalar<true>(scalar, arr, std::plus());
}

template<class Tp1_, class Tp2_>
constexpr auto operator-(Tp1_ scalar, const ndarray<Tp2_>& arr) {
    return broadcast_scalar<true>(scalar, arr, std::minus());
}

template<class Tp1_, class Tp2_>
constexpr auto operator*(Tp1_ scalar, const ndarray<Tp2_>& arr) {
    return broadcast_scalar<true>(scalar, arr, std::multiplies());
}

template<class Tp1_, class Tp2_>
constexpr auto operator/(Tp1_ scalar, const ndarray<Tp2_>& arr) {
    return broadcast_scalar<true>(scalar, arr, std::divides());
}

} // namespace ax

#endif /* NDARRAY_CORE_H_DEFINED */
