#ifndef EXTENTS_H_DEFINED
#define EXTENTS_H_DEFINED

#include <numeric>
#include <vector>

#include "../ranges/numeric.hpp"

namespace ax {

enum class stride_type : int {
    row_major,
    column_major
};

namespace detail {

constexpr auto flat_index(
    const std::size_t* indices,
    const std::size_t* strides,
    const std::size_t size) noexcept {
    std::size_t result = 0;
    for (std::size_t i = 0; i < size; ++i)
        result += indices[i] * strides[i];
    return result;
}

template<std::integral It_, std::integral... Its_>
constexpr auto flat_index(
    std::size_t& result,
    const std::size_t* strideptr, 
    It_ first, 
    Its_... rest) {
    result += first * (*strideptr);
    if constexpr (sizeof...(Its_) >= 1) 
        flat_index(result, strideptr + 1, rest...);
}

} // namespace detail

template<stride_type St_ = stride_type::row_major>
class ndarray_extents {
public:
    static constexpr auto stride_type = St_;
    
    constexpr auto extent(std::size_t rank = 0) const {
        return shape_.at(rank);
    }

    constexpr auto size() const noexcept {
        return size_;
    }

    constexpr auto rank() const noexcept {
        return shape_.size();
    }

    constexpr auto& shape() const noexcept {
        return shape_;
    }

    constexpr auto& strides() const noexcept {
        return strides_;
    }

    constexpr void set_conguitity(bool contiguity) noexcept {
        contiguity_ = contiguity;
    }

    constexpr auto is_contiguous() const noexcept {
        return contiguity_;
    }

    template<std::integral... Its_>
    requires (sizeof...(Its_) >= 1)
    constexpr auto index(Its_... idxs) const {
        std::size_t index = 0;
        detail::flat_index(index, strides_.data(), idxs...);
        return index;
    }

    template<std::integral It_>
    constexpr auto index(const std::vector<It_>& idxs) const {
        return detail::flat_index(idxs.data(), strides_.data(), rank());
    }

    ndarray_extents(const std::vector<std::size_t>& shape) : 
        shape_(shape), strides_(shape.size()), size_(product(shape)) {
        update_strides();
    }

    ndarray_extents(
        const std::vector<std::size_t>& shape,
        std::size_t size) : 
        shape_(shape), strides_(shape.size()), size_(size) {
        update_strides();
    }

    ndarray_extents(
        const std::vector<std::size_t>& shape,
        const std::vector<std::size_t>& strides,
        std::size_t size, bool contiguity = true) :
        shape_(shape), strides_(strides), 
        size_(size), contiguity_(contiguity) {}

    constexpr auto& operator=(const ndarray_extents& other) {
        shape_ = other.shape();
        strides_ = other.strides();
        size_ = other.size();
        contiguity_ = other.is_contiguous();
        return *this;
    }

    ndarray_extents(const ndarray_extents& extents) { *this = extents; }
    ndarray_extents(ndarray_extents&& extents) = delete;

    template<std::integral... Sz_>
    constexpr ndarray_extents(Sz_... shape) :
        shape_({ static_cast<std::size_t>(shape)... }),
        strides_(sizeof...(Sz_)), size_((1 * ... * shape)) {
        update_strides();
    }

private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::size_t size_;
    bool contiguity_ = true;

    constexpr void update_strides() {
        std::exclusive_scan(shape_.rbegin(), shape_.rend(), 
            strides_.rbegin(), 1, std::multiplies());
    }
};

} // namespace ax

#endif /* EXTENTS_H_DEFINED */
