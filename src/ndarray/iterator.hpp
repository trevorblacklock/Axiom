#ifndef NDARRAY_ITERATOR_H_DEFINED
#define NDARRAY_ITERATOR_H_DEFINED

#include <cstdlib>
#include <iterator>

namespace ax {

template<class Tp_>
class ndarray_iterator {
 public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = Tp_;
    using difference_type   = std::ptrdiff_t;
    using pointer           = void;
    using reference         = Tp_;

    explicit ndarray_iterator() = default;

    explicit ndarray_iterator(Tp_* view, std::size_t idx)
        : view_(view),
          idx_(idx) {
    }

    reference operator*() const {
        return view_->view(idx_);
    }

    reference operator[](difference_type n) const {
        return view_->view(idx_ + n);
    }

    auto& operator++() {
        ++idx_;
        return *this;
    }

    auto& operator--() {
        --idx_;
        return *this;
    }

    auto operator++(int) {
        auto tmp = *this;
        ++idx_;
        return tmp;
    }

    auto operator--(int) {
        auto tmp = *this;
        --idx_;
        return tmp;
    }

    auto& operator+=(difference_type n) {
        idx_ += n;
        return *this;
    }

    auto& operator-=(difference_type n) {
        idx_ -= n;
        return *this;
    }

    auto operator+(difference_type n) const {
        return base_iterator(view_, idx_ + n);
    }

    auto operator-(difference_type n) const {
        return base_iterator(view_, idx_ - n);
    }

    auto operator==(const ndarray_iterator<Tp_>& other) const {
        return idx_ == other.index();
    }

    auto operator!=(const ndarray_iterator<Tp_>& other) const {
        return idx_ != other.index();
    }

    auto operator<(const ndarray_iterator<Tp_>& other) const {
        return idx_ < other.index();
    }

    auto operator<=(const ndarray_iterator<Tp_>& other) const {
        return idx_ <= other.index();
    }

    auto operator>(const ndarray_iterator<Tp_>& other) const {
        return idx_ > other.index();
    }

    auto operator>=(const ndarray_iterator<Tp_>& other) const {
        return idx_ >= other.index();
    }

    inline auto index() const {
        return idx_;
    }

 private:
    Tp_*        view_;
    std::size_t idx_;
};

} // namespace ax

#endif /* NDARRAY_ITERATOR_H_DEFINED */
