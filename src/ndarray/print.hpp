#ifndef NDARRAY_PRINT_H_DEFINED
#define NDARRAY_PRINT_H_DEFINED

#include "core.hpp"

namespace ax {

namespace detail {

template<class Tp_>
void pretty_print(std::ostream& os, const ndarray<Tp_>& array, std::size_t ws) {
    if (array.rank() == 1) {
        os << '[';
        for (std::size_t i = 0; i < array.extent(); ++i) {
            os << array[i];
            if (i != array.extent() - 1) os << ", ";
        }
        os << ']';
    }
    else if (array.rank() == 2) {
        os << '[';
        for (std::size_t i = 0; i < array.extent(); ++i) {
            if (i != 0) os << std::string(ws, ' ');
            pretty_print(os, array.view(i), 0);
            if (i != array.extent() - 1) os << ",\n";
        }
        os << ']';
    }
    else {
        os << '[';
        for (std::size_t i = 0; i < array.extent(); ++i) {
            if (i != 0) os << std::string(ws, ' ');
            pretty_print(os, array.view(i), ws + 1);
            if (i != array.extent() - 1) os << ",\n\n";
        }
        os << ']';
    }
}

} // namespace detail

template<class Tp_>
std::ostream& operator<<(std::ostream& os, const ndarray<Tp_>& array) {
    os << "array(";
    detail::pretty_print(os, array, 7);
    return os << ")";
}

} // namespace ax

#endif /* NDARRAY_PRINT_H_DEFINED */
