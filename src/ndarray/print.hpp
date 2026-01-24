#ifndef NDARRAY_PRINT_H_DEFINED
#define NDARRAY_PRINT_H_DEFINED

#include "core.hpp"

#include <format>
#include <sstream>

namespace ax {

namespace detail {

template<class Tp_>
void pretty_print(std::ostream& os, const ndarray<Tp_>& array, std::size_t ws) {
    if (array.rank() == 1) {
        os << '[';
        for (std::size_t i = 0; i < array.extent(); ++i) {
            os << array[i];
            if (i != array.extent() - 1)
                os << ", ";
        }
        os << ']';
    } else if (array.rank() == 2) {
        os << '[';
        for (std::size_t i = 0; i < array.extent(); ++i) {
            if (i != 0)
                os << std::string(ws, ' ');
            pretty_print(os, array.view(i), 0);
            if (i != array.extent() - 1)
                os << ",\n";
        }
        os << ']';
    } else {
        os << '[';
        for (std::size_t i = 0; i < array.extent(); ++i) {
            if (i != 0)
                os << std::string(ws, ' ');
            pretty_print(os, array.view(i), ws + 1);
            if (i != array.extent() - 1)
                os << ",\n\n";
        }
        os << ']';
    }
}

} // namespace detail

template<class Tp_>
inline std::ostream& operator<<(std::ostream& os, const ndarray<Tp_>& array) {
    os << "array(";
    detail::pretty_print(os, array, 7);
    return os << ")";
}

} // namespace ax

template<typename Ch_>
struct basic_ostream_formatter :
    std::formatter<std::basic_string_view<Ch_>, Ch_> {
    template<typename Tp_, typename It_>
    auto format(const Tp_&                           value,
                std::basic_format_context<It_, Ch_>& ctx) const {
        std::basic_stringstream<Ch_> ss;
        ss << value;
        return std::formatter<std::basic_string_view<Ch_>, Ch_>::format(
            ss.view(), ctx);
    }
};

template<class Tp_>
struct std::formatter<ax::ndarray<Tp_>> : basic_ostream_formatter<char> {};

#endif /* NDARRAY_PRINT_H_DEFINED */
