#ifndef NDARRAY_CONCEPTS_H_DEFINED
#define NDARRAY_CONCEPTS_H_DEFINED

#include <type_traits>

namespace ax {

template<class Tp_>
class ndarray;

template<class>
struct is_ndarray : std::false_type {};

template<class Tp_>
struct is_ndarray<ndarray<Tp_>> : std::true_type {};

template<class Tp_>
constexpr auto is_ndarray_v = is_ndarray<Tp_>::value;

template<class Tp_>
concept ndarray_like = is_ndarray_v<Tp_>;

} // namespace ax

#endif /* CONCEPTS_H_DEFINED */
