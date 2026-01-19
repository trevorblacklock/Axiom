#ifndef OPERANDS_H_DEFINED
#define OPERANDS_H_DEFINED

namespace ax {

template<class Tp_>
constexpr auto factorial(Tp_ num) {
    auto result = num;
    while (num > 1) result *= (--num);
    return result;
}

} // namespace ax

#endif /* OPERANDS_H_DEFINED */
