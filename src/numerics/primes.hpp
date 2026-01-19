#ifndef PRIMES_H_DEFINED
#define PRIMES_H_DEFINED

#include <cmath>

namespace ax {

template<class Tp_>
constexpr bool is_prime(Tp_ num) {
    if (num <= 1) return false;
    else if (num == 2) return true;
    else if (num % 2 == 0) return false;

    for (Tp_ x = 3; x <= static_cast<Tp_>(std::sqrt(num)); x += 2)
        if (num % x == 0) return false;

    return true;
}

} // namespace ax

#endif /* PRIMES_H_DEFINED */
