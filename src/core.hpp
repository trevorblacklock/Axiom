#ifndef CORE_H_DEFINED
#define CORE_H_DEFINED

#include <iostream>

#if defined(NDEBUG)
#define ax_assert(condition, message)
#else
#define ax_assert(condition, message)                         \
    if (!(condition)) {                                       \
        std::cerr << __FILE__ << ':' << __LINE__ << ":\n"     \
                  << "Assertion failed: " << message << '\n'; \
        std::abort();                                         \
    }
#endif

#endif /* CORE_H_DEFINED */
