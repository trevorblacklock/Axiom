#ifndef RANDOM_H_DEFINED
#define RANDOM_H_DEFINED

#include <random>

#include "core.hpp"

namespace ax::random {

static std::random_device device_;
static std::mt19937 generator_(device_());

static inline auto randn(
    const std::vector<std::size_t>& shape,
    double mean = 0.0f, 
    double stdev = 1.0f) {
    std::normal_distribution gaussian(mean, stdev);
    auto array = ndarray<double>(shape);
    auto ptr = array.data();
    for (std::size_t i = 0; i < array.size(); ++i) 
        ptr[i] = gaussian(generator_);
    return array;
}

static inline auto randint(
    const std::vector<std::size_t>& shape,
    int low, int high) {
    ax_assert(low < high, "Low value cannot exceed or be equal to high value!");
    std::uniform_int_distribution uniform(low, high);
    auto array = ndarray<int>(shape);
    auto ptr = array.data();
    for (std::size_t i = 0; i < array.size(); ++i)
        ptr[i] = uniform(generator_);
    return array;
}

} // namespace ax::random

#endif /* RANDOM_H_DEFINED */
