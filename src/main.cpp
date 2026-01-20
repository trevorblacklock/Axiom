#include "ndarray.hpp"

#include <benchmark/benchmark.h>
#include <print>

static void broadcast_test(benchmark::State& state) {
    static auto b = ax::random::randn({100, 100}).transpose();
    for (auto _ : state) {
        auto result = ax::broadcast_scalar(10.0f, b, std::plus());
    }
}
BENCHMARK(broadcast_test);

BENCHMARK_MAIN();

// int main() {
//     auto x1 = ax::random::randn({3, 3, 3, 3});

//     std::println("{}", x1.extents().strides());
// }