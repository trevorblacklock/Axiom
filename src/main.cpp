#include "ndarray.hpp"
#include "ndarray/broadcast.hpp"

#include <benchmark/benchmark.h>
#include <print>

static void broadcast_test(benchmark::State& state) {
    static auto a = ax::random::randn({100, 100, 100, 100});
    static auto b = ax::random::randn({100, 100, 100, 100});
    for (auto _ : state) {
        auto result = ax::broadcast(a, b, std::multiplies());
    }
}
BENCHMARK(broadcast_test);

BENCHMARK_MAIN();

// int main() {
//     auto x1 = ax::random::randn({3, 3});
//     auto x2 = x1.transpose();

//     auto x3 = ax::broadcast(x1, x2, std::plus());

//     std::cout << x1 << std::endl;
//     std::cout << x2 << std::endl;
//     std::cout << x3 << std::endl;
// }