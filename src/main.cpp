#include "ndarray.hpp"

#include <benchmark/benchmark.h>
#include <generator>
#include <print>
#include <coroutine>

// static void broadcast_test(benchmark::State& state) {
//     static auto a = ax::random::randn({10, 10});
//     static auto b = ax::random::randn({10, 10});
//     for (auto _ : state) {
//         auto result = ax::broadcast(a, b, std::divides());
//     }
// }
// BENCHMARK(broadcast_test);

// BENCHMARK_MAIN();

int main() {
    auto a = ax::random::randn({3, 3, 3});
    auto b = ax::random::randn({3, 3});
    auto c = ax::broadcast(a, b, std::divides());
    auto d = ax::broadcast(b, a, std::divides());

    auto v = {a, b, c, d};

    for (auto x : v) std::cout << x << std::endl;
}