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
    static auto a = ax::random::randn({10, 10});

    std::vector<std::ranges::iota_view<std::size_t, std::size_t>> ranges(a.rank());
    for (auto&& [i, x] : a.extents().shape() | std::views::enumerate) 
        ranges[i] = std::views::iota(0ul, x);

    for (const auto& x : ranges) for (const auto& y : x) 
        std::cout << y << std::endl;
}