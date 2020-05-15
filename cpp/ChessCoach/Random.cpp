#include "Random.h"

#include <chrono>

std::atomic_uint Random::ThreadSeed;

thread_local std::default_random_engine Random::Engine(
    std::random_device{}() +
    static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()) +
    ++ThreadSeed);