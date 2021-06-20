#ifndef _RANDOM_H_
#define _RANDOM_H_

#include <random>
#include <atomic>

class Random
{
public:

    static std::atomic_uint ThreadSeed;
    thread_local static std::default_random_engine Engine;
};

#endif // _RANDOM_H_