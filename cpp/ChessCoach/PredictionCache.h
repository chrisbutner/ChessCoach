#ifndef _PREDICTIONCACHE_H_
#define _PREDICTIONCACHE_H_

#include <vector>

#include <Stockfish/types.h>

#include "Network.h"
#include "Platform.h"

struct alignas(128) PredictionCacheEntry
{
    // Positions with more moves don't fit in the cache and so shouldn't be probed/stored.
    static const int MaxMoveCount = 56;

    Key key;                                            // 8 bytes
    float value;                                        // 4 bytes
    int age;                                            // 4 bytes
    std::array<uint16_t, MaxMoveCount> policyPriors;    // 112 bytes
};
static_assert(sizeof(PredictionCacheEntry) == 128);

struct alignas(1024) PredictionCacheChunk
{
public:

    void Put(Key key, float value, int moveCount, const float* priors);

private:

    void Clear();
    bool TryGet(Key key, int moveCount, float* valueOut, float* priorsOut);

private:

    static const int EntryCount = 8;

    std::array<PredictionCacheEntry, EntryCount> _entries;  // 1024 bytes

    friend class PredictionCache;
};

static_assert(sizeof(PredictionCacheChunk) == 1024);

class PredictionCache
{
public:

    static PredictionCache Instance;

private:

    constexpr static const int MaxTableCount = (1 << 8);
    constexpr static const int MaxChunksPerTable = (1 << 20);

public:

    PredictionCache();
    ~PredictionCache();

    void Allocate(int sizeMebibytes);
    void Free();

    bool TryGetPrediction(Key key, int moveCount, PredictionCacheChunk** chunkOut, float* valueOut, float* priorsOut);
    void Clear();
    void ResetProbeMetrics();

    void PrintDebugInfo();
    int PermilleFull();
    int PermilleHits();
    int PermilleEvictions();

private:

    int _allocatedSizeMebibytes;
    std::vector<PredictionCacheChunk*> _tables;
    std::vector<void*> _allocations;
    std::vector<void*> _fallbackAllocations;
    int _chunksPerTable;

    uint64_t _hitCount;
    uint64_t _evictionCount;
    uint64_t _probeCount;

    uint64_t _entryCount;
    uint64_t _entryCapacity;

    friend struct PredictionCacheChunk;
};

#endif // _PREDICTIONCACHE_H_