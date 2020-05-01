#ifndef _PREDICTIONCACHE_H_
#define _PREDICTIONCACHE_H_

#include <vector>
#include <shared_mutex>

#include <Stockfish/types.h>

#include "Network.h"

struct PredictionCacheEntry
{
    // Positions with more moves don't fit in the cache and so shouldn't be probed/stored.
    static const int MaxMoveCount = 52;

    Key key;                                                            // 8 bytes
    float value;                                                        // 4 bytes
    std::array<uint8_t, MaxMoveCount> policyPriors;    // 52 bytes
};
static_assert(sizeof(PredictionCacheEntry) == 64);

struct PredictionCacheChunk
{
    void Clear();
    bool TryGet(Key key, int moveCount, float* valueOut, float* priorsOut);
    void Put(Key key, float value, int moveCount, float* priors);

private:

    static const int EntryCount = 7;

    std::array<PredictionCacheEntry, EntryCount> _entries;  // 384 bytes
    std::array<int, EntryCount> _ages;                      // 32 bytes
    std::shared_mutex _mutex;                               // 8 bytes
    char padding[24];                                       // 24 bytes

    friend class PredictionCache;
};
static_assert(sizeof(std::shared_mutex) == 8);
static_assert(sizeof(PredictionCacheChunk) == 512);

class PredictionCache
{
public:

    static PredictionCache Instance;

private:

    static const int TableBytes = 1024 * 1024 * 1024;
    static const int ChunksPerTable = (TableBytes / sizeof(PredictionCacheChunk));

public:

    PredictionCache();
    ~PredictionCache();

    void Allocate(int sizeGb);
    void Free();

    bool TryGetPrediction(Key key, int moveCount, PredictionCacheChunk** chunkOut, float* valueOut, float* priorsOut);
    void Clear();
    void ResetProbeMetrics();

    void PrintDebugInfo();
    int PermilleFull();
    int PermilleHits();
    int PermilleEvictions();

private:

    std::vector<PredictionCacheChunk*> _tables;

    uint64_t _hitCount;
    uint64_t _evictionCount;
    uint64_t _probeCount;

    uint64_t _entryCount;
    uint64_t _entryCapacity;

    friend struct PredictionCacheChunk;
};

#endif // _PREDICTIONCACHE_H_