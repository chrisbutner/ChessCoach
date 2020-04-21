#ifndef _PREDICTIONCACHE_H_
#define _PREDICTIONCACHE_H_

#include <vector>
#include <shared_mutex>

#include <Stockfish/types.h>

#include "Network.h"

struct PredictionCacheEntry
{
    Key key;                                        // 8 bytes
    float value;                                    // 4 bytes
    int moveCount;                                  // 4 bytes
    uint16_t policyMoves[Config::MaxBranchMoves];   // 160 bytes
    float policyPriors[Config::MaxBranchMoves];     // 320 bytes
};
static_assert(sizeof(std::shared_mutex) == 8);
static_assert(Config::MaxBranchMoves == 80);
static_assert(sizeof(PredictionCacheEntry) == 496);

struct PredictionCacheChunk
{
    void Clear();
    bool TryGet(Key key, float* valueOut, int* moveCountOut, uint16_t* movesOut, float* priorsOut);
    void Put(Key key, float value, int moveCount, uint16_t* moves, float* priors);

private:

    static const int EntryCount = 8;

    std::array<PredictionCacheEntry, EntryCount> _entries;  // 3968 bytes
    std::array<int, EntryCount> _ages;                      // 32 bytes
    std::shared_mutex _mutex;                               // 8 bytes
    char padding[88];                                       // 88 bytes

    friend class PredictionCache;
};
static_assert(sizeof(PredictionCacheChunk) == 4096);

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

    bool TryGetPrediction(Key key, PredictionCacheChunk** chunkOut, float* valueOut, int* moveCountOut, uint16_t* movesOut, float* priorsOut);
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