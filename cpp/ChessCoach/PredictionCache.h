#ifndef _PREDICTIONCACHE_H_
#define _PREDICTIONCACHE_H_

#include <vector>
#include <shared_mutex>

#include <Stockfish/types.h>

#include "Network.h"

struct PredictionCacheEntry
{
public:

    template <typename MIter, typename PIter>
    void Set(Key key, float value, int moveCount, MIter policyMovesBegin, PIter policyPriorsBegin)
    {
        assert(moveCount <= INetwork::MaxBranchMoves);

        std::unique_lock lock(_mutex);

        _key = key;
        _value = value;
        _moveCount = moveCount;
        std::copy(policyMovesBegin, policyMovesBegin + moveCount, _policyMoves);
        std::copy(policyPriorsBegin, policyPriorsBegin + moveCount, _policyPriors);
    }

    void Clear()
    {
        _key = 0;
    }

private:

    // pad to 512 bytes, 8 * 64

    std::shared_mutex _mutex;                               // 8 bytes
    Key _key;                                               // 8 bytes
    float _value;                                           // 4 bytes
    int _moveCount;                                         // 4 bytes
    char _padding[8];                                       // 8 bytes
    uint16_t _policyMoves[INetwork::MaxBranchMoves];        // 160 bytes
    float _policyPriors[INetwork::MaxBranchMoves];          // 320 bytes

    friend class PredictionCache;
};
static_assert(sizeof(std::shared_mutex) == 8);
static_assert(INetwork::MaxBranchMoves == 80);
static_assert(sizeof(PredictionCacheEntry) == 8 * 64);

class PredictionCache
{
public:

    static PredictionCache Instance;

public:

    PredictionCache();
    ~PredictionCache();

    void Allocate(int sizeGb);

    bool TryGetPrediction(Key key, PredictionCacheEntry** entryOut, float* valueOut, int* moveCountOut, Move* movesOut, float* priorsOut);
    void Clear();

    void PrintDebugInfo();

private:

    int _bucketEntryCount;
    std::vector<void*> _bucketMemory;
    std::vector<PredictionCacheEntry*> _bucketEntries;

    uint64_t _hitCount;
    uint64_t _collisionCount;
    uint64_t _probeCount;
};

#endif // _PREDICTIONCACHE_H_