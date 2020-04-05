#include "PredictionCache.h"

#include <iostream>
#include <cassert>

#include "PoolAllocator.h"

PredictionCache PredictionCache::Instance;

PredictionCache::PredictionCache()
    : _bucketEntryCount(0)
    , _hitCount(0)
    , _collisionCount(0)
    , _probeCount(0)
{
}

PredictionCache::~PredictionCache()
{
    for (void* memory : _bucketMemory)
    {
        if (memory)
        {
            LargePageAllocator::Free(memory);
        }
    }
    _bucketMemory.clear();
    _bucketEntries.clear();
}

void PredictionCache::Allocate(int sizeGb)
{
    const int bucketCount = sizeGb;
    const int bucketBytes = (1024 * 1024 * 1024);
    _bucketEntryCount = (bucketBytes / sizeof(PredictionCacheEntry));

    _bucketMemory.reserve(bucketCount);
    _bucketEntries.reserve(bucketCount);

    for (int i = 0; i < bucketCount; i++)
    {
        void* memory = LargePageAllocator::Allocate(bucketBytes);
        assert(memory);
        _bucketMemory.push_back(memory);
        _bucketEntries.push_back(static_cast<PredictionCacheEntry*>(memory));
    }
}

// If returning true, valueOut, moveCountOut, movesOut and priorsOut are populated.
// If returning false, valueOut, moveCountOut, movesOut and priorsOut are not populated; entryOut is populated only if the value/policy should be stored when available.
bool PredictionCache::TryGetPrediction(Key key, PredictionCacheEntry** entryOut, float* valueOut, int* moveCountOut, Move* movesOut, float* priorsOut)
{
    _probeCount++;

    // Use the high 16 bits to choose the bucket.
    const uint16_t bucketKey = (key >> 48);
    PredictionCacheEntry* bucket = _bucketEntries[bucketKey % _bucketEntries.size()];

    // Use the low 48 bits to choose the entry.
    const uint64_t entryKey = (key & 0xFFFFFFFFFFFF);
    PredictionCacheEntry& entry = bucket[entryKey % _bucketEntryCount];

    bool outerHit = (entry._key == key);
    if (!outerHit)
    {
        if (entry._key)
        {
            _collisionCount++;
        }
        *entryOut = &entry;
        return false;
    }

    {
        std::shared_lock lock(entry._mutex);

        bool hit = (entry._key == key);
        if (hit)
        {
            _hitCount++;
            *valueOut = entry._value;
            *moveCountOut = entry._moveCount;
            for (int i = 0; i < entry._moveCount; i++)
            {
                movesOut[i] = Move(entry._policyMoves[i]);
                priorsOut[i] = entry._policyPriors[i];
            }
        }
        else
        {
            if (entry._key)
            {
                _collisionCount++;
            }
            *entryOut = &entry;
        }
        return hit;
    }
}

void PredictionCache::PrintDebugInfo()
{
    std::cout << "Cache hit rate: " << (static_cast<float>(_hitCount) / _probeCount) << std::endl;
    std::cout << "Cache collision/evict rate: " << (static_cast<float>(_collisionCount) / _probeCount) << std::endl;
    for (PredictionCacheEntry* bucket : _bucketEntries)
    {
        int fullCount = 0;
        for (int i = 0; i < _bucketEntryCount; i++)
        {
            if (bucket[i]._key)
            {
                fullCount++;
            }
        }
        std::cout << "Bucket filled: " << (static_cast<float>(fullCount) / _bucketEntryCount) << std::endl;
    }
}