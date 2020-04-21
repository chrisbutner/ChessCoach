#include "PredictionCache.h"

#include <iostream>
#include <cassert>

#include "PoolAllocator.h"

PredictionCache PredictionCache::Instance;

void PredictionCacheChunk::Clear()
{
    // Don't lock here, assume it's a quiet single-threaded section.
    for (int i = 0; i < EntryCount; i++)
    {
        _entries[i].key = 0;
        _ages[i] = 0;
    }
}

bool PredictionCacheChunk::TryGet(Key key, float* valueOut, int* moveCountOut, uint16_t* movesOut, float* priorsOut)
{
    // Only ages are mutated and the operations are atomic on basically all hardware, so use a shared_lock.
    std::shared_lock lock(_mutex);

    for (int& age : _ages)
    {
        age++;
    }

    for (int i = 0; i < EntryCount; i++)
    {
        if (_entries[i].key == key)
        {
            _ages[i] = std::numeric_limits<int>::min();
            *valueOut = _entries[i].value;
            *moveCountOut = _entries[i].moveCount;
            std::copy(_entries[i].policyMoves, _entries[i].policyMoves + _entries[i].moveCount, movesOut);
            std::copy(_entries[i].policyPriors, _entries[i].policyPriors + _entries[i].moveCount, priorsOut);
            return true;
        }
    }

    return false;
}

void PredictionCacheChunk::Put(Key key, float value, int moveCount, uint16_t* moves, float* priors)
{
    std::unique_lock lock(_mutex);

    int oldestIndex = 0;
    for (int i = 1; i < EntryCount; i++)
    {
        if (_ages[i] > _ages[oldestIndex])
        {
            oldestIndex = i;
        }
    }

    // Hackily reach into the singleton PredictionCache to update metrics.
    if (_entries[oldestIndex].key)
    {
        PredictionCache::Instance._evictionCount++;
    }
    else
    {
        PredictionCache::Instance._entryCount++;
    }

    _ages[oldestIndex] = std::numeric_limits<int>::min();
    _entries[oldestIndex].key = key;
    _entries[oldestIndex].value = value;
    _entries[oldestIndex].moveCount = moveCount;
    std::copy(moves, moves + moveCount, _entries[oldestIndex].policyMoves);
    std::copy(priors, priors + moveCount, _entries[oldestIndex].policyPriors);
}

PredictionCache::PredictionCache()
    : _hitCount(0)
    , _evictionCount(0)
    , _probeCount(0)
    , _entryCount(0)
    , _entryCapacity(0)
{
}

PredictionCache::~PredictionCache()
{
    Free();
}

void PredictionCache::Allocate(int sizeGb)
{
    Free();

    const int tableCount = sizeGb;

    _tables.reserve(tableCount);

    for (int i = 0; i < tableCount; i++)
    {
        // Memory is already zero-filled, so no need to clear chunks.
        void* memory = LargePageAllocator::Allocate(TableBytes);
        assert(memory);
        if (!memory)
        {
            throw std::bad_alloc();
        }

        _tables.push_back(reinterpret_cast<PredictionCacheChunk*>(memory));
    }

    _entryCapacity = (static_cast<uint64_t>(tableCount) * ChunksPerTable * PredictionCacheChunk::EntryCount);
}

void PredictionCache::Free()
{
    for (void* memory : _tables)
    {
        if (memory)
        {
            LargePageAllocator::Free(memory);
        }
    }

    _tables.clear();

    _hitCount = 0;
    _evictionCount = 0;
    _probeCount = 0;
    _entryCount = 0;
    _entryCapacity = 0;
}

// If returning true, valueOut, moveCountOut, movesOut and priorsOut are populated; chunkOut is not populated.
// If returning false, valueOut, moveCountOut, movesOut and priorsOut are not populated; chunkOut is populated only if the value/policy should be stored when available.
bool PredictionCache::TryGetPrediction(Key key, PredictionCacheChunk** chunkOut, float* valueOut, int* moveCountOut, uint16_t* movesOut, float* priorsOut)
{
    if (_tables.empty())
    {
        return false;
    }

    _probeCount++;

    // Use the high 16 bits to choose the table.
    const uint16_t tableKey = (key >> 48);
    PredictionCacheChunk* table = _tables[tableKey % _tables.size()];

    // Use the low 48 bits to choose the chunk.
    const uint64_t chunkKey = (key & 0xFFFFFFFFFFFF);
    PredictionCacheChunk& chunk = table[chunkKey % ChunksPerTable];

    if (chunk.TryGet(key, valueOut, moveCountOut, movesOut, priorsOut))
    {
        _hitCount++;
        return true;
    }

    *chunkOut = &chunk;
    return false;
}

void PredictionCache::Clear()
{
    for (PredictionCacheChunk* table : _tables)
    {
        for (int i = 0; i < ChunksPerTable; i++)
        {
            table[i].Clear();
        }
    }

    _hitCount = 0;
    _evictionCount = 0;
    _probeCount = 0;
    _entryCount = 0;
    _entryCapacity = 0;
}

void PredictionCache::PrintDebugInfo()
{
    std::cout << "Prediction cache full: " << (static_cast<float>(_entryCount) / _entryCapacity)
        << ", hit rate: " << (static_cast<float>(_hitCount) / _probeCount) 
        << ", evict rate: " << (static_cast<float>(_evictionCount) / _probeCount) << std::endl;
}

int PredictionCache::PermilleFull()
{
    return static_cast<int>(_entryCount * 1000 / _entryCapacity);
}