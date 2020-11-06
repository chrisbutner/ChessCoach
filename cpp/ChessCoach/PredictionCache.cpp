#include "PredictionCache.h"

#include <iostream>
#include <cassert>

#include "PoolAllocator.h"

PredictionCache PredictionCache::Instance;

void PredictionCacheChunk::Clear()
{
    for (int i = 0; i < EntryCount; i++)
    {
        _entries[i].key = 0;
        _ages[i] = 0;
    }
}

bool PredictionCacheChunk::TryGet(Key key, int moveCount, float* valueOut, float* priorsOut)
{
    for (int& age : _ages)
    {
        age++;
    }

    for (int i = 0; i < EntryCount; i++)
    {
        if (_entries[i].key == key)
        {
            // Various types of collisions and race conditions across threads are possible:
            //
            // - Key collisions (type-1 errors):
            //      - Can mitigate by increasing key size, table size, or e.g. for Stockfish,
            //        validating the stored information, a Move, against the probing position.
            //        Note that validating using key/input information rather than stored/output
            //        is equivalent to increasing key size. We can mitigate by summing policy for
            //        the probing legal move count and ensuring that it's nearly 1.0.
            // - Index collisions (type-2 errors):
            //      - Results from the bit shrinkage from the full key size down to the addressable
            //        space of the table. Can be mitigated by storing the the full key, or more of
            //        the key than the addressable space, and validating when probing. We store the
            //        full key.
            // - Spliced values from parallel thread writes:
            //      - Multiple threads may race to write to a single chunk and/or entry, splicing
            //        values - either intra-field (e.g. writing to different parts of the policy)
            //        or inter-field (e.g. one thread writing key/value and the other writing policy).
            //        We have to accept seeing an incorrect value, but intra and inter-policy splicing
            //        can potentially be detected by again summing policy for the probing legal move count
            //        and ensuring that it's nearly 1.0.
            //
            // Use the provided "priorsOut" as writable scratch space even if we return false.
            //
            // Allow for 3 quanta error, ~1%.

            int priorSum = 0;
            for (int m = 0; m < moveCount; m++)
            {
                const uint8_t quantizedPrior = _entries[i].policyPriors[m];
                priorSum += quantizedPrior;

                const float prior = INetwork::DequantizeProbability(quantizedPrior);
                priorsOut[m] = prior;
            }

            // Check for type-1 errors and splices and return false. It's important not to
            // freshen the age in these cases so that splices can be overwritten with good data.
            static_assert(INetwork::DequantizeProbability(255) == 1.f);
            if ((priorSum < 252) || (priorSum > 258))
            {
                return false;
            }

            // The entry is valid, as far as we can tell, so freshen its age and return it.
            _ages[i] = std::numeric_limits<int>::min();
            *valueOut = _entries[i].value;
            return true;
        }
    }

    return false;
}

void PredictionCacheChunk::Put(Key key, float value, int moveCount, float* priors)
{
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
    for (int m = 0; m < moveCount; m++)
    {
        _entries[oldestIndex].policyPriors[m] = INetwork::QuantizeProbability(priors[m]);
    }

    // Place a "guard" probability of 1.0 immediately after the N legal moves' probabilities
    // so that "TryGet" can more often detect incorrect probability sums (rather than potentially
    // seeing only trailing zeros and still summing to 1.0).
    if (moveCount < _entries[oldestIndex].policyPriors.size())
    {
        _entries[oldestIndex].policyPriors[moveCount] = INetwork::QuantizeProbability(1.f);
    }
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
        // Allocate table with large page support.
        void* memory = LargePageAllocator::Allocate(TableBytes);
        assert(memory);
        if (!memory)
        {
            throw std::bad_alloc();
        }

        _tables.push_back(reinterpret_cast<PredictionCacheChunk*>(memory));
    }

#ifdef CHESSCOACH_WINDOWS
    // Memory is already zero-filled by VirtualAlloc on Windows, so no need to clear chunks.
#else
    // Memory comes from std::aligned_alloc in Linux with madvise hint, so no zero-filling guarantees
    // and may not even be large pages. We can either Clear() or memset, haven't timed them.
    Clear();
#endif

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

    ResetProbeMetrics();

    _entryCount = 0;
    _entryCapacity = 0;
}

// If returning true, valueOut and priorsOut are populated; chunkOut is not populated.
// If returning false, valueOut and priorsOut are not populated; chunkOut is populated only if the value/policy should be stored when available.
bool PredictionCache::TryGetPrediction(Key key, int moveCount, PredictionCacheChunk** chunkOut, float* valueOut, float* priorsOut)
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

    if (chunk.TryGet(key, moveCount, valueOut, priorsOut))
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

    ResetProbeMetrics();

    _entryCount = 0;
}

void PredictionCache::ResetProbeMetrics()
{
    _hitCount = 0;
    _evictionCount = 0;
    _probeCount = 0;
}

void PredictionCache::PrintDebugInfo()
{
    std::cout << "Prediction cache full: " << (static_cast<float>(_entryCount) / _entryCapacity)
        << ", hit rate: " << (static_cast<float>(_hitCount) / _probeCount) 
        << ", eviction rate: " << (static_cast<float>(_evictionCount) / _probeCount) << std::endl;
}

int PredictionCache::PermilleFull()
{
    return ((_entryCapacity == 0) ? 0 : static_cast<int>(_entryCount * 1000 / _entryCapacity));
}

int PredictionCache::PermilleHits()
{
    return ((_probeCount == 0) ? 0 : static_cast<int>(_hitCount * 1000 / _probeCount));
}

int PredictionCache::PermilleEvictions()
{
    return ((_probeCount == 0) ? 0 : static_cast<int>(_evictionCount * 1000 / _probeCount));
}