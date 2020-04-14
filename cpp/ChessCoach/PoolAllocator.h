#ifndef _POOLALLOCATOR_H_
#define _POOLALLOCATOR_H_

#include <vector>
#include <algorithm>
#include <cassert>

class LargePageAllocator
{
private:

    static size_t LargePageMinimum;

public:

    static void* Allocate(size_t byteCount);
    static void Free(void* memory);

private:

    static void Initialize();
};

struct Chunk
{
    Chunk* next;
};

template <class T, size_t BlockSizeBytes, size_t MaxBlocks = std::numeric_limits<size_t>::max()>
class PoolAllocator
{
public:

    PoolAllocator()
        : _next(nullptr)
        , _currentlyAllocatedCount(0)
        , _peakAllocatedCount(0)
    {
    }

    ~PoolAllocator()
    {
        for (void* block : _blocks)
        {
            LargePageAllocator::Free(block);
        }
        _blocks.clear();
        _next = nullptr;
    }
    

    void* Allocate()
    {
        if (!_next)
        {
            AllocateBlock();
        }

#ifdef _DEBUG
        _currentlyAllocatedCount++;
        _peakAllocatedCount = std::max(_peakAllocatedCount, _currentlyAllocatedCount);
#endif
        
        void* allocation = _next;
        _next = _next->next;

        return allocation;
    }

    void Free(void* memory)
    {
        if (!memory)
        {
            return;
        }

#ifdef _DEBUG
        _currentlyAllocatedCount--;
#endif

        Chunk* asChunk = reinterpret_cast<Chunk*>(memory);
        
        asChunk->next = _next;
        _next = asChunk;
    }

    std::pair<int, int> DebugAllocations()
    {
        return std::pair(_currentlyAllocatedCount, _peakAllocatedCount);
    }

    int DebugBlockCount()
    {
        return static_cast<int>(_blocks.size());
    }

private:

    void AllocateBlock()
    {
        if (_blocks.size() >= MaxBlocks)
        {
            throw std::bad_alloc();
        }

        // VirtualAlloc aligns to 64KiB, definitely covers base alignment.
        const size_t alignment = std::max(__STDCPP_DEFAULT_NEW_ALIGNMENT__, alignof(T));
        const size_t alignedItemSizeBytes = ((sizeof(T) + alignment - 1) / alignment) * alignment;
        const size_t itemCount = (BlockSizeBytes / alignedItemSizeBytes);
        if (itemCount == 0)
        {
            throw std::bad_alloc();
        }

        void* block = LargePageAllocator::Allocate(BlockSizeBytes);
        assert(block);
        if (!block)
        {
            throw std::bad_alloc();
        }
        _blocks.push_back(block);

        Chunk* first = reinterpret_cast<Chunk*>(block);
        Chunk* chunk = first;

        for (int i = 0; i < itemCount - 1; i++)
        {
            chunk->next = reinterpret_cast<Chunk*>(reinterpret_cast<char*>(chunk) + alignedItemSizeBytes);
            chunk = chunk->next;
        }

        // Join the tail to any existing free chunks, then put this new block's chunks at the head.
        chunk->next = _next;
        _next = first;
    }

private:

    std::vector<void*> _blocks;
    Chunk* _next;
    int _currentlyAllocatedCount;
    int _peakAllocatedCount;
};

#endif // _POOLALLOCATOR_H_