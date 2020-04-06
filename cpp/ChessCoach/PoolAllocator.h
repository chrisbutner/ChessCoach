#ifndef _POOLALLOCATOR_H_
#define _POOLALLOCATOR_H_

#include <algorithm>
#include <cassert>

class LargePageAllocator
{
private:

    static size_t LargePageMinimum;

public:

    static void Initialize();

    static void* Allocate(size_t byteCount);
    static void Free(void* memory);
};

struct Chunk
{
    Chunk* next;
};

template <class T>
class PoolAllocator
{
public:

    PoolAllocator()
        : _memory(nullptr)
        , _next(nullptr)
        , _currentlyAllocatedCount(0)
    {
    }

    ~PoolAllocator()
    {
        if (_memory)
        {
            LargePageAllocator::Free(_memory);
            _memory = nullptr;
        }
        _next = nullptr;
    }

    void Initialize(size_t itemCount)
    {
        if (_memory)
        {
            LargePageAllocator::Free(_memory);
            _memory = nullptr;
        }

        // VirtualAlloc aligns to 64KiB, definitely covers base alignment.
        const size_t alignment = std::max(__STDCPP_DEFAULT_NEW_ALIGNMENT__, alignof(T));
        const size_t alignedSizeBytes = ((sizeof(T) + alignment - 1) / alignment) * alignment;
        const size_t poolSizeBytes = alignedSizeBytes * itemCount;

        _memory = LargePageAllocator::Allocate(poolSizeBytes);
        assert(_memory);
        if (!_memory)
        {
            throw std::bad_alloc();
        }

        Chunk* chunk = reinterpret_cast<Chunk*>(_memory);
        _next = chunk;
        
        for (int i = 0; i < itemCount - 1; i++)
        {
            chunk->next = reinterpret_cast<Chunk*>(reinterpret_cast<char*>(chunk) + alignedSizeBytes);
            chunk = chunk->next;
        }

        chunk->next = nullptr;
    }
    

    void* Allocate()
    {
        if (!_next)
        {
            throw std::bad_alloc();
        }

#ifdef _DEBUG
        _currentlyAllocatedCount++;
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

private:

    void* _memory;
    Chunk* _next;
    int _currentlyAllocatedCount;
};

#endif // _POOLALLOCATOR_H_