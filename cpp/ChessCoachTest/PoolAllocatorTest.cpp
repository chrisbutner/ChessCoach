#include <gtest/gtest.h>

#include <array>

#include <ChessCoach/SelfPlay.h>
#include <ChessCoach/PoolAllocator.h>

TEST(PoolAllocator, Basic)
{
    const size_t blockSizeBytes = (sizeof(std::tuple<int, int, int, int>) * 3);
    const size_t maxBlocks = 1;
    PoolAllocator<std::tuple<int, int, int, int>, blockSizeBytes, maxBlocks> poolAllocator;

    // Allocate all 3 items from the pool then expect it to be empty.

    void* item1 = poolAllocator.Allocate();
    EXPECT_NE(item1, nullptr);

    void* item2 = poolAllocator.Allocate();
    EXPECT_NE(item2, nullptr);

    void* item3 = poolAllocator.Allocate();
    EXPECT_NE(item3, nullptr);

    EXPECT_THROW(poolAllocator.Allocate(), std::bad_alloc);

    // Free 1 item, allocate 1 item, then expect it to be empty.

    poolAllocator.Free(item3);

    void* item4 = poolAllocator.Allocate();
    EXPECT_NE(item4, nullptr);

    EXPECT_EQ(item3, item4);

    EXPECT_THROW(poolAllocator.Allocate(), std::bad_alloc);
}

TEST(PoolAllocator, Alignment)
{
    const size_t alignment = std::max(static_cast<size_t>(__STDCPP_DEFAULT_NEW_ALIGNMENT__), static_cast<size_t>(alignof(int)));
    EXPECT_GT(alignment, sizeof(int));

    const int itemCount = 5;
    const size_t blockSizeBytes = 1024;
    PoolAllocator<int, blockSizeBytes> poolAllocator;

    for (int i = 0; i < itemCount; i++)
    {
        uintptr_t item = reinterpret_cast<uintptr_t>(poolAllocator.Allocate());
        EXPECT_EQ(item % alignment, 0);
    }
}