// ChessCoach, a neural network-based chess engine capable of natural-language commentary
// Copyright 2021 Chris Butner
//
// ChessCoach is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ChessCoach is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ChessCoach. If not, see <https://www.gnu.org/licenses/>.

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