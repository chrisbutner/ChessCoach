#include <gtest/gtest.h>

#include <array>

#include <ChessCoach/SelfPlay.h>
#include <ChessCoach/PoolAllocator.h>

TEST(PoolAllocator, Basic)
{
    LargePageAllocator::Initialize();

    PoolAllocator<int> poolAllocator;

    const int itemCount = 3;
    poolAllocator.Initialize(itemCount);

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

TEST(PoolAllocator, Node)
{
    LargePageAllocator::Initialize();

    const int itemCount = 3;
    Node::Allocator.Initialize(itemCount);

    // Create all 3 nodes then expect it to be empty.

    Node* node1 = new Node(0.f);
    EXPECT_NE(node1, nullptr);

    Node* node2 = new Node(0.f);
    EXPECT_NE(node2, nullptr);

    Node* node3 = new Node(0.f);
    EXPECT_NE(node3, nullptr);

    EXPECT_THROW(new Node(0.f), std::bad_alloc);

    // Delete 1, create 1 then expect it to be empty.

    delete node3;

    Node* node4 = new Node(0.f);
    EXPECT_NE(node4, nullptr);

    EXPECT_EQ(node3, node4);

    EXPECT_THROW(new Node(0.f), std::bad_alloc);
}

TEST(PoolAllocator, Alignment)
{
    const size_t alignment = std::max(__STDCPP_DEFAULT_NEW_ALIGNMENT__, alignof(int));
    EXPECT_GT(alignment, sizeof(int));

    LargePageAllocator::Initialize();

    PoolAllocator<int> poolAllocator;

    const int itemCount = 5;
    poolAllocator.Initialize(itemCount);
        
    for (int i = 0; i < itemCount; i++)
    {
        uintptr_t item = reinterpret_cast<uintptr_t>(poolAllocator.Allocate());
        EXPECT_EQ(item % alignment, 0);
    }
}