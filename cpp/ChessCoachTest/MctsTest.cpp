#include <gtest/gtest.h>

#include <ChessCoach/SelfPlay.h>
#include <ChessCoach/ChessCoach.h>

SelfPlayGame& PlayGame(SelfPlayWorker& selfPlayWorker)
{
    const int index = 0;
    SelfPlayGame* game;
    SelfPlayState* state;
    float* values;
    INetwork::OutputPlanes* policies;

    selfPlayWorker.DebugGame(index, &game, &state, &values, &policies);

    selfPlayWorker.SetUpGame(index);

    while (true)
    {
        // CPU work
        selfPlayWorker.Play(index);

        if (*state == SelfPlayState::Finished)
        {
            return *game;
        }

        // "GPU" work. Pretend to predict for a batch.
        std::fill(values, values + Config::PredictionBatchSize, CHESSCOACH_VALUE_DRAW);

        float* policiesPtr = reinterpret_cast<float*>(policies);
        const int policyCount = (Config::PredictionBatchSize * INetwork::OutputPlanesFloatCount);
        std::fill(policiesPtr, policiesPtr + policyCount, 0.f);
    }
}

TEST(Mcts, NodeLeaks)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    SelfPlayWorker selfPlayWorker;
    selfPlayWorker.Initialize(nullptr /* storage */);

    auto [currentBefore, peakBefore] = Node::Allocator.DebugAllocations();
    EXPECT_EQ(currentBefore, 0);
    EXPECT_EQ(peakBefore, 0);

    SelfPlayGame& game = PlayGame(selfPlayWorker);

    auto [currentAfter, peakAfter] = Node::Allocator.DebugAllocations();
    EXPECT_EQ(currentAfter, 0);
    EXPECT_GT(peakAfter, 0);
}