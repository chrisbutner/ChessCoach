#include <gtest/gtest.h>

#include <functional>

#include <ChessCoach/SelfPlay.h>
#include <ChessCoach/ChessCoach.h>

SelfPlayGame& PlayGame(SelfPlayWorker& selfPlayWorker, std::function<void (SelfPlayGame&)> tickCallback)
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

        tickCallback(*game);
    }
}

std::vector<Node*> GeneratePrincipleVariation(const SelfPlayGame& game)
{
    Node* node = game.Root();
    std::vector<Node*> principleVariation;

    while (!node->children.empty())
    {
        int maxVisitCount = std::numeric_limits<int>::min();
        std::pair<Move, Node*> maxVisited;
        for (const auto& pair : node->children)
        {
            // node->mostVisitedChild needs to win ties.
            const int visitCount = pair.second->visitCount;
            if ((visitCount > maxVisitCount) ||
                ((visitCount == maxVisitCount) && node->mostVisitedChild.second && (pair.second == node->mostVisitedChild.second)))
            {
                maxVisitCount = visitCount;
                maxVisited = pair;
            }
        }
        EXPECT_NE(maxVisited.second, nullptr);
        if (maxVisited.second->visitCount > 0)
        {
            EXPECT_EQ(maxVisited.second, node->mostVisitedChild.second);
            principleVariation.push_back(maxVisited.second);
        }
        node = maxVisited.second;
    }

    EXPECT_EQ(node->mostVisitedChild.second, nullptr);

    return principleVariation;
}

TEST(Mcts, NodeLeaks)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    SelfPlayWorker selfPlayWorker;

    auto [currentBefore, peakBefore] = Node::Allocator.DebugAllocations();
    EXPECT_EQ(currentBefore, 0);
    EXPECT_EQ(peakBefore, 0);

    PlayGame(selfPlayWorker, [](auto&) {});

    auto [currentAfter, peakAfter] = Node::Allocator.DebugAllocations();
    EXPECT_EQ(currentAfter, 0);
    EXPECT_GT(peakAfter, 0);
}

TEST(Mcts, PrincipleVariation)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    SelfPlayWorker selfPlayWorker;
    SearchState& searchState = selfPlayWorker.DebugSearchState();

    std::vector<Node*> latestPrincipleVariation;
    PlayGame(selfPlayWorker, [&](SelfPlayGame& game)
        {
            std::vector<Node*> principleVariation = GeneratePrincipleVariation(game);
            if (searchState.principleVariationChanged)
            {
                EXPECT_NE(principleVariation, latestPrincipleVariation);
                searchState.principleVariationChanged = false;
            }
            else
            {
                EXPECT_EQ(principleVariation, latestPrincipleVariation);
            }
            latestPrincipleVariation = principleVariation;
        });
}