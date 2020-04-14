#include <gtest/gtest.h>

#include <ChessCoach/SelfPlay.h>
#include <ChessCoach/ChessCoach.h>

TEST(Network, Policy)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    std::vector<INetwork::InputPlanes> images(Config::PredictionBatchSize);
    std::vector<float> values(Config::PredictionBatchSize);
    std::vector<INetwork::OutputPlanes> policies(Config::PredictionBatchSize);

    SelfPlayGame game("3rkb1r/p2nqppp/5n2/1B2p1B1/4P3/1Q6/PPP2PPP/2KR3R w k - 3 13", {}, false /* tryHard */, &images[0], &values[0], &policies[0]);
    
    MoveList legalMoves = MoveList<LEGAL>(game.DebugPosition());
    const int legalMoveCount = static_cast<int>(legalMoves.size());

    // Give 5 visits evenly across legal moves, then the rest to the first move.
    const int evenCount = 5;
    for (Move move : legalMoves)
    {
        Node* child = new Node(0.f);
        child->visitCount += evenCount;
        game.Root()->children[move] = child;
    }
    Move firstMove = *legalMoves.begin();
    game.Root()->children[firstMove]->visitCount += (Config::NumSimulations - (legalMoveCount * evenCount));

    // Generate policy labels. Make sure that legal moves are non-zero and the rest are zero.
    game.StoreSearchStatistics();
    game.ApplyMoveWithRootAndHistory(firstMove, game.Root()->children[firstMove]);
    game.Complete();
    INetwork::OutputPlanes labels = game.GeneratePolicy(game.Save().childVisits[0]);
    
    for (Move move : legalMoves)
    {
        EXPECT_GT(game.PolicyValue(labels, move), 0.f);
    }

    int zeroCount = 0;
    float* value = reinterpret_cast<float*>(labels.data());
    for (int i = 0; i < INetwork::OutputPlanesFloatCount; i++)
    {
        if (*(value++) == 0.f) zeroCount++;
    }
    EXPECT_EQ(zeroCount, INetwork::OutputPlanesFloatCount - legalMoveCount);

    // This isn't a test, just checking some ballpark loss (~3.93).
    // Check categorical cross-entropy loss. Fake a uniform policy, post-softmax.
    float prediction = (1.f / legalMoveCount);
    float sum = 0;
    for (int i = 0; i < legalMoveCount; i++)
    {
        sum += (game.PolicyValue(labels, *(legalMoves.begin() + i)) * -std::logf(prediction));
    }
}
