#include <gtest/gtest.h>

#include <ChessCoach/SelfPlay.h>
#include <ChessCoach/ChessCoach.h>

TEST(Network, Policy)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    // Just use training network config.
    const NetworkConfig& networkConfig = Config::TrainingNetwork;

    std::vector<INetwork::InputPlanes> images(networkConfig.SelfPlay.PredictionBatchSize);
    std::vector<float> values(networkConfig.SelfPlay.PredictionBatchSize);
    std::vector<INetwork::OutputPlanes> policies(networkConfig.SelfPlay.PredictionBatchSize);

    SelfPlayGame game("3rkb1r/p2nqppp/5n2/1B2p1B1/4P3/1Q6/PPP2PPP/2KR3R w k - 3 13", {}, false /* tryHard */, &images[0], &values[0], &policies[0]);
    
    MoveList legalMoves = MoveList<LEGAL>(game.DebugPosition());
    const int legalMoveCount = static_cast<int>(legalMoves.size());

    // Give 5 visits evenly across legal moves, then the rest to the first move.
    const int evenCount = 5;
    Node* lastSibling = nullptr;
    for (Move move : legalMoves)
    {
        Node* child = new Node(move, 0.f);
        child->visitCount += evenCount;
        if (lastSibling)
        {
            lastSibling->nextSibling = child;
        }
        else
        {
            game.Root()->firstChild = child;
        }
        lastSibling = child;
    }
    Move firstMove = *legalMoves.begin();
    game.Root()->firstChild->visitCount += (networkConfig.SelfPlay.NumSimulations - (legalMoveCount * evenCount));

    // Generate policy labels. Make sure that legal moves are non-zero and the rest are zero.
    game.StoreSearchStatistics();
    game.ApplyMoveWithRootAndHistory(firstMove, game.Root()->firstChild);
    game.Complete();
    std::unique_ptr<INetwork::OutputPlanes> labels(std::make_unique<INetwork::OutputPlanes>());
    game.GeneratePolicy(game.Save().childVisits[0], *labels);
    
    for (Move move : legalMoves)
    {
        EXPECT_GT(game.PolicyValue(*labels, move), 0.f);
    }

    int zeroCount = 0;
    INetwork::PlanesPointerFlat value = reinterpret_cast<INetwork::PlanesPointerFlat>(labels.get());
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
        sum += (game.PolicyValue(*labels, *(legalMoves.begin() + i)) * -::logf(prediction));
    }
}

TEST(Network, ImagePieceHistoryPlanes)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    Game game;

    const int finalHistoryPlanes = (INetwork::InputPreviousPositionCount - 1) * INetwork::InputPiecePlanesPerPosition;
    const int currentPositionPlanes = INetwork::InputPreviousPositionCount * INetwork::InputPiecePlanesPerPosition;

    // Ensure that the final history plane is all zeros.
    std::unique_ptr<INetwork::InputPlanes> image1(std::make_unique<INetwork::InputPlanes>());
    game.GenerateImage(*image1);
    const INetwork::PackedPlane startingPositionOurPawns = (*image1)[currentPositionPlanes + 0];
    for (int i = 0; i < INetwork::InputPiecePlanesPerPosition; i++)
    {
        EXPECT_EQ((*image1)[finalHistoryPlanes + i], 0);
    }

    // Make a move. Ensure that the final history our-pawns plane equals the starting position's.
    game.ApplyMove(make_move(SQ_E2, SQ_E4));
    std::unique_ptr<INetwork::InputPlanes> image2(std::make_unique<INetwork::InputPlanes>());
    game.GenerateImage(*image2);
    EXPECT_EQ((*image2)[finalHistoryPlanes + 0], startingPositionOurPawns);
}