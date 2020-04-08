#include <array>

#include <gtest/gtest.h>

#include <ChessCoach/SelfPlay.h>
#include <ChessCoach/PredictionCache.h>
#include <ChessCoach/ChessCoach.h>

bool TryGetPrediction(Key key)
{
    PredictionCacheEntry* entry;
    float value;
    int moveCount;
    std::array<Move, INetwork::MaxBranchMoves> moves;
    std::array<float, INetwork::MaxBranchMoves> priors;

    bool hit = PredictionCache::Instance.TryGetPrediction(key, &entry, &value, &moveCount, moves.data(), priors.data());
    if (!hit)
    {
        entry->Set(key, 0.f, 0, moves.data(), priors.data());
    }
    return hit;
}

TEST(PredictionCache, Basic)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    Game startingPosition;
    Game game1 = startingPosition;
    Game game2 = startingPosition;

    for (Move move : { make_move(SQ_A2, SQ_A3), make_move(SQ_A7, SQ_A6), make_move(SQ_A3, SQ_A4)})
    {
        game1.ApplyMove(move);
        game2.ApplyMove(move);
    }

    Key game1_key = game1.GenerateImageKey();
    Key game2_key = game2.GenerateImageKey();

    EXPECT_EQ(game1.DebugPosition().key(), game2.DebugPosition().key());
    EXPECT_EQ(game1_key, game2_key);
    EXPECT_FALSE(TryGetPrediction(game1_key));
    EXPECT_TRUE(TryGetPrediction(game2_key));

    EXPECT_TRUE(TryGetPrediction(game1_key));
    EXPECT_TRUE(TryGetPrediction(game2_key));
}

TEST(PredictionCache, PathDependence)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    Game startingPosition;
    Game game1 = startingPosition;
    Game game2 = startingPosition;

    // Make sure that two starting positions match in the prediction cache.

    Key game1_key1 = game1.GenerateImageKey();
    Key game2_key1 = game2.GenerateImageKey();
    
    EXPECT_EQ(game1.DebugPosition().key(), game2.DebugPosition().key());
    EXPECT_EQ(game1_key1, game2_key1);
    EXPECT_FALSE(TryGetPrediction(game1_key1));
    EXPECT_TRUE(TryGetPrediction(game2_key1));

    // Make different first moves. Make sure that the keys and games don't match in the prediction cache.

    game1.ApplyMove(make_move(SQ_A2, SQ_A3));
    game2.ApplyMove(make_move(SQ_B2, SQ_B3));

    Key game1_key2 = game1.GenerateImageKey();
    Key game2_key2 = game2.GenerateImageKey();

    EXPECT_NE(game1.DebugPosition().key(), game2.DebugPosition().key());
    EXPECT_NE(game1_key2, game2_key2);
    EXPECT_FALSE(TryGetPrediction(game1_key2));
    EXPECT_FALSE(TryGetPrediction(game2_key2));

    // Transpose to the same position in both games. Make sure that the position keys match but the games
    // don't match in the prediction cache because of different move paths.

    game1.ApplyMove(make_move(SQ_A7, SQ_A6));
    game2.ApplyMove(make_move(SQ_A7, SQ_A6));

    game1.ApplyMove(make_move(SQ_B2, SQ_B3));
    game2.ApplyMove(make_move(SQ_A2, SQ_A3));
    
    Key game1_key3 = game1.GenerateImageKey();
    Key game2_key3 = game2.GenerateImageKey();

    EXPECT_EQ(game1.DebugPosition().key(), game2.DebugPosition().key());
    EXPECT_NE(game1_key3, game2_key3);
    EXPECT_FALSE(TryGetPrediction(game1_key3));
    EXPECT_FALSE(TryGetPrediction(game2_key3));
}
