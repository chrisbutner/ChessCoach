#include <gtest/gtest.h>

#include <array>

#include <ChessCoach/SelfPlay.h>
#include <ChessCoach/PredictionCache.h>
#include <ChessCoach/ChessCoach.h>

bool TryGetPrediction(Key key)
{
    PredictionCacheChunk* chunk;
    float value;
    std::array<float, 4> priors = { 0.1f, 0.2f, 0.3f, 0.4f };

    bool hit = PredictionCache::Instance.TryGetPrediction(key, static_cast<int>(priors.size()), &chunk, &value, priors.data());
    if (!hit)
    {
        chunk->Put(key, 0.33f, static_cast<int>(priors.size()), priors.data());
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

TEST(PredictionCache, Quantization)
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

    // Make sure that the two games' moves are the same.
    MoveList<LEGAL> moves1(game1.DebugPosition());
    MoveList<LEGAL> moves2(game2.DebugPosition());
    EXPECT_EQ(moves1.size(), moves2.size());
    for (int i = 0; i < moves1.size(); i++)
    {
        EXPECT_EQ(moves1.begin()[i].move, moves2.begin()[i].move);
    }

    // Generate priors.
    const int moveCount = static_cast<int>(moves1.size());
    std::vector<float> priors1(moveCount);
    for (int i = 0; i < moveCount; i++)
    {
        priors1[i] = (1.f / moveCount);
    }

    // Put into the cache.
    PredictionCacheChunk* chunk;
    float value;
    const bool hit1 = PredictionCache::Instance.TryGetPrediction(game1_key, moveCount, &chunk, &value, priors1.data());
    EXPECT_FALSE(hit1);
    chunk->Put(game1_key, value, moveCount, priors1.data());

    // Get from the cache.
    std::vector<float> priors2(moveCount);
    const bool hit2 = PredictionCache::Instance.TryGetPrediction(game2_key, moveCount, &chunk, &value, priors2.data());
    EXPECT_TRUE(hit2);

    // Make sure that the cached priors are close enough, up to quantization (use a more permissive epsilon).
    for (int i = 0; i < moveCount; i++)
    {
        EXPECT_NEAR(priors1[i], priors2[i], 1.f / std::numeric_limits<uint8_t>::max());
    }
}