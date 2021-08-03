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
#include <ChessCoach/PredictionCache.h>
#include <ChessCoach/ChessCoach.h>

std::vector<uint16_t> Quantize(const std::vector<float>& priors)
{
    std::vector<uint16_t> quantizedPriors(priors.size());
    for (int i = 0; i < priors.size(); i++)
    {
        quantizedPriors[i] = INetwork::QuantizeProbabilityNoZero(priors[i]);
    }
    return quantizedPriors;
}

bool TryGetPrediction(Key key, bool putOnFailedGet = true, std::vector<float> priors = { 0.1f, 0.2f, 0.3f, 0.4f })
{
    PredictionCacheChunk* chunk;
    float value;

    std::vector<uint16_t> quantizedPriors = Quantize(priors);
    std::vector<uint16_t> trashablePriors(quantizedPriors);
    bool hit = PredictionCache::Instance.TryGetPrediction(key, static_cast<int>(trashablePriors.size()), &chunk, &value, trashablePriors.data());
    if (!hit && putOnFailedGet)
    {
        chunk->Put(key, 0.33f, static_cast<int>(priors.size()), quantizedPriors.data());
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

    Key game1_key = game1.GenerateImageKey(false /* tryHard */);
    Key game2_key = game2.GenerateImageKey(false /* tryHard */);

    EXPECT_EQ(game1.GetPosition().key(), game2.GetPosition().key());
    EXPECT_EQ(game1_key, game2_key);
    EXPECT_FALSE(TryGetPrediction(game1_key));
    EXPECT_TRUE(TryGetPrediction(game2_key));

    EXPECT_TRUE(TryGetPrediction(game1_key));
    EXPECT_TRUE(TryGetPrediction(game2_key));
}

TEST(PredictionCache, SumValidation)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    const Game startingPosition;
    Game game1 = startingPosition;
    const Key game1_key1 = game1.GenerateImageKey(false /* tryHard */);

    // Simple priors
    EXPECT_FALSE(TryGetPrediction(game1_key1, true, { 0.1f, 0.2f, 0.3f, 0.4f }));
    EXPECT_TRUE(TryGetPrediction(game1_key1, false, { 0.0f, 0.0f, 0.0f, 0.0f }));
    EXPECT_FALSE(TryGetPrediction(game1_key1, false, { 0.0f, 0.0f, 0.0f }));
    EXPECT_FALSE(TryGetPrediction(game1_key1, false, { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }));

    // Trailing zeros
    EXPECT_FALSE(TryGetPrediction(game1_key1, true, { 0.1f, 0.2f, 0.3f, 0.4f, 0.f, 0.f }));
    EXPECT_TRUE(TryGetPrediction(game1_key1, false, { 0.0f, 0.0f, 0.0f, 0.0f, 0.f, 0.f }));
    EXPECT_TRUE(TryGetPrediction(game1_key1, false, { 0.0f, 0.0f, 0.0f, 0.f, 0.f })); // Unfortunately cannot detect this case: want FALSE but expect TRUE
    EXPECT_FALSE(TryGetPrediction(game1_key1, false, { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.f, 0.f }));

    // Must sum to one
    EXPECT_FALSE(TryGetPrediction(game1_key1, true, { 0.1f, 0.2f, 0.3f }));
    EXPECT_FALSE(TryGetPrediction(game1_key1, false, { 0.f, 0.f, 0.f }));

    // Avoid duplicate entries
    EXPECT_FALSE(TryGetPrediction(game1_key1, true, { 0.1f, 0.2f, 0.3f, 0.4f }));
    EXPECT_FALSE(TryGetPrediction(game1_key1, true, { 0.1f, 0.1f, 0.1f, 0.3f, 0.4f }));
    EXPECT_FALSE(TryGetPrediction(game1_key1, true, { 0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.4f }));
    EXPECT_FALSE(TryGetPrediction(game1_key1, false, { 0.0f, 0.0f, 0.0f, 0.f }));
    EXPECT_FALSE(TryGetPrediction(game1_key1, false, { 0.0f, 0.0f, 0.0f, 0.f, 0.f }));
    EXPECT_TRUE(TryGetPrediction(game1_key1, true, { 0.0f, 0.0f, 0.0f, 0.f, 0.f, 0.f }));
}

TEST(PredictionCache, PathDependence)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    Game startingPosition;
    Game game1 = startingPosition;
    Game game2 = startingPosition;

    // Make sure that two starting positions match in the prediction cache.

    Key game1_key1 = game1.GenerateImageKey(false /* tryHard */);
    Key game2_key1 = game2.GenerateImageKey(false /* tryHard */);
    
    EXPECT_EQ(game1.GetPosition().key(), game2.GetPosition().key());
    EXPECT_EQ(game1_key1, game2_key1);
    EXPECT_FALSE(TryGetPrediction(game1_key1));
    EXPECT_TRUE(TryGetPrediction(game2_key1));

    // Make different first moves. Make sure that the keys and games don't match in the prediction cache.

    game1.ApplyMove(make_move(SQ_A2, SQ_A3));
    game2.ApplyMove(make_move(SQ_B2, SQ_B3));

    Key game1_key2 = game1.GenerateImageKey(false /* tryHard */);
    Key game2_key2 = game2.GenerateImageKey(false /* tryHard */);

    EXPECT_NE(game1.GetPosition().key(), game2.GetPosition().key());
    EXPECT_NE(game1_key2, game2_key2);
    EXPECT_FALSE(TryGetPrediction(game1_key2));
    EXPECT_FALSE(TryGetPrediction(game2_key2));

    // Transpose to the same position in both games. Make sure that the position keys match but the games
    // don't match in the prediction cache because of different move paths.

    game1.ApplyMove(make_move(SQ_A7, SQ_A6));
    game2.ApplyMove(make_move(SQ_A7, SQ_A6));

    game1.ApplyMove(make_move(SQ_B2, SQ_B3));
    game2.ApplyMove(make_move(SQ_A2, SQ_A3));
    
    Key game1_key3 = game1.GenerateImageKey(false /* tryHard */);
    Key game2_key3 = game2.GenerateImageKey(false /* tryHard */);

    EXPECT_EQ(game1.GetPosition().key(), game2.GetPosition().key());
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

    Key game1_key = game1.GenerateImageKey(false /* tryHard */);
    Key game2_key = game2.GenerateImageKey(false /* tryHard */);

    EXPECT_EQ(game1.GetPosition().key(), game2.GetPosition().key());
    EXPECT_EQ(game1_key, game2_key);

    // Make sure that the two games' moves are the same.
    MoveList<LEGAL> moves1(game1.GetPosition());
    MoveList<LEGAL> moves2(game2.GetPosition());
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
    std::vector<uint16_t> quantizedPriors1 = Quantize(priors1);
    std::vector<uint16_t> trashablePriors1(quantizedPriors1);
    const bool hit1 = PredictionCache::Instance.TryGetPrediction(game1_key, moveCount, &chunk, &value, trashablePriors1.data());
    EXPECT_FALSE(hit1);
    chunk->Put(game1_key, value, moveCount, quantizedPriors1.data());

    // Get from the cache.
    std::vector<uint16_t> quantizedPriors2(moveCount);
    const bool hit2 = PredictionCache::Instance.TryGetPrediction(game2_key, moveCount, &chunk, &value, quantizedPriors2.data());
    EXPECT_TRUE(hit2);

    // Make sure that the cached priors are close enough, up to quantization (use a more permissive epsilon).
    for (int i = 0; i < moveCount; i++)
    {
        EXPECT_NEAR(INetwork::DequantizeProbabilityNoZero(quantizedPriors1[i]), INetwork::DequantizeProbabilityNoZero(quantizedPriors2[i]), 1.f / std::numeric_limits<uint8_t>::max());
    }
}