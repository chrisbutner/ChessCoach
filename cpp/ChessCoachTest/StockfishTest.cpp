#include <gtest/gtest.h>

#include <ChessCoach/SelfPlay.h>
#include <ChessCoach/ChessCoach.h>

TEST(Stockfish, Evaluation)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    Game whiteToPlayWhiteQueen = Game("rnb1kbnr/ppp1pppp/8/3Q4/8/8/PPP1PPPP/RNB1KBNR w KQkq - 0 1");

    EXPECT_TRUE(whiteToPlayWhiteQueen.StockfishCanEvaluate());
    EXPECT_GT(whiteToPlayWhiteQueen.StockfishEvaluation(), 0.95f);
    
    Game blackToPlayWhiteQueen = whiteToPlayWhiteQueen;
    blackToPlayWhiteQueen.ApplyMove(make_move(SQ_A2, SQ_A3));

    EXPECT_TRUE(blackToPlayWhiteQueen.StockfishCanEvaluate());
    EXPECT_LT(blackToPlayWhiteQueen.StockfishEvaluation(), 0.05f);

    Game blackToPlayBlackQueen = Game("rnb1kbnr/ppp1pppp/8/8/3q4/8/PPP1PPPP/RNB1KBNR b KQkq - 0 1");

    EXPECT_TRUE(blackToPlayBlackQueen.StockfishCanEvaluate());
    EXPECT_GT(blackToPlayBlackQueen.StockfishEvaluation(), 0.95f);

    Game whiteToPlayBlackQueen = blackToPlayBlackQueen;
    whiteToPlayBlackQueen.ApplyMove(make_move(SQ_A7, SQ_A6));

    EXPECT_TRUE(whiteToPlayBlackQueen.StockfishCanEvaluate());
    EXPECT_LT(whiteToPlayBlackQueen.StockfishEvaluation(), 0.05f);

    Game inCheck = Game("rnbqkbnr/ppp1pppp/8/1Q6/8/8/PPP1PPPP/RNB1KBNR b KQkq - 0 1");

    EXPECT_FALSE(inCheck.StockfishCanEvaluate());
}

TEST(Stockfish, CentipawnConversion)
{
    const float epsilon = 0.001f;

    EXPECT_NEAR(Game::CentipawnsToProbability(0), CHESSCOACH_VALUE_DRAW, epsilon);
    EXPECT_NEAR(Game::CentipawnsToProbability(12800), CHESSCOACH_VALUE_WIN, epsilon);
    EXPECT_NEAR(Game::CentipawnsToProbability(-12800), CHESSCOACH_VALUE_LOSS, epsilon);
    EXPECT_NEAR(Game::CentipawnsToProbability(32000), CHESSCOACH_VALUE_WIN, epsilon);
    EXPECT_NEAR(Game::CentipawnsToProbability(-32000), CHESSCOACH_VALUE_LOSS, epsilon);
}