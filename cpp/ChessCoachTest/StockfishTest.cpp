#include <gtest/gtest.h>

#include <ChessCoach/SelfPlay.h>
#include <ChessCoach/ChessCoach.h>

TEST(Stockfish, Evaluation)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    Game whiteToPlayWhiteQueen = Game("rnb1kbnr/ppp1pppp/8/3Q4/8/8/PPP1PPPP/RNB1KBNR w KQkq - 0 1", {});

    EXPECT_TRUE(whiteToPlayWhiteQueen.StockfishCanEvaluate());
    EXPECT_GT(whiteToPlayWhiteQueen.StockfishEvaluation(), 0.95f);
    
    Game blackToPlayWhiteQueen = whiteToPlayWhiteQueen;
    blackToPlayWhiteQueen.ApplyMove(make_move(SQ_A2, SQ_A3));

    EXPECT_TRUE(blackToPlayWhiteQueen.StockfishCanEvaluate());
    EXPECT_LT(blackToPlayWhiteQueen.StockfishEvaluation(), 0.05f);

    Game blackToPlayBlackQueen = Game("rnb1kbnr/ppp1pppp/8/8/3q4/8/PPP1PPPP/RNB1KBNR b KQkq - 0 1", {});

    EXPECT_TRUE(blackToPlayBlackQueen.StockfishCanEvaluate());
    EXPECT_GT(blackToPlayBlackQueen.StockfishEvaluation(), 0.95f);

    Game whiteToPlayBlackQueen = blackToPlayBlackQueen;
    whiteToPlayBlackQueen.ApplyMove(make_move(SQ_A7, SQ_A6));

    EXPECT_TRUE(whiteToPlayBlackQueen.StockfishCanEvaluate());
    EXPECT_LT(whiteToPlayBlackQueen.StockfishEvaluation(), 0.05f);

    Game inCheck = Game("rnbqkbnr/ppp1pppp/8/1Q6/8/8/PPP1PPPP/RNB1KBNR b KQkq - 0 1", {});

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

TEST(Stockfish, EmptyPosition)
{
    Position position{};
    EXPECT_EQ(position.state_info(), nullptr);

    for (Rank rank = RANK_1; rank <= RANK_8; ++rank)
    {
        for (File file = FILE_A; file <= FILE_H; ++file)
        {
            EXPECT_EQ(position.piece_on(make_square(file, rank)), NO_PIECE);
        }
    }

    EXPECT_EQ(position.pieces(WHITE, PAWN), 0);
    EXPECT_EQ(position.pieces(WHITE, KNIGHT), 0);
    EXPECT_EQ(position.pieces(WHITE, BISHOP), 0);
    EXPECT_EQ(position.pieces(WHITE, ROOK), 0);
    EXPECT_EQ(position.pieces(WHITE, QUEEN), 0);
    EXPECT_EQ(position.pieces(WHITE, KING), 0);

    EXPECT_EQ(position.pieces(BLACK, PAWN), 0);
    EXPECT_EQ(position.pieces(BLACK, KNIGHT), 0);
    EXPECT_EQ(position.pieces(BLACK, BISHOP), 0);
    EXPECT_EQ(position.pieces(BLACK, ROOK), 0);
    EXPECT_EQ(position.pieces(BLACK, QUEEN), 0);
    EXPECT_EQ(position.pieces(BLACK, KING), 0);
}

TEST(Stockfish, EmptyGamePositionHistory)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    Game game;

    for (int i = 0; i < INetwork::InputPreviousPositionCount; i++)
    {
        const Position& position = game.DebugPreviousPosition(i);
        EXPECT_EQ(position.state_info(), nullptr);

        for (Rank rank = RANK_1; rank <= RANK_8; ++rank)
        {
            for (File file = FILE_A; file <= FILE_H; ++file)
            {
                EXPECT_EQ(position.piece_on(make_square(file, rank)), NO_PIECE);
            }
        }
    }
}