#include <gtest/gtest.h>

#include <ChessCoach/SelfPlay.h>
#include <ChessCoach/ChessCoach.h>

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
    Position position{}; // Zero the position, since no set() call.
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