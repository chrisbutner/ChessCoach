#include <gtest/gtest.h>

#include <ChessCoach/SelfPlay.h>
#include <ChessCoach/ChessCoach.h>

TEST(Game, Flip)
{
    Move move = make_move(SQ_B3, SQ_E5);
    Move moveFlipped = make_move(SQ_B6, SQ_E4);

    // Flip moves.
    EXPECT_EQ(Game::FlipMove(WHITE, move), move);
    EXPECT_EQ(Game::FlipMove(BLACK, move), moveFlipped);
    EXPECT_EQ(Game::FlipMove(WHITE, moveFlipped), moveFlipped);
    EXPECT_EQ(Game::FlipMove(BLACK, moveFlipped), move);

    // Flip squares.
    EXPECT_EQ(Game::FlipSquare(WHITE, from_sq(move)), from_sq(move));
    EXPECT_EQ(Game::FlipSquare(WHITE, to_sq(move)), to_sq(move));
    EXPECT_EQ(Game::FlipSquare(BLACK, from_sq(move)), from_sq(moveFlipped));
    EXPECT_EQ(Game::FlipSquare(BLACK, to_sq(move)), to_sq(moveFlipped));
    EXPECT_EQ(Game::FlipSquare(WHITE, from_sq(moveFlipped)), from_sq(moveFlipped));
    EXPECT_EQ(Game::FlipSquare(WHITE, to_sq(moveFlipped)), to_sq(moveFlipped));
    EXPECT_EQ(Game::FlipSquare(BLACK, from_sq(moveFlipped)), from_sq(move));
    EXPECT_EQ(Game::FlipSquare(BLACK, to_sq(moveFlipped)), to_sq(move));

    // Flipping value is statically tested enough in Game.h.

    // Flip pieces.
    std::array<Piece, 6> whitePieces = { W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING };
    std::array<Piece, 6> blackPieces = { B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING };
    EXPECT_EQ(Game::FlipPiece[WHITE][NO_PIECE], NO_PIECE);
    EXPECT_EQ(Game::FlipPiece[BLACK][NO_PIECE], NO_PIECE);
    for (int i = 0; i < whitePieces.size(); i++)
    {
        EXPECT_EQ(Game::FlipPiece[WHITE][whitePieces[i]], whitePieces[i]);
        EXPECT_EQ(Game::FlipPiece[BLACK][whitePieces[i]], blackPieces[i]);
        EXPECT_EQ(Game::FlipPiece[WHITE][blackPieces[i]], blackPieces[i]);
        EXPECT_EQ(Game::FlipPiece[BLACK][blackPieces[i]], whitePieces[i]);
    }
}