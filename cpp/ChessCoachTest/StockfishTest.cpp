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

#include <ChessCoach/SelfPlay.h>
#include <ChessCoach/ChessCoach.h>

TEST(Stockfish, CentipawnConversion)
{
    EXPECT_EQ(Game::CentipawnsToProbability(VALUE_DRAW), CHESSCOACH_VALUE_DRAW);
    EXPECT_EQ(Game::CentipawnsToProbability(VALUE_MATE), CHESSCOACH_VALUE_WIN);
    EXPECT_EQ(Game::CentipawnsToProbability(-VALUE_MATE), CHESSCOACH_VALUE_LOSS);

    EXPECT_EQ(Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_WIN), CHESSCOACH_VALUE_WIN);
    EXPECT_EQ(Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_DRAW), CHESSCOACH_VALUE_DRAW);
    EXPECT_EQ(Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_LOSS), CHESSCOACH_VALUE_LOSS);

    EXPECT_EQ(Game::ProbabilityToCentipawns(CHESSCOACH_VALUE_WIN), CHESSCOACH_CENTIPAWNS_WIN);
    EXPECT_EQ(Game::ProbabilityToCentipawns(CHESSCOACH_VALUE_DRAW), CHESSCOACH_CENTIPAWNS_DRAW);
    EXPECT_EQ(Game::ProbabilityToCentipawns(CHESSCOACH_VALUE_LOSS), CHESSCOACH_CENTIPAWNS_LOSS);

    EXPECT_LT(Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_WIN - 1), CHESSCOACH_VALUE_WIN);
    EXPECT_EQ(Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_WIN + 1), CHESSCOACH_VALUE_WIN);

    EXPECT_GT(Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_LOSS + 1), CHESSCOACH_VALUE_LOSS);
    EXPECT_EQ(Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_LOSS - 1), CHESSCOACH_VALUE_LOSS);

    EXPECT_GT(Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_DRAW + 1), CHESSCOACH_VALUE_DRAW);
    EXPECT_LT(Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_DRAW - 1), CHESSCOACH_VALUE_DRAW);

    const float sixDecimalPlaces = 1000000.f;
    const int insufficientQuantum = 2;
    EXPECT_EQ(std::round(sixDecimalPlaces * Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_WIN - insufficientQuantum)) / sixDecimalPlaces, CHESSCOACH_VALUE_WIN);
    EXPECT_NE(std::round(sixDecimalPlaces * Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_WIN - CHESSCOACH_CENTIPAWNS_SYZYGY_QUANTUM)) / sixDecimalPlaces, CHESSCOACH_CENTIPAWNS_WIN);
    EXPECT_NE(std::round(sixDecimalPlaces * Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_DRAW - CHESSCOACH_CENTIPAWNS_SYZYGY_QUANTUM)) / sixDecimalPlaces, CHESSCOACH_CENTIPAWNS_DRAW);
    EXPECT_NE(std::round(sixDecimalPlaces * Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_DRAW + CHESSCOACH_CENTIPAWNS_SYZYGY_QUANTUM)) / sixDecimalPlaces, CHESSCOACH_CENTIPAWNS_DRAW);
    EXPECT_NE(std::round(sixDecimalPlaces * Game::CentipawnsToProbability(CHESSCOACH_CENTIPAWNS_LOSS + CHESSCOACH_CENTIPAWNS_SYZYGY_QUANTUM)) / sixDecimalPlaces, CHESSCOACH_CENTIPAWNS_LOSS);
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