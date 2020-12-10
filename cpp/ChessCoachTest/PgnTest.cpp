#include <gtest/gtest.h>

#include <ChessCoach/ChessCoach.h>
#include <ChessCoach/Game.h>
#include <ChessCoach/Pgn.h>

struct SanTestCase
{
    std::string fen;
    Move move;
    std::string san;
};

const SanTestCase SanTestCases[] =
{
    // Basic moves
    { "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", make_move(SQ_E2, SQ_E4), "e4" },
    { "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", make_move(SQ_G8, SQ_F6), "Nf6" },
    { "r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R w KQkq - 0 1", make<CASTLING>(SQ_E1, SQ_H1), "O-O" },
    { "r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R b KQkq - 0 1", make<CASTLING>(SQ_E8, SQ_A8), "O-O-O" },

    // Captures
    { "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2", make_move(SQ_D4, SQ_E5), "dxe5" },
    { "rnbqkb1r/pppp1ppp/5n2/4p3/3P4/2K5/PPP1PPPP/RNBQ1BNR b kq - 0 3", make_move(SQ_E5, SQ_D4), "exd4+" },
    { "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3", make_move(SQ_F3, SQ_E5), "Nxe5" },

    // En passant
    { "rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", make<ENPASSANT>(SQ_E5, SQ_D6), "exd6" },
    { "rnbq1bnr/pp2pppp/2k5/2ppP3/8/5B2/PPPP1PPP/RNBQK1NR w KQ d6 0 3", make<ENPASSANT>(SQ_E5, SQ_D6), "exd6+" },

    // Promotions
    { "7k/3P2pp/8/8/8/8/3p2PP/4N2K w - - 0 0", make<PROMOTION>(SQ_D7, SQ_D8, QUEEN), "d8=Q#" },
    { "7k/3P2pp/8/8/8/8/3p2PP/4N2K b - - 0 0", make<PROMOTION>(SQ_D2, SQ_E1, BISHOP), "dxe1=B" },

    // Standard disambiguations
    { "rnbqkb1r/ppp2ppp/4pn2/3p4/3P4/2N1P3/PPP2PPP/R1BQKBNR w KQkq - 0 4", make_move(SQ_G1, SQ_E2), "Nge2" },
    { "7k/1n6/3P4/1n3K2/8/8/8/8 b - - 0 1", make_move(SQ_B5, SQ_D6), "N5xd6+" },

    // Tricky disambiguations
    { "rnbqk2r/ppp2ppp/4pn2/3p4/1bPP4/2N1P3/PP3PPP/R1BQKBNR w KQkq - 0 5", make_move(SQ_G1, SQ_E2), "Ne2" },
    { "3Rr2k/6rp/8/8/3Q4/8/8/3K4 b - - 0 1", make_move(SQ_E8, SQ_G8), "Rg8" },
    { "3k4/1Q3Q2/8/1Q6/8/8/8/3K4 w - - 0 0", make_move(SQ_B7, SQ_D7), "Qb7d7#" },

    // Castling check/checkmate
    { "rnbk1bnr/ppp1pppp/8/8/8/8/PPP1PPPP/R3KBNR w KQ - 0 1", make<CASTLING>(SQ_E1, SQ_A1), "O-O-O+" },
    { "rnb1k2r/ppppp1pp/1q6/8/7b/8/PPPPP1PP/RNBQ1K1R b kq - 0 1", make<CASTLING>(SQ_E8, SQ_H8), "O-O#" },
};

void TestSan(const std::string& fen, Move move, const std::string expectSan)
{
    Game game(fen, {});

    const std::string san = Pgn::San(game.GetPosition(), move, true /* showCheckmate */);
    EXPECT_EQ(san, expectSan);
}

void TestParseSan(const std::string& fen, const std::string& san, Move expectMove)
{
    Game game(fen, {});

    const Move move = Pgn::ParseSan(game.GetPosition(), san);
    EXPECT_EQ(type_of(move), type_of(expectMove));
    EXPECT_EQ(promotion_type(move), promotion_type(expectMove));
    EXPECT_EQ(from_sq(move), from_sq(expectMove));
    EXPECT_EQ(to_sq(move), to_sq(expectMove));
}

TEST(Pgn, San)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    for (const SanTestCase& testCase : SanTestCases)
    {
        TestSan(testCase.fen, testCase.move, testCase.san);
    }
}

TEST(Pgn, ParseSan)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    for (const SanTestCase& testCase : SanTestCases)
    {
        TestParseSan(testCase.fen, testCase.san, testCase.move);
    }
}