#include <gtest/gtest.h>

#include <filesystem>
#include <cstdio>
#include <vector>
#include <algorithm>

#include <ChessCoach/SelfPlay.h>
#include <ChessCoach/ChessCoach.h>

void ApplyMove(SelfPlayGame& game, Move move)
{
    Node* child = new Node(0.f);
    child->visitCount++;
    game.Root()->children[move] = child;
    game.StoreSearchStatistics();
    game.ApplyMoveWithRootAndHistory(move, child);
}

TEST(Storage, Basic)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    // Initialize the Node allocator.
    SelfPlayWorker selfPlayWorker;
    selfPlayWorker.Initialize(nullptr /* storage */);

    std::filesystem::path tempPath = std::filesystem::temp_directory_path();
    
#pragma warning(disable:4996) // Internal buffer is immediately consumed and detached.
    std::filesystem::path gamesPath = tempPath / std::tmpnam(nullptr);
    std::filesystem::create_directory(gamesPath);
    std::filesystem::path networksPath = tempPath / std::tmpnam(nullptr);
    std::filesystem::create_directory(networksPath);
#pragma warning(default:4996) // Internal buffer is immediately consumed and detached.

    std::unique_ptr<Storage> storage1(new Storage(gamesPath, networksPath));

    // Expect zero games in a new, temporary directory.

    EXPECT_EQ(storage1->GamesPlayed(), 0);
    EXPECT_EQ(storage1->CountNetworks(), 0);

    storage1->LoadExistingGames();
    EXPECT_EQ(storage1->GamesPlayed(), 0);

    // Play out some moves and check the StoredGame.

    SelfPlayGame game(nullptr, nullptr, nullptr);
    std::vector<Move> moves = { make_move(SQ_A2, SQ_A3), make_move(SQ_A7, SQ_A6), make_move(SQ_A3, SQ_A4), make_move(SQ_A6, SQ_A5) };
    for (const Move& move : moves)
    {
        ApplyMove(game, move);
    }
    game.Complete();

    StoredGame stored = game.Store();
    EXPECT_EQ(stored.result, CHESSCOACH_VALUE_DRAW);
    EXPECT_EQ(stored.moveCount, moves.size());
    EXPECT_EQ(stored.moves.size(), moves.size());
    EXPECT_EQ(stored.childVisits.size(), moves.size());

    // Add to storage.

    storage1->AddGame(StoredGame(stored));
    EXPECT_EQ(storage1->GamesPlayed(), 1);

    // Load a second Storage over the same directories and check that game loads.

    std::unique_ptr<Storage> storage2(new Storage(gamesPath, networksPath));
    storage2->LoadExistingGames();

    EXPECT_EQ(storage2->GamesPlayed(), 1);
    EXPECT_EQ(storage2->CountNetworks(), 0);
}

TEST(Storage, SampleBatch)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    // Initialize the Node allocator.
    SelfPlayWorker selfPlayWorker;
    selfPlayWorker.Initialize(nullptr /* storage */);

    std::filesystem::path tempPath = std::filesystem::temp_directory_path();

#pragma warning(disable:4996) // Internal buffer is immediately consumed and detached.
    std::filesystem::path gamesPath = tempPath / std::tmpnam(nullptr);
    std::filesystem::create_directory(gamesPath);
    std::filesystem::path networksPath = tempPath / std::tmpnam(nullptr);
    std::filesystem::create_directory(networksPath);
#pragma warning(default:4996) // Internal buffer is immediately consumed and detached.

    std::unique_ptr<Storage> storage1(new Storage(gamesPath, networksPath));

    // Expect zero games in a new, temporary directory.

    EXPECT_EQ(storage1->GamesPlayed(), 0);
    EXPECT_EQ(storage1->CountNetworks(), 0);

    storage1->LoadExistingGames();
    EXPECT_EQ(storage1->GamesPlayed(), 0);

    // Play exactly two moves.

    SelfPlayGame game(nullptr, nullptr, nullptr);
    const SelfPlayGame gameWhiteToPlay = game;

    ApplyMove(game, make_move(SQ_A2, SQ_A3));
    const SelfPlayGame gameBlackToPlay = game;

    ApplyMove(game, make_move(SQ_A7, SQ_A6));
    game.Complete();

    StoredGame stored = game.Store();
    EXPECT_EQ(stored.result, CHESSCOACH_VALUE_DRAW);
    EXPECT_EQ(stored.moveCount, 2);
    EXPECT_EQ(stored.moves.size(), 2);
    EXPECT_EQ(stored.childVisits.size(), 2);

    // Fudge the result to a win for white.
    stored.result = CHESSCOACH_VALUE_WIN;

    // Add to storage.

    storage1->AddGame(StoredGame(stored));
    EXPECT_EQ(storage1->GamesPlayed(), 1);

    // Load a second Storage over the same directories and check that game loads.

    std::unique_ptr<Storage> storage2(new Storage(gamesPath, networksPath));
    storage2->LoadExistingGames();

    EXPECT_EQ(storage2->GamesPlayed(), 1);
    EXPECT_EQ(storage2->CountNetworks(), 0);

    // Sample a batch from the second Storage and make sure that the data is consistent.

    const TrainingBatch* batch = storage2->SampleBatch();
    int indexWhiteToPlay = static_cast<int>(std::distance(batch->values.begin(), std::find(batch->values.begin(), batch->values.end(), CHESSCOACH_VALUE_WIN)));
    int indexBlackToPlay = static_cast<int>(std::distance(batch->values.begin(), std::find(batch->values.begin(), batch->values.end(), CHESSCOACH_VALUE_LOSS)));

    EXPECT_EQ(gameWhiteToPlay.GenerateImage(), batch->images[indexWhiteToPlay]);
    EXPECT_EQ(gameBlackToPlay.GenerateImage(), batch->images[indexBlackToPlay]);

    EXPECT_EQ(gameWhiteToPlay.GeneratePolicy(stored.childVisits[0]), batch->policies[indexWhiteToPlay]);
    EXPECT_EQ(gameBlackToPlay.GeneratePolicy(stored.childVisits[1]), batch->policies[indexBlackToPlay]);
}