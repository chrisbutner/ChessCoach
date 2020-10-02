#include <gtest/gtest.h>

#include <filesystem>
#include <cstdio>
#include <vector>
#include <algorithm>

#include <ChessCoach/SelfPlay.h>
#include <ChessCoach/ChessCoach.h>

void ApplyMove(SelfPlayGame& game, Move move)
{
    Node* child = new Node(move, 0.f);
    game.Root()->firstChild = child;
    for (Node* node = game.Root(); node != nullptr; node = node->firstChild)
    {
        node->visitCount++;
    }
    game.StoreSearchStatistics();
    game.ApplyMoveWithRootAndHistory(move, child);
}

void Basic(GameType gameType)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    std::filesystem::path tempPath = std::filesystem::temp_directory_path();
    
#pragma warning(disable:4996) // Internal buffer is immediately consumed and detached.
    std::filesystem::path gamesSupervisedPath = tempPath / std::tmpnam(nullptr);
    std::filesystem::create_directory(gamesSupervisedPath);
    std::filesystem::path gamesTrainingPath = tempPath / std::tmpnam(nullptr);
    std::filesystem::create_directory(gamesTrainingPath);
    std::filesystem::path gamesValidationPath = tempPath / std::tmpnam(nullptr);
    std::filesystem::create_directory(gamesValidationPath);
    std::filesystem::path pgnsPath = tempPath / std::tmpnam(nullptr);
    std::filesystem::create_directory(pgnsPath);
    std::filesystem::path networksPath = tempPath / std::tmpnam(nullptr);
    std::filesystem::create_directory(networksPath);
#pragma warning(default:4996) // Internal buffer is immediately consumed and detached.

    std::unique_ptr<Storage> storage1(new Storage(Config::TrainingNetwork,
        gamesSupervisedPath, gamesTrainingPath, gamesValidationPath, pgnsPath, networksPath));

    // Expect zero games in a new, temporary directory.

    EXPECT_EQ(storage1->GamesPlayed(gameType), 0);
    storage1->LoadExistingGames(gameType, std::numeric_limits<int>::max());
    EXPECT_EQ(storage1->GamesPlayed(gameType), 0);

    // Play out some moves and check the StoredGame.

    SelfPlayGame game(nullptr, nullptr, nullptr);
    std::vector<Move> moves = { make_move(SQ_A2, SQ_A3), make_move(SQ_A7, SQ_A6), make_move(SQ_A3, SQ_A4), make_move(SQ_A6, SQ_A5) };
    for (const Move& move : moves)
    {
        ApplyMove(game, move);
    }
    game.Complete();

    SavedGame saved = game.Save();
    EXPECT_EQ(saved.result, CHESSCOACH_VALUE_DRAW);
    EXPECT_EQ(saved.moveCount, moves.size());
    EXPECT_EQ(saved.moves.size(), moves.size());
    EXPECT_EQ(saved.childVisits.size(), moves.size());

    // Add to storage.

    storage1->AddGame(gameType, SavedGame(saved));
    EXPECT_EQ(storage1->GamesPlayed(gameType), 1);

    // Load a second Storage over the same directories and check that game loads.

    std::unique_ptr<Storage> storage2(new Storage(Config::TrainingNetwork,
        gamesSupervisedPath, gamesTrainingPath, gamesValidationPath, pgnsPath, networksPath));
    storage2->LoadExistingGames(gameType, std::numeric_limits<int>::max());

    EXPECT_EQ(storage2->GamesPlayed(gameType), 1);
}

void SampleBatch(GameType gameType)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    std::filesystem::path tempPath = std::filesystem::temp_directory_path();

#pragma warning(disable:4996) // Internal buffer is immediately consumed and detached.
    std::filesystem::path gamesSupervisedPath = tempPath / std::tmpnam(nullptr);
    std::filesystem::create_directory(gamesSupervisedPath);
    std::filesystem::path gamesTrainingPath = tempPath / std::tmpnam(nullptr);
    std::filesystem::create_directory(gamesTrainingPath);
    std::filesystem::path gamesValidationPath = tempPath / std::tmpnam(nullptr);
    std::filesystem::create_directory(gamesValidationPath);
    std::filesystem::path pgnsPath = tempPath / std::tmpnam(nullptr);
    std::filesystem::create_directory(pgnsPath);
    std::filesystem::path networksPath = tempPath / std::tmpnam(nullptr);
    std::filesystem::create_directory(networksPath);
#pragma warning(default:4996) // Internal buffer is immediately consumed and detached.

    // Set up storage and a simple sampling window.
    std::unique_ptr<Storage> storage1(new Storage(Config::TrainingNetwork,
        gamesSupervisedPath, gamesTrainingPath, gamesValidationPath, pgnsPath, networksPath));
    storage1->SetWindow(gameType, { 0, 1, 1 });

    // Expect zero games in a new, temporary directory.

    EXPECT_EQ(storage1->GamesPlayed(gameType), 0);
    storage1->LoadExistingGames(gameType, std::numeric_limits<int>::max());
    EXPECT_EQ(storage1->GamesPlayed(gameType), 0);

    // Play exactly two moves.

    SelfPlayGame game(nullptr, nullptr, nullptr);
    const SelfPlayGame gameWhiteToPlay = game;

    ApplyMove(game, make_move(SQ_A2, SQ_A3));
    const SelfPlayGame gameBlackToPlay = game;

    ApplyMove(game, make_move(SQ_A7, SQ_A6));
    game.Complete();

    SavedGame saved = game.Save();
    EXPECT_EQ(saved.result, CHESSCOACH_VALUE_DRAW);
    EXPECT_EQ(saved.moveCount, 2);
    EXPECT_EQ(saved.moves.size(), 2);
    EXPECT_EQ(saved.childVisits.size(), 2);

    // Fudge the result to a win for white.
    saved.result = CHESSCOACH_VALUE_WIN;

    // Add to storage.

    storage1->AddGame(gameType, SavedGame(saved));
    EXPECT_EQ(storage1->GamesPlayed(gameType), 1);

    // Load a second Storage over the same directories with a simple sampling window and check that the game loads.

    std::unique_ptr<Storage> storage2(new Storage(Config::TrainingNetwork,
        gamesSupervisedPath, gamesTrainingPath, gamesValidationPath, pgnsPath, networksPath));
    storage2->SetWindow(gameType, { 0, 1, 1 });
    storage2->LoadExistingGames(gameType, std::numeric_limits<int>::max());

    EXPECT_EQ(storage2->GamesPlayed(gameType), 1);

    // Sample a batch from the second Storage and make sure that the data is consistent.

    const TrainingBatch* batch = storage2->SampleBatch(gameType);
    int indexWhiteToPlay = static_cast<int>(std::distance(batch->values.begin(), std::find(batch->values.begin(), batch->values.end(), CHESSCOACH_VALUE_WIN)));
    int indexBlackToPlay = static_cast<int>(std::distance(batch->values.begin(), std::find(batch->values.begin(), batch->values.end(), CHESSCOACH_VALUE_LOSS)));

    EXPECT_EQ(gameWhiteToPlay.GenerateImage(), batch->images[indexWhiteToPlay]);
    EXPECT_EQ(gameBlackToPlay.GenerateImage(), batch->images[indexBlackToPlay]);

    EXPECT_EQ(gameWhiteToPlay.GeneratePolicy(saved.childVisits[0]), batch->policies[indexWhiteToPlay]);
    EXPECT_EQ(gameBlackToPlay.GeneratePolicy(saved.childVisits[1]), batch->policies[indexBlackToPlay]);
}

static_assert(GameType_Count == 3);

TEST(Storage, Basic_Supervised)
{
    Basic(GameType_Supervised);
}

TEST(Storage, Basic_Training)
{
    Basic(GameType_Training);
}

TEST(Storage, Basic_Validation)
{
    Basic(GameType_Validation);
}

TEST(Storage, SampleBatch_Supervised)
{
    SampleBatch(GameType_Supervised);
}

TEST(Storage, SampleBatch_Training)
{
    SampleBatch(GameType_Training);
}

TEST(Storage, SampleBatch_Validation)
{
    SampleBatch(GameType_Validation);
}