#include <iostream>
#include <thread>

#include <Stockfish/bitboard.h>
#include <Stockfish/position.h>
#include <Stockfish/thread.h>
#include <Stockfish/tt.h>
#include <Stockfish/uci.h>
#include <Stockfish/movegen.h>

#include <ChessCoach/ChessCoach.h>
#include <ChessCoach/Threading.h>
#include <ChessCoach/SelfPlay.h>

class ChessCoachTrain : public ChessCoach
{
public:

    void TrainChessCoach();
    void DebugGame();

private:

    void PlayGames(WorkCoordinator& workCoordinator, int gamesPerNetwork);
    void TrainNetwork(const SelfPlayWorker& worker, INetwork* network, int stepCount, int checkpoint);
};

int main(int argc, char* argv[])
{
    ChessCoachTrain chessCoachTrain;

    chessCoachTrain.Initialize();

#if DEBUG_MCTS
    chessCoachTrain.DebugGame();
#else
    chessCoachTrain.TrainChessCoach();
#endif

    chessCoachTrain.Finalize();

    return 0;
}

void ChessCoachTrain::TrainChessCoach()
{
    std::unique_ptr<INetwork> network(CreateNetwork());
    Storage storage;

    storage.LoadExistingGames();

    // Set up configuration for full training.
    const int networkCount = (Config::TrainingSteps / Config::CheckpointInterval);
    const int gamesPerNetwork = (Config::SelfPlayGames / networkCount);
    
    std::vector<SelfPlayWorker> selfPlayWorkers(Config::SelfPlayWorkerCount);
    std::vector<std::thread> selfPlayThreads;

    WorkCoordinator workCoordinator(static_cast<int>(selfPlayWorkers.size()));
    for (int i = 0; i < selfPlayWorkers.size(); i++)
    {
        std::cout << "Starting self-play thread " << (i + 1) << " of " << selfPlayWorkers.size() <<
            " (" << Config::PredictionBatchSize << " games per thread)" << std::endl;

        selfPlayThreads.emplace_back(&SelfPlayWorker::PlayGames, &selfPlayWorkers[i], std::ref(workCoordinator), &storage, network.get());
    }

    // If existing games found, resume progress.
    int existingNetworks = storage.CountNetworks();
    int bonusGames = std::max(0, storage.GamesPlayed() - gamesPerNetwork * existingNetworks);

    for (int n = existingNetworks; n < networkCount; n++)
    {
        // Play self-play games (skip if we already have enough to train this checkpoint).
        const int actualGames = (gamesPerNetwork - bonusGames);
        if (actualGames > 0)
        {
            PlayGames(workCoordinator, actualGames);
        }
        bonusGames = std::max(0, bonusGames - gamesPerNetwork);

        // Train the network.
        const int checkpoint = (n + 1) * Config::CheckpointInterval;
        TrainNetwork(selfPlayWorkers.front(), network.get(), Config::CheckpointInterval, checkpoint);

        // Clear the prediction cache to prepare for the new network.
        PredictionCache::Instance.PrintDebugInfo();
        PredictionCache::Instance.Clear();
        PredictionCache::Instance.PrintDebugInfo();
    }
}

void ChessCoachTrain::PlayGames(WorkCoordinator& workCoordinator, int gamesPerNetwork)
{
    std::cout << "Playing " << gamesPerNetwork << " games..." << std::endl;
    assert(gamesPerNetwork > 0);

    workCoordinator.ResetWorkItemsRemaining(gamesPerNetwork);

    workCoordinator.WaitForWorkers();
}

void ChessCoachTrain::TrainNetwork(const SelfPlayWorker& selfPlayWorker, INetwork* network, int stepCount, int checkpoint)
{
    std::cout << "Training..." << std::endl;

    selfPlayWorker.TrainNetwork(network, stepCount, checkpoint);
}

void ChessCoachTrain::DebugGame()
{
    std::unique_ptr<INetwork> network(CreateNetwork());
    Storage storage;

    SelfPlayWorker worker;
    worker.Initialize(&storage);

    StoredGame stored = storage.LoadFromDisk("path_to_game");
    worker.DebugGame(network.get(), 0, stored, 23);
}