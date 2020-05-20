#include <iostream>
#include <thread>
#include <algorithm>

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

    void PlayGames(WorkCoordinator& workCoordinator, int gameCount);
    void TrainNetwork(SelfPlayWorker& worker, INetwork* network, int stepCount, int checkpoint);
};

int main(int argc, char* argv[])
{
    ChessCoachTrain chessCoachTrain;

    chessCoachTrain.PrintExceptions();
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
    const NetworkConfig& config = Config::TrainingNetwork;
    std::unique_ptr<INetwork> network(CreateNetwork(config));
    Storage storage(config, Config::Misc);

    // Load existing games.
    storage.LoadExistingGames(GameType_Training, 500000/*std::numeric_limits<int>::max()*/);
    storage.LoadExistingGames(GameType_Validation, 50000/*std::numeric_limits<int>::max()*/);

    // Start self-play worker threads.
    std::vector<std::unique_ptr<SelfPlayWorker>> selfPlayWorkers(config.SelfPlay.NumWorkers);
    std::vector<std::thread> selfPlayThreads;

    WorkCoordinator workCoordinator(config.SelfPlay.NumWorkers);
    for (int i = 0; i < config.SelfPlay.NumWorkers; i++)
    {
        std::cout << "Starting self-play thread " << (i + 1) << " of " << selfPlayWorkers.size() <<
            " (" << config.SelfPlay.PredictionBatchSize << " games per thread)" << std::endl;

        selfPlayWorkers[i].reset(new SelfPlayWorker(config, &storage));
        selfPlayThreads.emplace_back(&SelfPlayWorker::PlayGames, selfPlayWorkers[i].get(), std::ref(workCoordinator), network.get());
    }

    // Wait until all self-play workers are initialized.
    workCoordinator.WaitForWorkers();

    // Plan full training and resume progress. If the network's step count isn't a multiple of the checkpoint interval, round down.
    const int networkCount = (config.Training.Steps / config.Training.CheckpointInterval);
    const int gamesPerNetwork = (config.Training.NumGames / networkCount);
    const int startingNetwork = (storage.NetworkStepCount(config.Name) / config.Training.CheckpointInterval);
    int gameCount = storage.GamesPlayed(GameType_Training);

    // Run through all checkpoints with n in [1, networkCount].
    for (int n = startingNetwork + 1; n <= networkCount; n++)
    {
        // Play self-play games (skip if we already have enough to train this checkpoint).
        const int gameTarget = (n * gamesPerNetwork);
        const int gamesToPlay = std::max(0, gameTarget - gameCount);
        if (gamesToPlay > 0)
        {
            PlayGames(workCoordinator, gamesToPlay);
            gameCount += gamesToPlay;
        }

        // Train the network.
        const int checkpoint = (n * config.Training.CheckpointInterval);
        TrainNetwork(*selfPlayWorkers.front(), network.get(), config.Training.CheckpointInterval, checkpoint);

        // Clear the prediction cache to prepare for the new network.
        if (gamesToPlay > 0)
        {
            PredictionCache::Instance.PrintDebugInfo();
            PredictionCache::Instance.Clear();
        }
    }
}

void ChessCoachTrain::PlayGames(WorkCoordinator& workCoordinator, int gameCount)
{
    std::cout << "Playing " << gameCount << " games..." << std::endl;
    assert(gameCount > 0);

    workCoordinator.ResetWorkItemsRemaining(gameCount);

    workCoordinator.WaitForWorkers();
}

void ChessCoachTrain::TrainNetwork(SelfPlayWorker& selfPlayWorker, INetwork* network, int stepCount, int checkpoint)
{
    std::cout << "Training..." << std::endl;

    selfPlayWorker.TrainNetwork(network, stepCount, checkpoint);
}

void ChessCoachTrain::DebugGame()
{
    std::unique_ptr<INetwork> network(CreateNetwork(Config::TrainingNetwork));
    Storage storage(Config::TrainingNetwork, Config::Misc);

    SelfPlayWorker worker(Config::TrainingNetwork, &storage);

    SavedGame saved = storage.LoadSingleGameFromDisk("path_to_game");
    worker.DebugGame(network.get(), 0, saved, 23);
}