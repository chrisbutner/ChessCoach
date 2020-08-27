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

private:

    void PlayGames(WorkCoordinator& workCoordinator, int gameCount);
    void TrainNetwork(SelfPlayWorker& worker, INetwork* network, int stepCount, int checkpoint);
    Window CalculateWindow(const NetworkConfig& config, int totalGamesCount, int networkCount, int network);
};

int main(int argc, char* argv[])
{
    ChessCoachTrain chessCoachTrain;

    chessCoachTrain.PrintExceptions();
    chessCoachTrain.Initialize();

    chessCoachTrain.TrainChessCoach();

    chessCoachTrain.Finalize();

    return 0;
}

void ChessCoachTrain::TrainChessCoach()
{
    const NetworkConfig& config = Config::TrainingNetwork;
    std::unique_ptr<INetwork> network(CreateNetwork(config));
    Storage storage(config, Config::Misc);
    bool loadedGames = false;

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
    const int totalGamesCount = config.Training.NumGames;
    const int networkCount = (config.Training.Steps / config.Training.CheckpointInterval);
    const int startingNetwork = (storage.NetworkStepCount(config.Name) / config.Training.CheckpointInterval);

    // Run through all checkpoints with n in [1, networkCount].
    for (int n = startingNetwork + 1; n <= networkCount; n++)
    {
        // Configure the replay buffer windows for position sampling (neural network training).
        const Window trainingWindow = CalculateWindow(config, totalGamesCount, networkCount, n);
        const Window fullWindow = { 0, std::numeric_limits<int>::max(), config.Training.BatchSize };
        storage.SetWindow(GameType_Training, trainingWindow);
        storage.SetWindow(GameType_Validation, fullWindow);
        static_assert(GameType_Count == 2);

        // Load games if we haven't yet.
        if (!loadedGames)
        {
            loadedGames = true;
            storage.LoadExistingGames(GameType_Training, std::numeric_limits<int>::max());
            storage.LoadExistingGames(GameType_Validation, std::numeric_limits<int>::max());
        }

        // Play self-play games (skip if we already have enough to train this checkpoint).
        const int gameTarget = trainingWindow.TrainingGameMax;
        const int gameCount = storage.GamesPlayed(GameType_Training);
        const int gamesToPlay = std::max(0, gameTarget - gameCount);
        if (gamesToPlay > 0)
        {
            PlayGames(workCoordinator, gamesToPlay);
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

Window ChessCoachTrain::CalculateWindow(const NetworkConfig& config, int totalGamesCount, int networkCount, int network)
{
    const int gamesPerNetworkAfterFirstWindow = ((totalGamesCount - config.Training.WindowSizeStart) / networkCount);
    const int gameTarget = (config.Training.WindowSizeStart + ((network - 1) * gamesPerNetworkAfterFirstWindow));
    const float t = (static_cast<float>(network - 1) / (networkCount - 1));
    const int windowSize = static_cast<int>(config.Training.WindowSizeStart +
        t * (config.Training.WindowSizeFinish - config.Training.WindowSizeStart));

    // Min is inclusive, max is exclusive, both 0-based.
    return { std::max(0, gameTarget - windowSize), gameTarget, config.Training.BatchSize };
}