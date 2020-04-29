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

    void PlayGames(WorkCoordinator& workCoordinator, int gamesPerNetwork);
    void TrainNetwork(SelfPlayWorker& worker, INetwork* network, GameType gameType, int stepCount, int checkpoint);
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
    std::unique_ptr<INetwork> network(CreateNetwork(Config::TrainingNetwork));
    Storage storage;

    storage.LoadExistingGames(GameType_Train, std::numeric_limits<int>::max());
    storage.LoadExistingGames(GameType_Test, std::numeric_limits<int>::max());

    // Set up configuration for full training.
    const int networkCount = (config.Training.Steps / config.Training.CheckpointInterval);
    const int gamesPerNetwork = (config.Training.NumGames / networkCount);
    
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

    // Wait until all workers are initialized.
    workCoordinator.WaitForWorkers();

        //// (Temporarily testing supervised learning)
        //int existingSupervisedNetworks = storage.CountNetworks();
        //storage.LoadExistingGames(GameType_Supervised, std::numeric_limits<int>::max());
        //const int supervisedStepCount = static_cast<int>(Config::SampleBatchesPerGame * storage.GamesPlayed(GameType_Supervised));
        //const int supervisedNetworkCount = (supervisedStepCount / Config::CheckpointInterval[GameType_Supervised]);
        //const int overtrainMultiple = 2;
        //for (int n = existingSupervisedNetworks; n < overtrainMultiple * supervisedNetworkCount; n++)
        //{
        //    const int checkpoint = (n + 1) * Config::CheckpointInterval[GameType_Supervised];
        //    TrainNetwork(selfPlayWorkers.front(), network.get(), GameType_Supervised, Config::CheckpointInterval[GameType_Supervised], checkpoint);
        //}
        //return;

    // If existing games found, resume progress.
    int existingNetworks = (storage.NetworkStepCount(config.Name) / config.Training.CheckpointInterval);
    int bonusGames = std::max(0, storage.GamesPlayed(GameType_Train) - gamesPerNetwork * existingNetworks);

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
        const int checkpoint = (n + 1) * config.Training.CheckpointInterval;
        TrainNetwork(*selfPlayWorkers.front(), network.get(), GameType_Train, config.Training.CheckpointInterval, checkpoint);

        // Clear the prediction cache to prepare for the new network.
        PredictionCache::Instance.PrintDebugInfo();
        PredictionCache::Instance.Clear();
    }
}

void ChessCoachTrain::PlayGames(WorkCoordinator& workCoordinator, int gamesPerNetwork)
{
    std::cout << "Playing " << gamesPerNetwork << " games..." << std::endl;
    assert(gamesPerNetwork > 0);

    workCoordinator.ResetWorkItemsRemaining(gamesPerNetwork);

    workCoordinator.WaitForWorkers();
}

void ChessCoachTrain::TrainNetwork(SelfPlayWorker& selfPlayWorker, INetwork* network, GameType gameType, int stepCount, int checkpoint)
{
    std::cout << "Training (" << GameTypeNames[gameType] << ")..." << std::endl;

    selfPlayWorker.TrainNetwork(network, gameType, stepCount, checkpoint);
}

void ChessCoachTrain::DebugGame()
{
    std::unique_ptr<INetwork> network(CreateNetwork(Config::TrainingNetwork));
    Storage storage;

    SelfPlayWorker worker(Config::TrainingNetwork, &storage);

    SavedGame saved = storage.LoadSingleGameFromDisk("path_to_game");
    worker.DebugGame(network.get(), 0, saved, 23);
}