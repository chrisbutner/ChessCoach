#include <iostream>
#include <thread>
#include <algorithm>
#include <functional>

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

    void StagePlay(const StageConfig& stage, const Storage& storage, Window trainingWindow, WorkCoordinator& workCoordinator);
    void StageTrain(const StageConfig& stage, Storage& storage, Window trainingWindow,
        SelfPlayWorker& selfPlayWorker, INetwork* network, int stepCount, int checkpoint);
    void StageTrainCommentary(const StageConfig& stage, SelfPlayWorker& selfPlayWorker, INetwork* network, int stepCount, int checkpoint);
    void StageSave(const StageConfig& stage, SelfPlayWorker& selfPlayWorker, INetwork* network, int checkpoint);
    void StageStrengthTest(const StageConfig& stage, SelfPlayWorker& selfPlayWorker, INetwork* network, int checkpoint);

    Window CalculateWindow(const NetworkConfig& config, const StageConfig& stageConfig, int totalGamesCount, int networkCount, int network);
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

    // Load games, up to the total number required for training.
    // It can be dangerous to download partial games then self-play, since new games overwrite the ones not yet loaded,
    // but self-play shouldn't need to generate any games beyond the total required for training anyway.
    storage.LoadExistingGames(GameType_Supervised, totalGamesCount);
    storage.LoadExistingGames(GameType_Training, totalGamesCount);
    storage.LoadExistingGames(GameType_Validation, totalGamesCount);

    // Always use full window for validation.
    const int minimumSamplableGames = std::min(totalGamesCount, config.Training.BatchSize);
    const Window fullWindow = { 0, std::numeric_limits<int>::max(), minimumSamplableGames };
    storage.SetWindow(GameType_Validation, fullWindow);

    // Run through all checkpoints with n in [1, networkCount].
    for (int n = startingNetwork + 1; n <= networkCount; n++)
    {
        const int checkpoint = (n * config.Training.CheckpointInterval);

        // Run through all stages in the checkpoint.
        for (const StageConfig& stage : config.Training.Stages)
        {
            // Calculate the replay buffer window for position sampling (network training).
            const Window trainingWindow = CalculateWindow(config, stage, totalGamesCount, networkCount, n);
            
            // Run the stage.
            switch (stage.Stage)
            {
            case StageType_Play:
                StagePlay(stage, storage, trainingWindow, workCoordinator);
                break;
            case StageType_Train:
                StageTrain(stage, storage, trainingWindow, *selfPlayWorkers.front(), network.get(),
                    config.Training.CheckpointInterval, checkpoint);
                break;
            case StageType_TrainCommentary:
                StageTrainCommentary(stage, *selfPlayWorkers.front(), network.get(),
                    config.Training.CheckpointInterval, checkpoint);
                break;
            case StageType_Save:
                StageSave(stage, *selfPlayWorkers.front(), network.get(), checkpoint);
                break;
            case StageType_StrengthTest:
                StageStrengthTest(stage, *selfPlayWorkers.front(), network.get(), checkpoint);
                break;
            }
        }
    }
}

void ChessCoachTrain::StagePlay(const StageConfig& stage, const Storage& storage, Window trainingWindow, WorkCoordinator& workCoordinator)
{
    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[stage.Stage] << "][" << trainingWindow.TrainingGameMin << " - "
        << trainingWindow.TrainingGameMax << "]" << std::endl;

    // Always save games to the GameType_Training store.
    const GameType gameType = GameType_Training;

    // Need to play enough games to reach the training window maximum (skip if already enough).
    const int gameTarget = trainingWindow.TrainingGameMax;
    const int gameCount = storage.GamesPlayed(gameType);
    const int gamesToPlay = std::max(0, gameTarget - gameCount);

    std::cout << "Playing " << gamesToPlay << " games..." << std::endl;
    if (gamesToPlay <= 0)
    {
        return;
    }

    // Play the games.
    workCoordinator.ResetWorkItemsRemaining(gamesToPlay);
    workCoordinator.WaitForWorkers();

    // Clear the prediction cache to prepare for the new network.
    PredictionCache::Instance.PrintDebugInfo();
    PredictionCache::Instance.Clear();
}

void ChessCoachTrain::StageTrain(const StageConfig& stage, Storage& storage, Window trainingWindow,
    SelfPlayWorker& selfPlayWorker, INetwork* network, int stepCount, int checkpoint)
{
    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[stage.Stage] << "][" << NetworkTypeNames[stage.Target] << "][" << GameTypeNames[stage.Type] << "]["
        << trainingWindow.TrainingGameMin << " - " << trainingWindow.TrainingGameMax << "]" << std::endl;

    // Configure the replay buffer window for position sampling (network training).
    storage.SetWindow(stage.Type, trainingWindow);

    // Train the network.
    std::cout << "Training..." << std::endl;
    selfPlayWorker.TrainNetwork(network, stage.Target, stage.Type, stepCount, checkpoint);
}

void ChessCoachTrain::StageTrainCommentary(const StageConfig& stage, SelfPlayWorker& selfPlayWorker, INetwork* network, int stepCount, int checkpoint)
{
    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[stage.Stage] << "][" << NetworkTypeNames[stage.Target] << "][" << GameTypeNames[stage.Type] << "]" << std::endl;

    // Only the teacher network supports commentary training.
    assert(stage.Target == NetworkType_Teacher);
    if (stage.Target != NetworkType_Teacher) throw std::runtime_error("Only the teacher network supports commentary training");

    // Only supervised data supported in commentary training.
    assert(stage.Type == GameType_Supervised);
    if (stage.Type != GameType_Supervised) throw std::runtime_error("Only supervised data supported in commentary training");

    // Train the main model and commentary decoder.
    std::cout << "Training commentary..." << std::endl;
    selfPlayWorker.TrainNetworkWithCommentary(network, stepCount, checkpoint);
}

void ChessCoachTrain::StageSave(const StageConfig& stage, SelfPlayWorker& selfPlayWorker, INetwork* network, int checkpoint)
{
    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[stage.Stage] << "][" << NetworkTypeNames[stage.Target] << "]" << std::endl;

    // Save the network. It logs enough internally.
    selfPlayWorker.SaveNetwork(network, stage.Target, checkpoint);
}

void ChessCoachTrain::StageStrengthTest(const StageConfig& stage, SelfPlayWorker& selfPlayWorker, INetwork* network, int checkpoint)
{
    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[stage.Stage] << "][" << NetworkTypeNames[stage.Target] << "]" << std::endl;

    // Potentially strength test the network, if the checkpoint is a multiple of the strength test interval.
    selfPlayWorker.StrengthTestNetwork(network, stage.Target, checkpoint);
}

Window ChessCoachTrain::CalculateWindow(const NetworkConfig& config, const StageConfig& stageConfig, int totalGamesCount, int networkCount, int network)
{
    float t = 0.f;
    int gameTarget = stageConfig.WindowSizeStart;

    if (networkCount > 1)
    {
        const int gamesPerNetworkAfterFirstWindow = ((totalGamesCount - stageConfig.WindowSizeStart) / (networkCount - 1));
        gameTarget = (stageConfig.WindowSizeStart + ((network - 1) * gamesPerNetworkAfterFirstWindow));
        t = (static_cast<float>(network - 1) / (networkCount - 1));
    }

    const int windowSize = static_cast<int>(stageConfig.WindowSizeStart +
        t * (stageConfig.WindowSizeFinish - stageConfig.WindowSizeStart));
    const int minimumSamplableGames = std::min(gameTarget, config.Training.BatchSize);

    // Min is inclusive, max is exclusive, both 0-based.
    return { std::max(0, gameTarget - windowSize), gameTarget, minimumSamplableGames };
}