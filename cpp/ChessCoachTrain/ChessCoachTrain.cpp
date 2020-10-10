#include <iostream>
#include <thread>
#include <algorithm>
#include <functional>
#include <numeric>

#include <ChessCoach/ChessCoach.h>
#include <ChessCoach/Threading.h>
#include <ChessCoach/SelfPlay.h>

class ChessCoachTrain : public ChessCoach
{
public:

    void TrainChessCoach();

private:

    void StagePlay(const StageConfig& stage, const Storage& storage, Window trainingWindow, WorkCoordinator& workCoordinator);
    void StageTrain(const std::vector<StageConfig>& stages, int& stageIndex, const NetworkConfig& config, Storage& storage,
        SelfPlayWorker& selfPlayWorker, INetwork* network, int networkCount, int n, int stepCount, int checkpoint);
    void StageTrainCommentary(const StageConfig& stage, SelfPlayWorker& selfPlayWorker, INetwork* network, int stepCount, int checkpoint);
    void StageSave(const StageConfig& stage, SelfPlayWorker& selfPlayWorker, INetwork* network, int checkpoint);
    void StageStrengthTest(const StageConfig& stage, SelfPlayWorker& selfPlayWorker, INetwork* network, int checkpoint);

    Window CalculateWindow(const NetworkConfig& config, const StageConfig& stageConfig, int networkCount, int network);
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
    const int networkCount = (config.Training.Steps / config.Training.CheckpointInterval);
    const int startingNetwork = (storage.NetworkStepCount(config.Name) / config.Training.CheckpointInterval);

    // Load games up to the highest number required for training. It's always safe to try load more games, but this can
    // make e.g. tiny runs load way too many games, too slowly.
    //
    // It could be dangerous to download partial games then self-play, since new games overwrite the ones not yet loaded,
    // but self-play shouldn't need to generate any games beyond the total required for training anyway.
    int maxNumGames = 0;
    for (const StageConfig& stage : config.Training.Stages)
    {
        if (stage.NumGames > maxNumGames)
        {
            maxNumGames = stage.NumGames;
        }
    }
    storage.LoadExistingGames(GameType_Supervised, maxNumGames);
    storage.LoadExistingGames(GameType_Training, maxNumGames);
    storage.LoadExistingGames(GameType_Validation, maxNumGames);

    // Always use full window for validation.
    const int minimumSamplableGames = std::min(maxNumGames, config.Training.BatchSize);
    const Window fullWindow = { 0, std::numeric_limits<int>::max(), minimumSamplableGames };
    storage.SetWindow(GameType_Validation, fullWindow);

    // Run through all checkpoints with n in [1, networkCount].
    for (int n = startingNetwork + 1; n <= networkCount; n++)
    {
        const int checkpoint = (n * config.Training.CheckpointInterval);

        // Run through all stages in the checkpoint.
        for (int i = 0; i < config.Training.Stages.size(); i++)
        {
            // Run the stage.
            const StageConfig& stage = config.Training.Stages[i];
            switch (stage.Stage)
            {
            case StageType_Play:
            {
                // Calculate the replay buffer window for position sampling (network training)
                // and play enough games to satisfy it.
                const Window trainingWindow = CalculateWindow(config, stage, networkCount, n);
                StagePlay(stage, storage, trainingWindow, workCoordinator);
                break;
            }
            case StageType_Train:
            {
                StageTrain(config.Training.Stages, i, config, storage, *selfPlayWorkers.front(), network.get(),
                    networkCount, n, config.Training.CheckpointInterval, checkpoint);
                break;
            }
            case StageType_TrainCommentary:
            {
                StageTrainCommentary(stage, *selfPlayWorkers.front(), network.get(),
                    config.Training.CheckpointInterval, checkpoint);
                break;
            }
            case StageType_Save:
            {
                StageSave(stage, *selfPlayWorkers.front(), network.get(), checkpoint);
                break;
            }
            case StageType_StrengthTest:
            {
                StageStrengthTest(stage, *selfPlayWorkers.front(), network.get(), checkpoint);
                break;
            }
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

void ChessCoachTrain::StageTrain(const std::vector<StageConfig>& stages, int& stageIndex, const NetworkConfig& config, Storage& storage,
    SelfPlayWorker& selfPlayWorker, INetwork* network, int networkCount, int n, int stepCount, int checkpoint)
{
    const int first = stageIndex;
    std::vector<GameType> gameTypes;

    // Coalesce multiple adjacent "training" stages for the same target (e.g. "teacher" or "student")
    // but potentially different types (e.g. "supervised" or "training") into a single rotation.
    while ((stageIndex < stages.size()) &&
        (stages[stageIndex].Stage == stages[first].Stage) &&
        (stages[stageIndex].Target == stages[first].Target))
    {
        const StageConfig& stage = stages[stageIndex];
        gameTypes.push_back(stage.Type);

        // Calculate the replay buffer window for position sampling (network training).
        // NOTE: Window settings for the same game type in a coalesced rotation must be the same (last will clobber).
        const Window trainingWindow = CalculateWindow(config, stage, networkCount, n);

        // Log the stage info in a consistent format (just use one line per game type).
        std::cout << "Stage: [" << StageTypeNames[stage.Stage] << "][" << NetworkTypeNames[stage.Target] << "][" << GameTypeNames[stage.Type] << "]["
            << trainingWindow.TrainingGameMin << " - " << trainingWindow.TrainingGameMax << "]" << std::endl;

        storage.SetWindow(stage.Type, trainingWindow);

        // Skip past this stage in the outer training loop (mutable reference).
        stageIndex++;
    }

    // The outer training loop will also increment the stageIndex, so correct for it.
    stageIndex--;

    // Train the network.
    std::cout << "Training..." << std::endl;
    selfPlayWorker.TrainNetwork(network, stages[first].Target, gameTypes, stepCount, checkpoint);
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

Window ChessCoachTrain::CalculateWindow(const NetworkConfig& config, const StageConfig& stageConfig, int networkCount, int network)
{
    // The starting window (subscript 0) mins at zero.
    // The finishing window (subscript 1) maxes at totalGamesCount.
    // Lerping the window min and max also lerps the window size correctly.
    const int windowMin0 = 0;
    const int windowMin1 = (stageConfig.NumGames - stageConfig.WindowSizeFinish);
    const int windowMax0 = stageConfig.WindowSizeStart;
    const int windowMax1 = stageConfig.NumGames;

    const float t = (networkCount > 1) ? 
        (static_cast<float>(network - 1) / (networkCount - 1)) :
        0.f;

    const int windowMin = windowMin0 + static_cast<int>(t * (windowMin1 - windowMin0));
    const int windowMax = windowMax0 + static_cast<int>(t * (windowMax1 - windowMax0));
    const int minimumSamplableGames = std::min(windowMax, config.Training.BatchSize);

    // Min is inclusive, max is exclusive, both 0-based.
    return { windowMin, windowMax, minimumSamplableGames };
}