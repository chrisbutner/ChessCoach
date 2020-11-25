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

    void StagePlay(const NetworkConfig& config, const StageConfig& stage, const Storage& storage, WorkCoordinator& workCoordinator, INetwork* network,
        int networkCount, int n, int checkpoint);
    void StagePlayWait(const NetworkConfig& config, const Storage& storage, INetwork* network, const Window& trainingWindow);
    void StageTrain(const NetworkConfig& config, const std::vector<StageConfig>& stages, int& stageIndex, SelfPlayWorker& selfPlayWorker, INetwork* network,
        int networkCount, int n, int step, int checkpoint);
    void StageTrainCommentary(const NetworkConfig& config, const StageConfig& stage, SelfPlayWorker& selfPlayWorker, INetwork* network, int step, int checkpoint);
    void StageSave(const NetworkConfig& config, const StageConfig& stage, SelfPlayWorker& selfPlayWorker, INetwork* network, int checkpoint);
    void StageSaveWait(const NetworkConfig& config, const StageConfig& stage, INetwork* network, int checkpoint);
    void StageStrengthTest(const NetworkConfig& config, const StageConfig& stage, SelfPlayWorker& selfPlayWorker, INetwork* network, int checkpoint);

    Window CalculateWindow(const StageConfig& stageConfig, int networkCount, int network);
};

int main()
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

    // We need to reach into Python for network info in case it's coming from cloud storage.
    int networkStepCount;
    int ignore;
    network->GetNetworkInfo(NetworkType_Teacher, networkStepCount, ignore);

    // Initialize storage for training and take care of any game/chunk housekeeping from previous runs.
    storage.InitializeLocalGamesChunks(network.get());

    // Start self-play worker threads. Also create a self-play worker for this main thread so that
    // MCTS nodes are always allocated and freed from the correct thread's pool allocator.
    std::unique_ptr<SelfPlayWorker> mainWorker = std::make_unique<SelfPlayWorker>(config, &storage);
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
    const int startingNetwork = (networkStepCount / config.Training.CheckpointInterval);

    // Run through all checkpoints with n in [1, networkCount].
    for (int n = startingNetwork + 1; n <= networkCount; n++)
    {
        const int checkpoint = (n * config.Training.CheckpointInterval);
        const int step = (checkpoint - config.Training.CheckpointInterval + 1);

        // Run through all stages in the checkpoint.
        for (int i = 0; i < config.Training.Stages.size(); i++)
        {
            // Run the stage.
            const StageConfig& stage = config.Training.Stages[i];
            switch (stage.Stage)
            {
            case StageType_Play:
            {
                StagePlay(config, stage, storage, workCoordinator, network.get(), networkCount, n, checkpoint);
                break;
            }
            case StageType_Train:
            {
                StageTrain(config, config.Training.Stages, i, *mainWorker, network.get(),
                    networkCount, n, step, checkpoint);
                break;
            }
            case StageType_TrainCommentary:
            {
                StageTrainCommentary(config, stage, *mainWorker, network.get(), step, checkpoint);
                break;
            }
            case StageType_Save:
            {
                StageSave(config, stage, *mainWorker, network.get(), checkpoint);
                break;
            }
            case StageType_StrengthTest:
            {
                StageStrengthTest(config, stage, *mainWorker, network.get(), checkpoint);
                break;
            }
            case StageType_Count:
            {
                // Catch unreferenced enums in GCC.
                break;
            }
            }
        }
    }
}

void ChessCoachTrain::StagePlay(const NetworkConfig& config, const StageConfig& stage, const Storage& storage,
    WorkCoordinator& workCoordinator, INetwork* network, int networkCount, int n, int checkpoint)
{
    // Calculate the replay buffer window for position sampling (network training)
    // and play enough games to satisfy it.
    const Window trainingWindow = CalculateWindow(stage, networkCount, n);

    // If this machine isn't a player, wait until we see someone else generate the required games.
    if (!(config.Role & RoleType_Play))
    {
        std::cout << "Waiting: [" << StageTypeNames[stage.Stage] << "][" << checkpoint << "]["
            << trainingWindow.TrainingGameMin << " - " << trainingWindow.TrainingGameMax << "]" << std::endl;
        StagePlayWait(config, storage, network, trainingWindow);
        return;
    }

    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[stage.Stage] << "][" << checkpoint << "]["
        << trainingWindow.TrainingGameMin << " - " << trainingWindow.TrainingGameMax << "]" << std::endl;

    // Need to play enough games to reach the training window maximum (skip if already enough).
    // Loop and check in case of distributed scenarios where other machines are generating games/chunks,
    // or to avoid generating too few games/chunks after unexpected failures or outside intervention.
    while (true)
    {
        // Check every "wait_milliseconds" to see whether other machines have generated enough games/chunks.
        const bool workersReady = workCoordinator.WaitForWorkers(config.Training.WaitMilliseconds);

        // We need to reach into Python for network info in case it's coming from cloud storage.
        int ignore;
        int trainingChunkCount;
        network->GetNetworkInfo(NetworkType_Teacher, ignore, trainingChunkCount);
        const int gamesToPlay = storage.TrainingGamesToPlay(trainingChunkCount, trainingWindow.TrainingGameMax, false /* ignoreLocalGames */);

        if (gamesToPlay <= 0)
        {
            // Stop workers in case we finished because of other machines, in a distributed scenario.
            std::cout << "Finished playing games" << std::endl;
            workCoordinator.ResetWorkItemsRemaining(0);
            break;
        }

        if (workersReady)
        {
            // Coordinate workers and print when initially starting, or restarting after a false finish.
            std::cout << "Playing " << gamesToPlay << " games..." << std::endl;

            // Generate uniform predictions for the first network (rather than use random weights).
            workCoordinator.GenerateUniformPredictions() = (n == 1);

            // Play the games.
            workCoordinator.ResetWorkItemsRemaining(gamesToPlay);
        }
        else
        {
            // Loop back and keep waiting for workers.
        }
    }

    // Print prediction cache stats after finishing self-play.
    PredictionCache::Instance.PrintDebugInfo();
}

void ChessCoachTrain::StagePlayWait(const NetworkConfig& config, const Storage& storage, INetwork* network, const Window& trainingWindow)
{
    while (true)
    {
        int ignore;
        int trainingChunkCount;
        network->GetNetworkInfo(NetworkType_Teacher, ignore, trainingChunkCount);
        const int distributedGamesToPlay = storage.TrainingGamesToPlay(
            trainingChunkCount, trainingWindow.TrainingGameMax, true /* ignoreLocalGames */);
        if (distributedGamesToPlay <= 0)
        {
            return;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(config.Training.WaitMilliseconds));
    }
}

void ChessCoachTrain::StageTrain(const NetworkConfig& config, const std::vector<StageConfig>& stages, int& stageIndex,
    SelfPlayWorker& selfPlayWorker, INetwork* network, int networkCount, int n, int step, int checkpoint)
{
    const int first = stageIndex;
    std::vector<GameType> gameTypes;
    std::vector<Window> trainingWindows;

    // If this machine isn't a trainer, skip training (no need to coalesce).
    if (!(config.Role & RoleType_Train))
    {
        return;
    }

    // Coalesce multiple adjacent "training" stages for the same target (e.g. "teacher" or "student")
    // but potentially different types (e.g. "supervised" or "training") into a single rotation.
    while ((stageIndex < stages.size()) &&
        (stages[stageIndex].Stage == stages[first].Stage) &&
        (stages[stageIndex].Target == stages[first].Target))
    {
        const StageConfig& stage = stages[stageIndex];
        gameTypes.push_back(stage.Type);

        // Calculate the replay buffer window for position sampling (network training).
        const Window& trainingWindow = trainingWindows.emplace_back(CalculateWindow(stage, networkCount, n));

        // Log the stage info in a consistent format (just use one line per game type).
        std::cout << "Stage: [" << StageTypeNames[stage.Stage] << "][" << NetworkTypeNames[stage.Target] << "][" << GameTypeNames[stage.Type] << "]["
            << checkpoint << "]["
            << trainingWindow.TrainingGameMin << " - " << trainingWindow.TrainingGameMax << "]" << std::endl;

        // Skip past this stage in the outer training loop (mutable reference).
        stageIndex++;
    }

    // The outer training loop will also increment the stageIndex, so correct for it.
    stageIndex--;

    // Train the network.
    selfPlayWorker.TrainNetwork(network, stages[first].Target, gameTypes, trainingWindows, step, checkpoint);
}

void ChessCoachTrain::StageTrainCommentary(const NetworkConfig& config, const StageConfig& stage, SelfPlayWorker& selfPlayWorker, INetwork* network, int step, int checkpoint)
{
    // If this machine isn't a trainer, skip training commentary.
    if (!(config.Role & RoleType_Train))
    {
        return;
    }

    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[stage.Stage] << "][" << NetworkTypeNames[stage.Target] << "][" << GameTypeNames[stage.Type] << "][" 
        << checkpoint << "]" << std::endl;

    // Only the teacher network supports commentary training.
    assert(stage.Target == NetworkType_Teacher);
    if (stage.Target != NetworkType_Teacher) throw std::runtime_error("Only the teacher network supports commentary training");

    // Only supervised data supported in commentary training.
    assert(stage.Type == GameType_Supervised);
    if (stage.Type != GameType_Supervised) throw std::runtime_error("Only supervised data supported in commentary training");

    // Train the main model and commentary decoder.
    selfPlayWorker.TrainNetworkWithCommentary(network, step, checkpoint);
}

void ChessCoachTrain::StageSave(const NetworkConfig& config, const StageConfig& stage, SelfPlayWorker& selfPlayWorker, INetwork* network, int checkpoint)
{
    // If this machine isn't a trainer, and "wait_for_updated_network" is *true*, wait until we see someone else save this network type + checkpoint.
    // If this machine isn't a trainer, and "wait_for_updated_network" is *false*, return immediately and start playing the next set of games.
    // In either case, the network will be updated and used when available, even mid-game (detected and updated in network.py, observed and handled in SelfPlay.cpp).
    if (!(config.Role & RoleType_Train))
    {
        if (config.SelfPlay.WaitForUpdatedNetwork)
        {
            std::cout << "Waiting: [" << StageTypeNames[stage.Stage] << "][" << NetworkTypeNames[stage.Target] << "][" << checkpoint << "]" << std::endl;
            StageSaveWait(config, stage, network, checkpoint);
            return;
        }
        else
        {
            std::cout << "Not waiting: [" << StageTypeNames[stage.Stage] << "][" << NetworkTypeNames[stage.Target] << "][" << checkpoint << "]" << std::endl;
            return;
        }
    }

    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[stage.Stage] << "][" << NetworkTypeNames[stage.Target] << "][" << checkpoint << "]" << std::endl;

    // Save the network. It logs enough internally.
    selfPlayWorker.SaveNetwork(network, stage.Target, checkpoint);
}

void ChessCoachTrain::StageSaveWait(const NetworkConfig& config, const StageConfig& stage, INetwork* network, int checkpoint)
{
    while (true)
    {
        int networkStepCount;
        int ignore;
        network->GetNetworkInfo(stage.Target, networkStepCount, ignore);

        if (networkStepCount >= checkpoint)
        {
            return;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(config.Training.WaitMilliseconds));
    }
}

void ChessCoachTrain::StageStrengthTest(const NetworkConfig& config, const StageConfig& stage, SelfPlayWorker& selfPlayWorker, INetwork* network, int checkpoint)
{
    // If this machine isn't a trainer, skip Strength testing.
    if (!(config.Role & RoleType_Train))
    {
        return;
    }

    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[stage.Stage] << "][" << NetworkTypeNames[stage.Target] << "][" << checkpoint << "]" << std::endl;

    // Potentially strength test the network, if the checkpoint is a multiple of the strength test interval.
    selfPlayWorker.StrengthTestNetwork(network, stage.Target, checkpoint);
}

Window ChessCoachTrain::CalculateWindow(const StageConfig& stageConfig, int networkCount, int network)
{
    // The starting window (subscript 0) mins at zero.
    // The finishing window (subscript 1) maxes at totalGamesCount.
    // Lerping the window min and max also lerps the window size correctly.
    const int windowSizeStart = std::min(stageConfig.NumGames, stageConfig.WindowSizeStart);
    const int windowSizeFinish = std::min(stageConfig.NumGames, stageConfig.WindowSizeFinish);
    const int windowMin0 = 0;
    const int windowMin1 = (stageConfig.NumGames - windowSizeFinish);
    const int windowMax0 = windowSizeStart;
    const int windowMax1 = stageConfig.NumGames;

    const float t = (networkCount > 1) ? 
        (static_cast<float>(network - 1) / (networkCount - 1)) :
        0.f;

    const int windowMin = windowMin0 + static_cast<int>(t * (windowMin1 - windowMin0));
    const int windowMax = windowMax0 + static_cast<int>(t * (windowMax1 - windowMax0));

    // Min is inclusive, max is exclusive, both 0-based.
    return { windowMin, windowMax };
}