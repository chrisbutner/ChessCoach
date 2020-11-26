#include <iostream>
#include <thread>
#include <algorithm>
#include <functional>
#include <numeric>

#include <ChessCoach/ChessCoach.h>
#include <ChessCoach/Threading.h>
#include <ChessCoach/SelfPlay.h>

struct TrainingState
{
    const NetworkConfig* config;
    const MiscConfig* miscConfig;
    Storage* storage;
    INetwork* network;
    int networkCount;
    int networkNumber;
    int stageIndex;

    const StageConfig& Stage() const;
    int Checkpoint() const;
    int Step() const;
    bool HasRole(RoleType role) const;
    std::string StrengthTestMarkerRelativePath() const;
};

class ChessCoachTrain : public ChessCoach
{
public:

    void TrainChessCoach();

private:

    void StagePlay(const TrainingState& state, WorkCoordinator& workCoordinator);
    void StageTrain(TrainingState& stateInOut, SelfPlayWorker& selfPlayWorker);
    void StageTrainCommentary(const TrainingState& state, SelfPlayWorker& selfPlayWorker);
    void StageSave(const TrainingState& state, SelfPlayWorker& selfPlayWorker);
    void StageStrengthTest(const TrainingState& state, SelfPlayWorker& selfPlayWorker);
    
    void ResumeTraining(TrainingState& stateInOut);
    bool IsStageComplete(const TrainingState& state);

    Window CalculateWindow(const TrainingState& state);
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
    const MiscConfig& miscConfig = Config::Misc;
    std::unique_ptr<INetwork> network(CreateNetwork(config));
    Storage storage(config, miscConfig);

    // We need to reach into Python for network info in case it's coming from cloud storage.
    int networkStepCount;
    network->GetNetworkInfo(NetworkType_Teacher, &networkStepCount, nullptr, nullptr);

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
    // See if there's still work to do in the current/latest network, based on the presence of artifacts in storage.
    //
    // Note that the starting "networkNumber" is based on the latest saved teacher network, so it assumes that will be the first
    // saved artifact in a set of stages. If not, earlier artifacts will need to be redone. Shouldn't be a problem though: you need
    // a teacher to train a student, and you need a teacher or student to strength test.
    TrainingState state;
    state.config = &config;
    state.miscConfig = &miscConfig;
    state.storage = &storage;
    state.network = network.get();
    state.networkCount = (config.Training.Steps / config.Training.CheckpointInterval);
    state.networkNumber = (networkStepCount / config.Training.CheckpointInterval);
    state.stageIndex = 0;
    ResumeTraining(state);

    // Run through all checkpoints with n in [1, networkCount].
    for (; state.networkNumber <= state.networkCount; state.networkNumber++, state.stageIndex = 0)
    {
        // Run through all stages in the checkpoint.
        for (; state.stageIndex < config.Training.Stages.size(); state.stageIndex++)
        {
            // Run the stage.
            switch (state.Stage().Stage)
            {
            case StageType_Play:
            {
                StagePlay(state, workCoordinator);
                break;
            }
            case StageType_Train:
            {
                StageTrain(state, *mainWorker);
                break;
            }
            case StageType_TrainCommentary:
            {
                StageTrainCommentary(state, *mainWorker);
                break;
            }
            case StageType_Save:
            {
                StageSave(state, *mainWorker);
                break;
            }
            case StageType_StrengthTest:
            {
                StageStrengthTest(state, *mainWorker);
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

void ChessCoachTrain::StagePlay(const TrainingState& state, WorkCoordinator& workCoordinator)
{
    // Calculate the replay buffer window for position sampling (network training)
    // and play enough games to satisfy it.
    const Window trainingWindow = CalculateWindow(state);

    // If this machine isn't a player, wait until we see someone else generate the required games.
    if (!state.HasRole(RoleType_Play))
    {
        std::cout << "Waiting: [" << StageTypeNames[state.Stage().Stage] << "][" << state.Checkpoint() << "]["
            << trainingWindow.TrainingGameMin << " - " << trainingWindow.TrainingGameMax << "]" << std::endl;
        while (!IsStageComplete(state))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(state.config->Training.WaitMilliseconds));
        }
        return;
    }

    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[state.Stage().Stage] << "][" << state.Checkpoint() << "]["
        << trainingWindow.TrainingGameMin << " - " << trainingWindow.TrainingGameMax << "]" << std::endl;

    // Need to play enough games to reach the training window maximum (skip if already enough).
    // Loop and check in case of distributed scenarios where other machines are generating games/chunks,
    // or to avoid generating too few games/chunks after unexpected failures or outside intervention.
    while (true)
    {
        // Check every "wait_milliseconds" to see whether other machines have generated enough games/chunks.
        const bool workersReady = workCoordinator.WaitForWorkers(state.config->Training.WaitMilliseconds);

        // We need to reach into Python for network info in case it's coming from cloud storage.
        int trainingChunkCount;
        state.network->GetNetworkInfo(NetworkType_Teacher, nullptr, &trainingChunkCount, nullptr);
        const int gamesToPlay = state.storage->TrainingGamesToPlay(trainingChunkCount, trainingWindow.TrainingGameMax, false /* ignoreLocalGames */);

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
            workCoordinator.GenerateUniformPredictions() = (state.networkNumber == 1);

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

void ChessCoachTrain::StageTrain(TrainingState& stateInOut, SelfPlayWorker& selfPlayWorker)
{
    const int first = stateInOut.stageIndex;
    std::vector<GameType> gameTypes;
    std::vector<Window> trainingWindows;

    // If this machine isn't a trainer, skip training (no need to coalesce).
    if (!stateInOut.HasRole(RoleType_Train))
    {
        return;
    }

    // Coalesce multiple adjacent "training" stages for the same target (e.g. "teacher" or "student")
    // but potentially different types (e.g. "supervised" or "training") into a single rotation.
    const std::vector<StageConfig>& stages = stateInOut.config->Training.Stages;
    while ((stateInOut.stageIndex < stages.size()) &&
        (stateInOut.Stage().Stage == stages[first].Stage) &&
        (stateInOut.Stage().Target == stages[first].Target))
    {
        const StageConfig& stage = stateInOut.Stage();
        gameTypes.push_back(stage.Type);

        // Calculate the replay buffer window for position sampling (network training).
        const Window& trainingWindow = trainingWindows.emplace_back(CalculateWindow(stateInOut));

        // Log the stage info in a consistent format (just use one line per game type).
        std::cout << "Stage: [" << StageTypeNames[stage.Stage] << "][" << NetworkTypeNames[stage.Target] << "][" << GameTypeNames[stage.Type] << "]["
            << stateInOut.Checkpoint() << "]["
            << trainingWindow.TrainingGameMin << " - " << trainingWindow.TrainingGameMax << "]" << std::endl;

        // Skip past this stage in the outer training loop (mutable reference).
        stateInOut.stageIndex++;
    }

    // The outer training loop will also increment the stageIndex, so correct for it.
    stateInOut.stageIndex--;

    // Train the network.
    selfPlayWorker.TrainNetwork(stateInOut.network, stages[first].Target, gameTypes, trainingWindows, stateInOut.Step(), stateInOut.Checkpoint());
}

void ChessCoachTrain::StageTrainCommentary(const TrainingState& state, SelfPlayWorker& selfPlayWorker)
{
    // If this machine isn't a trainer, skip training commentary.
    if (!state.HasRole(RoleType_Train))
    {
        return;
    }

    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[state.Stage().Stage] << "][" << NetworkTypeNames[state.Stage().Target] << "][" << GameTypeNames[state.Stage().Type] << "]["
        << state.Checkpoint() << "]" << std::endl;

    // Only the teacher network supports commentary training.
    assert(state.Stage().Target == NetworkType_Teacher);
    if (state.Stage().Target != NetworkType_Teacher) throw std::runtime_error("Only the teacher network supports commentary training");

    // Only supervised data supported in commentary training.
    assert(state.Stage().Type == GameType_Supervised);
    if (state.Stage().Type != GameType_Supervised) throw std::runtime_error("Only supervised data supported in commentary training");

    // Train the main model and commentary decoder.
    selfPlayWorker.TrainNetworkWithCommentary(state.network, state.Step(), state.Checkpoint());
}

void ChessCoachTrain::StageSave(const TrainingState& state, SelfPlayWorker& selfPlayWorker)
{
    // If this machine isn't a trainer, and "wait_for_updated_network" is *true*, wait until we see someone else save this network type + checkpoint.
    // If this machine isn't a trainer, and "wait_for_updated_network" is *false*, return immediately and start playing the next set of games,
    // *except* when this is the first network and we're generating uniform predictions.
    // In either case, the network will be updated and used when available, even mid-game (detected and updated in network.py, observed and handled in SelfPlay.cpp).
    const int checkpoint = state.Checkpoint();
    if (!state.HasRole(RoleType_Train))
    {
        if (state.config->SelfPlay.WaitForUpdatedNetwork || (state.networkNumber == 1))
        {
            std::cout << "Waiting: [" << StageTypeNames[state.Stage().Stage] << "][" << NetworkTypeNames[state.Stage().Target] << "][" << checkpoint << "]" << std::endl;
            while (!IsStageComplete(state))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(state.config->Training.WaitMilliseconds));
            }
            return;
        }
        else
        {
            std::cout << "Not waiting: [" << StageTypeNames[state.Stage().Stage] << "][" << NetworkTypeNames[state.Stage().Target] << "][" << checkpoint << "]" << std::endl;
            return;
        }
    }

    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[state.Stage().Stage] << "][" << NetworkTypeNames[state.Stage().Target] << "][" << checkpoint << "]" << std::endl;

    // Save the network. It logs enough internally.
    selfPlayWorker.SaveNetwork(state.network, state.Stage().Target, checkpoint);
}

void ChessCoachTrain::StageStrengthTest(const TrainingState& state, SelfPlayWorker& selfPlayWorker)
{
    // If this machine isn't a trainer, skip Strength testing.
    if (!state.HasRole(RoleType_Train))
    {
        return;
    }

    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[state.Stage().Stage] << "][" << NetworkTypeNames[state.Stage().Target] << "][" << state.Checkpoint() << "]" << std::endl;

    // Potentially strength test the network, if the checkpoint is a multiple of the strength test interval.
    const bool strengthTested = selfPlayWorker.StrengthTestNetwork(state.network, state.Stage().Target, state.Checkpoint());
    if (strengthTested)
    {
        const std::string markerRelativePath = state.StrengthTestMarkerRelativePath();
        if (!markerRelativePath.empty())
        {
            state.network->SaveFile(markerRelativePath, "");
        }
    }
}

void ChessCoachTrain::ResumeTraining(TrainingState& stateInOut)
{
    // Logic: Start immediately after the last complete stage seen via artifacts in storage.
    //
    // This handles the most common scenarios, like (1) present, (2)/(3) missing, or (1)/(2) present, (3) missing,
    // but doesn't handle scenarios like (1)/(3) present, (2) missing, or parts inside of (1), (2) or (3) missing,
    // which are treated as unlikely corruption. Python code may still accidentally work around cases like
    // (1)/(3) present, (2) missing by e.g. loading a student network from a previous checkpoint.

    // Iterate backwards and break on the first complete stage seen.
    for (stateInOut.stageIndex = static_cast<int>(stateInOut.config->Training.Stages.size()) - 1;stateInOut.stageIndex >= 0; stateInOut.stageIndex--)
    {
        if (IsStageComplete(stateInOut))
        {
            break;
        }
    }

    // If no stages were complete then we increment "stageIndexOut" from -1 to 0 and run them all.
    // If all stages were complete then we roll over and move on to the next network.
    if (++stateInOut.stageIndex >= stateInOut.config->Training.Stages.size())
    {
        stateInOut.networkNumber++;
        stateInOut.stageIndex = 0;
    }
}

bool ChessCoachTrain::IsStageComplete(const TrainingState& state)
{
    // There is no zeroth network: start on the first.
    if (state.networkNumber < 1)
    {
        return true;
    }

    // Note that any "GetNetworkInfo" checks will use the latest teacher or student network found by Python,
    // and will not take into account any modified "TrainingState.networkNumber" or ".Checkpoint()".
    switch (state.Stage().Stage)
    {
    case StageType_Play:
    {
        // Look for sufficient game chunks in storage.
        // Technically we don't need this for "ResumeTraining", only waiting in "StagePlay",
        // but it's safe, unlikely to be hit unnecessarily (empty dir?), and useful to co-locate.
        int trainingChunkCount;
        state.network->GetNetworkInfo(NetworkType_Teacher, nullptr, &trainingChunkCount, nullptr);

        const Window trainingWindow = CalculateWindow(state);
        const int distributedGamesToPlay = state.storage->TrainingGamesToPlay(
            trainingChunkCount, trainingWindow.TrainingGameMax, true /* ignoreLocalGames */);
        return (distributedGamesToPlay <= 0);
    }
    case StageType_Train:
    {
        // Indeterminate; we only care about saved artifacts.
        return false;
    }
    case StageType_TrainCommentary:
    {
        // Indeterminate; we only care about saved artifacts.
        return false;
    }
    case StageType_Save:
    {
        // Look for the teacher or student network in storage.
        int networkStepCount;
        state.network->GetNetworkInfo(state.Stage().Target, &networkStepCount, nullptr, nullptr);
        return (networkStepCount >= state.Checkpoint());
    }
    case StageType_StrengthTest:
    {
        // Look for the teacher or student strength test marker file in storage.
        const std::string markerRelativePath = state.StrengthTestMarkerRelativePath();
        return markerRelativePath.empty() ? false : state.network->FileExists(markerRelativePath);
    }
    case StageType_Count:
    {
        // Catch unreferenced enums in GCC.
        throw std::runtime_error("Unexpected: StageType_Count");
    }
    }

    throw std::runtime_error("Unexpected StageType");
}

Window ChessCoachTrain::CalculateWindow(const TrainingState& state)
{
    // The starting window (subscript 0) mins at zero.
    // The finishing window (subscript 1) maxes at totalGamesCount.
    // Lerping the window min and max also lerps the window size correctly.
    const int numGames = state.Stage().NumGames;
    const int windowSizeStart = std::min(numGames, state.Stage().WindowSizeStart);
    const int windowSizeFinish = std::min(numGames, state.Stage().WindowSizeFinish);
    const int windowMin0 = 0;
    const int windowMin1 = (state.Stage().NumGames - windowSizeFinish);
    const int windowMax0 = windowSizeStart;
    const int windowMax1 = numGames;

    const float t = (state.networkCount > 1) ? 
        (static_cast<float>(state.networkNumber - 1) / (state.networkCount - 1)) :
        0.f;

    const int windowMin = windowMin0 + static_cast<int>(t * (windowMin1 - windowMin0));
    const int windowMax = windowMax0 + static_cast<int>(t * (windowMax1 - windowMax0));

    // Min is inclusive, max is exclusive, both 0-based.
    return { windowMin, windowMax };
}

const StageConfig& TrainingState::Stage() const
{
    return config->Training.Stages[stageIndex];
}

int TrainingState::Checkpoint() const
{
    return (networkNumber * config->Training.CheckpointInterval);
}

int TrainingState::Step() const
{
    return (((networkNumber - 1) * config->Training.CheckpointInterval) + 1);
}

bool TrainingState::HasRole(RoleType role) const
{
    return (config->Role & role);
}

std::string TrainingState::StrengthTestMarkerRelativePath() const
{
    std::string relativePath;
    network->GetNetworkInfo(Stage().Target, nullptr, nullptr, &relativePath);
    if (relativePath.empty())
    {
        // No path, not saved in storage yet.
        return relativePath;
    }

    // Always use a forward slash for potential compatibility with gs://
    relativePath += '/';
    relativePath += miscConfig->Paths_StrengthTestMarkerPrefix;
    relativePath += NetworkTypeNames[Stage().Target];

    return relativePath;
}