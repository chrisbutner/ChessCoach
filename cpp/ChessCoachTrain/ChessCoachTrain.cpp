#include <iostream>
#include <thread>
#include <algorithm>
#include <functional>
#include <numeric>

#include <ChessCoach/ChessCoach.h>
#include <ChessCoach/Threading.h>
#include <ChessCoach/WorkerGroup.h>
#include <ChessCoach/Syzygy.h>

struct TrainingState
{
    Storage* storage;
    INetwork* network;
    WorkerGroup* workerGroup;
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

    void StagePlay(const TrainingState& state);
    void StageTrain(const TrainingState& state);
    void StageTrainCommentary(const TrainingState& state);
    void StageSave(const TrainingState& state);
    void StageSaveSwa(const TrainingState& state);
    void StageStrengthTest(const TrainingState& state);
    
    void ValidateSchedule(const TrainingState& state);
    void ResumeTraining(TrainingState& stateInOut);
    bool IsStageComplete(const TrainingState& state);
    void WaitForUpdatedNetwork(const TrainingState& state);

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
    std::unique_ptr<INetwork> network(CreateNetwork());
    Storage storage;

    // We need to reach into Python for network info in case it's coming from cloud storage.
    int networkStepCountTeacher;
    int networkStepCountStudent;
    network->GetNetworkInfo(NetworkType_Teacher, &networkStepCountTeacher, nullptr, nullptr, nullptr);
    network->GetNetworkInfo(NetworkType_Student, &networkStepCountStudent, nullptr, nullptr, nullptr);
    const int networkStepCount = std::max(networkStepCountTeacher, networkStepCountStudent);

    // Initialize storage for training and take care of any game/chunk housekeeping from previous runs.
    storage.InitializeLocalGamesChunks(network.get());

    // Initialize Syzygy endgame tablebases if needed.
    if (Config::Network.SelfPlay.SyzygyProbeProportion > 0.f)
    {
        Syzygy::Reload();
    }

    // Start self-play worker threads.
    WorkerGroup workerGroup;
    workerGroup.Initialize(network.get(), &storage, Config::Network.SelfPlay.PredictionNetworkType, Config::Network.SelfPlay.NumWorkers,
        Config::Network.SelfPlay.PredictionBatchSize, &SelfPlayWorker::LoopSelfPlay);
    for (int i = 0; i < Config::Network.SelfPlay.NumWorkers; i++)
    {
        std::cout << "Starting self-play thread " << (i + 1) << " of " << Config::Network.SelfPlay.NumWorkers <<
            " (" << Config::Network.SelfPlay.PredictionBatchSize << " games per thread)" << std::endl;
    }

    // Wait until all self-play workers are initialized.
    workerGroup.workCoordinator->WaitForWorkers();

    // Plan full training and resume progress. If the network's step count isn't a multiple of the checkpoint interval, round down.
    // See if there's still work to do in the current/latest network, based on the presence of artifacts in storage.
    //
    // Note that the starting "networkNumber" is based on the latest saved teacher or student network, so it assumes that will be the first
    // saved artifact in a set of stages. If not, earlier artifacts will need to be redone. Shouldn't be a problem though: you need
    // a teacher or student network to strength test.
    TrainingState state;
    state.storage = &storage;
    state.network = network.get();
    state.workerGroup = &workerGroup;
    state.networkCount = (Config::Network.Training.Steps / Config::Network.Training.CheckpointInterval);
    state.networkNumber = (networkStepCount / Config::Network.Training.CheckpointInterval);
    state.stageIndex = 0;
    ValidateSchedule(state);
    ResumeTraining(state);

    // Run through all checkpoints with n in [1, networkCount].
    for (; state.networkNumber <= state.networkCount; state.networkNumber++, state.stageIndex = 0)
    {
        // Run through all stages in the checkpoint.
        for (; state.stageIndex < Config::Network.Training.Stages.size(); state.stageIndex++)
        {
            // Run the stage.
            switch (state.Stage().Stage)
            {
            case StageType_Play:
            {
                StagePlay(state);
                break;
            }
            case StageType_Train:
            {
                StageTrain(state);
                break;
            }
            case StageType_TrainCommentary:
            {
                StageTrainCommentary(state);
                break;
            }
            case StageType_Save:
            {
                StageSave(state);
                break;
            }
            case StageType_SaveSwa:
            {
                StageSaveSwa(state);
                break;
            }
            case StageType_StrengthTest:
            {
                StageStrengthTest(state);
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

    workerGroup.ShutDown();
}

void ChessCoachTrain::StagePlay(const TrainingState& state)
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
            std::this_thread::sleep_for(std::chrono::milliseconds(Config::Network.Training.WaitMilliseconds));
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
        const bool workersReady = state.workerGroup->workCoordinator->WaitForWorkers(Config::Network.Training.WaitMilliseconds);

        // We need to reach into Python for network info in case it's coming from cloud storage.
        int trainingChunkCount;
        state.network->GetNetworkInfo(NetworkType_Teacher, nullptr, nullptr, &trainingChunkCount, nullptr);
        const int gamesToPlay = state.storage->TrainingGamesToPlay(trainingChunkCount, trainingWindow.TrainingGameMax, false /* ignoreLocalGames */);

        if (gamesToPlay <= 0)
        {
            // Stop workers in case we finished because of other machines, in a distributed scenario.
            std::cout << "Finished playing games" << std::endl;
            state.workerGroup->workCoordinator->ResetWorkItemsRemaining(0);
            break;
        }

        if (workersReady)
        {
            // Coordinate workers and print when initially starting, or restarting after a false finish.
            std::cout << "Playing " << gamesToPlay << " games..." << std::endl;

            // Generate uniform predictions for the first network (rather than use random weights).
            state.workerGroup->workCoordinator->GenerateUniformPredictions() = (state.networkNumber == 1);

            // Play the games.
            state.workerGroup->workCoordinator->ResetWorkItemsRemaining(gamesToPlay);
        }
        else
        {
            // Loop back and keep waiting for workers.
        }
    }

    // Print prediction cache stats after finishing self-play.
    PredictionCache::Instance.PrintDebugInfo();
}

void ChessCoachTrain::StageTrain(const TrainingState& state)
{
    // If this machine isn't a trainer, skip training.
    if (!state.HasRole(RoleType_Train))
    {
        return;
    }

    // Calculate the replay buffer window for position sampling (network training).
    const Window& trainingWindow = CalculateWindow(state);

    // Log the stage info in a consistent format (just use one line per game type).
        std::cout << "Stage: [" << StageTypeNames[state.Stage().Stage] << "][" << NetworkTypeNames[state.Stage().Target] << "]["
            << state.Checkpoint() << "][" << trainingWindow.TrainingGameMin << " - " << trainingWindow.TrainingGameMax << "]" << std::endl;

    // Train the network.
    state.workerGroup->controllerWorker->TrainNetwork(state.network, state.Stage().Target, state.Step(), state.Checkpoint());
}

void ChessCoachTrain::StageTrainCommentary(const TrainingState& state)
{
    // If this machine isn't a trainer, skip training commentary.
    if (!state.HasRole(RoleType_Train))
    {
        return;
    }

    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[state.Stage().Stage] << "][" << NetworkTypeNames[state.Stage().Target] << "]["
        << state.Checkpoint() << "]" << std::endl;

    // Only the teacher network supports commentary training.
    assert(state.Stage().Target == NetworkType_Teacher);
    if (state.Stage().Target != NetworkType_Teacher) throw std::runtime_error("Only the teacher network supports commentary training");

    // Train the main model and commentary decoder.
    state.workerGroup->controllerWorker->TrainNetworkWithCommentary(state.network, state.Step(), state.Checkpoint());
}

void ChessCoachTrain::StageSave(const TrainingState& state)
{
    // If this machine isn't a trainer then we may need to wait for an updated network, or we may be able to keep playing (see inside "WaitForUpdatedNetwork").
    // In either case, the network will be updated and used when available, even mid-game (detected and updated in network.py, observed and handled in SelfPlay.cpp).
    const int checkpoint = state.Checkpoint();
    if (!state.HasRole(RoleType_Train))
    {
        WaitForUpdatedNetwork(state);
        return;
    }

    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[state.Stage().Stage] << "][" << NetworkTypeNames[state.Stage().Target] << "][" << checkpoint << "]" << std::endl;

    // Save the network. It logs enough internally.
    state.workerGroup->controllerWorker->SaveNetwork(state.network, state.Stage().Target, checkpoint);
}

void ChessCoachTrain::StageSaveSwa(const TrainingState& state)
{
    // If this machine isn't a trainer then we may need to wait for an updated network, or we may be able to keep playing (see inside "WaitForUpdatedNetwork").
    // In either case, the network will be updated and used when available, even mid-game (detected and updated in network.py, observed and handled in SelfPlay.cpp).
    const int checkpoint = state.Checkpoint();
    if (!state.HasRole(RoleType_Train))
    {
        WaitForUpdatedNetwork(state);
        return;
    }

    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[state.Stage().Stage] << "][" << NetworkTypeNames[state.Stage().Target] << "][" << checkpoint << "]" << std::endl;

    // Save the network. It logs enough internally.
    state.workerGroup->controllerWorker->SaveSwaNetwork(state.network, state.Stage().Target, checkpoint);
}

void ChessCoachTrain::WaitForUpdatedNetwork(const TrainingState& state)
{
    // If "wait_for_updated_network" is *false*, return immediately and start playing the next set of games,
    // *except* when (a) this is the first network and we're generating uniform predictions (don't generate too much junk)
    // or (b) when we don't see any network weights to use (e.g. switching to teacher-only-prediction-with-SWA mid-training).
    if (!Config::Network.SelfPlay.WaitForUpdatedNetwork && (state.networkNumber > 1))
    {
        // Check for at least one set of network weights to use by faking a TrainingState.
        TrainingState imitateFirstNetwork = state;
        imitateFirstNetwork.networkNumber = 1;
        if (IsStageComplete(imitateFirstNetwork))
        {
            std::cout << "Not waiting: [" << StageTypeNames[state.Stage().Stage] << "][" << NetworkTypeNames[state.Stage().Target] << "][" << state.Checkpoint() << "]" << std::endl;
            return;
        }
    }

    // We have to wait here until we see someone else save this network type + checkpoint.
    std::cout << "Waiting: [" << StageTypeNames[state.Stage().Stage] << "][" << NetworkTypeNames[state.Stage().Target] << "][" << state.Checkpoint() << "]" << std::endl;
    while (!IsStageComplete(state))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(Config::Network.Training.WaitMilliseconds));
    }
    return;
}

void ChessCoachTrain::StageStrengthTest(const TrainingState& state)
{
    // If this machine isn't a trainer, skip Strength testing.
    if (!state.HasRole(RoleType_Train))
    {
        return;
    }

    // Log the stage info in a consistent format.
    std::cout << "Stage: [" << StageTypeNames[state.Stage().Stage] << "][" << NetworkTypeNames[state.Stage().Target] << "][" << state.Checkpoint() << "]" << std::endl;

    // Strength-test the engine every "StrengthTestInterval" steps.
    assert(Config::Network.Training.StrengthTestInterval >= Config::Network.Training.CheckpointInterval);
    assert((Config::Network.Training.StrengthTestInterval % Config::Network.Training.CheckpointInterval) == 0);
    if ((state.Checkpoint() % Config::Network.Training.StrengthTestInterval) != 0)
    {
        return;
    }

    // Start up a fresh worker group to avoid disturbing paused self-play games.
    // Use 1*16 parallelism for STS for now for consistency/comparability with results from earlier training runs,
    // since higher parallelism causes large score/rating drop in the 200 ms STS budget (still needs investigation).
    const int stsThreads = 1;
    const int stsParallelism = 16;
    WorkerGroup strengthTestWorkerGroup;
    strengthTestWorkerGroup.Initialize(state.network, state.storage, state.Stage().Target,
        stsThreads, stsParallelism, &SelfPlayWorker::LoopStrengthTest);

    // Strength-test the network.
    strengthTestWorkerGroup.controllerWorker->StrengthTestNetwork(
        strengthTestWorkerGroup.workCoordinator.get(), state.network, state.Stage().Target, state.Checkpoint());
    const std::string markerRelativePath = state.StrengthTestMarkerRelativePath();
    if (!markerRelativePath.empty())
    {
        state.network->SaveFile(markerRelativePath, "");
    }

    // Shut down and wait for the strength test worker group.
    strengthTestWorkerGroup.ShutDown();
}

void ChessCoachTrain::ValidateSchedule(const TrainingState& /* state */)
{
    for (const StageConfig& stage : Config::Network.Training.Stages)
    {
        if (stage.Stage == StageType_Train)
        {
            // Calculate using floats: need to avoid 32-bit integer overflow, and need to end up with a ratio.
            const float positionsPerGame = 135.f; // Estimate
            const float sampleRatio =
                (static_cast<float>(Config::Network.Training.Steps) * Config::Network.Training.BatchSize) // total samples
                / (positionsPerGame * Config::Network.Training.NumGames); // total positions
            if (sampleRatio >= 1.f)
            {
                throw std::invalid_argument("Invalid training schedule; sample ratio is too high; increase num_games and window");
            }
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
    for (stateInOut.stageIndex = static_cast<int>(Config::Network.Training.Stages.size()) - 1;stateInOut.stageIndex >= 0; stateInOut.stageIndex--)
    {
        if (IsStageComplete(stateInOut))
        {
            break;
        }
    }

    // If no stages were complete then we increment "stageIndexOut" from -1 to 0 and run them all.
    // If all stages were complete then we roll over and move on to the next network.
    if (++stateInOut.stageIndex >= Config::Network.Training.Stages.size())
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
        state.network->GetNetworkInfo(NetworkType_Teacher, nullptr, nullptr, &trainingChunkCount, nullptr);

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
        state.network->GetNetworkInfo(state.Stage().Target, &networkStepCount, nullptr, nullptr, nullptr);
        return (networkStepCount >= state.Checkpoint());
    }
    case StageType_SaveSwa:
    {
        // Look for the teacher or student network in storage.
        int networkSwaStepCount;
        state.network->GetNetworkInfo(state.Stage().Target, nullptr, &networkSwaStepCount, nullptr, nullptr);
        return (networkSwaStepCount >= state.Checkpoint());
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
    // The window will grow until reaching the desired size, then slide.
    const int windowMax = (state.networkNumber * (Config::Network.Training.NumGames / state.networkCount));
    const int windowMin = std::max(0, windowMax - Config::Network.Training.WindowSize);

    // Min is inclusive, max is exclusive, both 0-based.
    return { windowMin, windowMax };
}

const StageConfig& TrainingState::Stage() const
{
    return Config::Network.Training.Stages[stageIndex];
}

int TrainingState::Checkpoint() const
{
    return (networkNumber * Config::Network.Training.CheckpointInterval);
}

int TrainingState::Step() const
{
    return (((networkNumber - 1) * Config::Network.Training.CheckpointInterval) + 1);
}

bool TrainingState::HasRole(RoleType role) const
{
    return (Config::Network.Role & role);
}

std::string TrainingState::StrengthTestMarkerRelativePath() const
{
    std::string relativePath;
    network->GetNetworkInfo(Stage().Target, nullptr, nullptr, nullptr, &relativePath);
    if (relativePath.empty())
    {
        // No path, not saved in storage yet.
        return relativePath;
    }

    // Always use a forward slash for potential compatibility with gs://
    relativePath += '/';
    relativePath += Config::Misc.Paths_StrengthTestMarkerPrefix;
    relativePath += NetworkTypeNames[Stage().Target];

    return relativePath;
}