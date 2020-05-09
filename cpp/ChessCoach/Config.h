#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <string>

#define DEBUG_MCTS 0
#define SAMPLE_BATCH_FIXED 0

struct TrainingConfig
{
    int BatchSize;
    int Steps;
    int PgnInterval;
    int ValidationInterval;
    int CheckpointInterval;
    int StrengthTestInterval;
    int NumGames;
    int WindowSize;
    std::string GamesPathTraining;
    std::string GamesPathValidation;
};

struct SelfPlayConfig
{
    int NumWorkers;
    int PredictionBatchSize;

    int NumSampingMoves;
    int MaxMoves;
    int NumSimulations;

    float RootDirichletAlpha;
    float RootExplorationFraction;

    float ExplorationRateBase;
    float ExplorationRateInit;
};

struct NetworkConfig
{
    std::string Name;
    TrainingConfig Training;
    SelfPlayConfig SelfPlay;
};

struct MiscConfig
{
    // Prediction cache
    int PredictionCache_SizeGb;
    int PredictionCache_MaxPly;

    // Time control
    int TimeControl_SafetyBufferMs;
    int TimeControl_FractionOfRemaining;

    // Search
    int Search_MctsParallelism;

    // Storage
    int Storage_MaxGamesPerFile;
    
    // Paths
    std::string Paths_Networks;
    std::string Paths_TensorBoard;
    std::string Paths_Logs;
    std::string Paths_Pgns;
};

class Config
{
public:

    static constexpr const char StartingPosition[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

public:

    static NetworkConfig TrainingNetwork;
    static NetworkConfig UciNetwork;
    static MiscConfig Misc;

public:

    static void Initialize();
};

#endif // _CONFIG_H_