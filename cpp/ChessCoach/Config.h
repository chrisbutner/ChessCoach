#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <string>
#include <vector>
#include <map>

enum StageType
{
    StageType_Play,
    StageType_Train,
    StageType_TrainCommentary,
    StageType_Save,
    StageType_StrengthTest,

    StageType_Count,
};
constexpr const char* StageTypeNames[StageType_Count] = { "Play", "Train", "TrainCommentary", "Save", "StrengthTest" };
static_assert(StageType_Count == 5);

enum NetworkType
{
    NetworkType_Teacher,
    NetworkType_Student,

    NetworkType_Count,
};
constexpr const char* NetworkTypeNames[NetworkType_Count] = { "Teacher", "Student" };
static_assert(NetworkType_Count == 2);

enum GameType
{
    GameType_Supervised = 0, // Hard-coded in training.py
    GameType_Training = 1, // Hard-coded in training.py
    GameType_Validation,

    GameType_Count,
};
constexpr const char* GameTypeNames[GameType_Count] = { "Supervised", "Training", "Validation" };
static_assert(GameType_Count == 3);

enum RoleType {
    RoleType_None = 0,
    RoleType_Train = (1 << 0),
    RoleType_Play = (1 << 1),
};

enum PredictionStatus {
    PredictionStatus_None = 0,
    PredictionStatus_UpdatedNetwork = (1 << 0),
};

struct Window
{
    // E.g. for 5000 games per network, with current window size of 10000, on network #4,
    // set TrainingGameMin=10000, TrainingGameMax=20000.
    int TrainingGameMin; // Inclusive, 0-based
    int TrainingGameMax; // Exclusive, 0-based
};

struct StageConfig
{
    StageType Stage;
    NetworkType Target;
    GameType Type;
    int WindowSizeStart;
    int WindowSizeFinish;
    int NumGames;
};

struct TrainingConfig
{
    int BatchSize;
    int CommentaryBatchSize;
    int Steps;
    int WarmupSteps;
    int PgnInterval;
    int ValidationInterval;
    int CheckpointInterval;
    int StrengthTestInterval;
    int WaitMilliseconds;
    std::vector<StageConfig> Stages;
    std::string VocabularyFilename;
    std::string GamesPathSupervised;
    std::string GamesPathTraining;
    std::string GamesPathValidation;
    std::string CommentaryPathSupervised;
    std::string CommentaryPathTraining;
    std::string CommentaryPathValidation;
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

    float LinearExplorationRate;
    float LinearExplorationBase;
    float VirtualLossCoefficient;
    float MovingAverageBuild;
    float MovingAverageCap;
    float BackpropagationPuctThreshold;

    float NetworkUpdateCheckIntervalSeconds;
    bool WaitForUpdatedNetwork;
};

struct NetworkConfig
{
    std::string Name;
    RoleType Role;
    TrainingConfig Training;
    SelfPlayConfig SelfPlay;
};

struct MiscConfig
{
    // Prediction cache
    int PredictionCache_RequestGibibytes;
    int PredictionCache_MinGibibytes;
    int PredictionCache_MaxPly;

    // Time control
    int TimeControl_SafetyBufferMilliseconds;
    int TimeControl_FractionOfRemaining;

    // Search
    int Search_SearchThreads;
    int Search_SearchParallelism;
    int Search_GuiUpdateIntervalNodes;

    // Storage
    int Storage_GamesPerChunk;
    
    // Paths
    std::string Gcloud_Bucket;
    std::string Gcloud_Prefix;
    std::string Paths_Networks;
    std::string Paths_TensorBoard;
    std::string Paths_Logs;
    std::string Paths_Pgns;
    std::string Paths_StrengthTestMarkerPrefix;

    // Optimization
    std::string Optimization_Epd;
    int Optimization_Nodes;
    int Optimization_FailureNodes;
    int Optimization_PositionLimit;
};

class Config
{
public:

    static constexpr const char StartingPosition[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

public:

    static NetworkConfig Network;
    static MiscConfig Misc;

public:

    static void Initialize();
    static void UpdateParameters(const std::map<std::string, float>& parameters);

private:

    static void Parse(const std::map<std::string, float>& parameters);
};

#endif // _CONFIG_H_