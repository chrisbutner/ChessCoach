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
    StageType_SaveSwa,
    StageType_StrengthTest,

    StageType_Count,
};
constexpr const char* StageTypeNames[StageType_Count] = { "Play", "Train", "TrainCommentary", "Save", "SaveSwa", "StrengthTest" };
static_assert(StageType_Count == 6);

enum NetworkType
{
    NetworkType_Teacher,
    NetworkType_Student,

    NetworkType_Count,
};
constexpr const char* NetworkTypeKeys[NetworkType_Count] = { "teacher", "student" };
constexpr const char* NetworkTypeNames[NetworkType_Count] = { "Teacher", "Student" };
static_assert(NetworkType_Count == 2);

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
};

struct TrainingConfig
{
    int NumGames;
    int WindowSize;
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
    std::string GamesPathTraining;
    std::string GamesPathValidation;
    std::string CommentaryPathTraining;
    std::string CommentaryPathValidation;
};

struct SelfPlayConfig
{
    NetworkType NetworkType;
    std::string NetworkWeights;

    int NumWorkers;
    int PredictionBatchSize;

    int NumSampingMoves;
    int MaxMoves;
    int NumSimulations;

    float RootDirichletAlpha;
    float RootExplorationFraction;

    float ExplorationRateBase;
    float ExplorationRateInit;

    bool UseSblePuct;
    float LinearExplorationRate;
    float LinearExplorationBase;
    float VirtualLossCoefficient;
    float MovingAverageBuild;
    float MovingAverageCap;
    float BackpropagationPuctThreshold;

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

    // UCI options
    std::map<std::string, std::string> UciOptions;
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
    static void Update(const std::map<std::string, float>& floatUpdates, const std::map<std::string, std::string>& stringUpdates, const std::map<std::string, bool>& boolUpdates);
    static void LookUp(std::map<std::string, int>& intLookups, std::map<std::string, std::string>& stringLookups, std::map<std::string, bool>& boolLookups);

};

#endif // _CONFIG_H_