#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <string>
#include <vector>

#define SAMPLE_BATCH_FIXED 0

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
    GameType_Supervised,
    GameType_Training,
    GameType_Validation,

    GameType_Count,
};
constexpr const char* GameTypeNames[GameType_Count] = { "Supervised", "Training", "Validation" };
static_assert(GameType_Count == 3);

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
    int PgnInterval;
    int ValidationInterval;
    int CheckpointInterval;
    int StrengthTestInterval;
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

    float NetworkUpdateCheckIntervalSeconds;
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
    int PredictionCache_SizeGibibytes;
    int PredictionCache_MaxPly;

    // Time control
    int TimeControl_SafetyBufferMilliseconds;
    int TimeControl_FractionOfRemaining;

    // Search
    int Search_MctsParallelism;

    // Storage
    int Storage_GamesPerChunk;
    
    // Paths
    std::string Gcloud_Bucket;
    std::string Gcloud_Prefix;
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