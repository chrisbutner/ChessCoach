// ChessCoach, a neural network-based chess engine capable of natural-language commentary
// Copyright 2021 Chris Butner
//
// ChessCoach is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ChessCoach is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ChessCoach. If not, see <https://www.gnu.org/licenses/>.

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
    std::string CommentaryPath;
};

struct SelfPlayConfig
{
    NetworkType PredictionNetworkType;
    std::string NetworkWeights;
    bool AllowUniform;

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
    int EliminationBaseExponent;
    float MoveDiversityValueDeltaThreshold;
    float MoveDiversityTemperature;
    int MoveDiversityPlies;
    int TranspositionProgressThreshold;
    int ProgressDecayDivisor;
    int MinimaxMaterialMaximum;
    int MinimaxVisitsRecurse;
    float MinimaxVisitsIgnore;

    bool WaitForUpdatedNetwork;
};

struct NetworkConfig
{
    std::string Name;
    RoleType Role;
    TrainingConfig Training;
    SelfPlayConfig SelfPlay;
};

struct UciOptionConfig
{
    std::string Type;
    int Min;
    int Max;
};

struct MiscConfig
{
    // Prediction cache
    int PredictionCache_SizeMebibytes;
    int PredictionCache_MaxPly;

    // Time control
    int TimeControl_SafetyBufferMilliseconds;
    int TimeControl_FractionOfRemaining;
    int TimeControl_AbsoluteMinimumMilliseconds;

    // Search
    int Search_SearchThreads;
    int Search_SearchParallelism;
    int Search_SlowstartNodes;
    int Search_SlowstartThreads;
    int Search_SlowstartParallelism;
    int Search_GuiUpdateIntervalNodes;

    // Bot
    int Bot_CommentaryMinimumRemainingMilliseconds;
    int Bot_PonderBufferMaxMilliseconds;
    int Bot_PonderBufferMinMilliseconds;
    float Bot_PonderBufferProportion;

    // Storage
    int Storage_GamesPerChunk;
    
    // Paths
    std::string Paths_Networks;
    std::string Paths_TensorBoard;
    std::string Paths_Logs;
    std::string Paths_Pgns;
    std::string Paths_Syzygy;
    std::string Paths_StrengthTestMarkerPrefix;

    // Optimization
    std::string Optimization_Mode;
    std::string Optimization_Epd;
    int Optimization_EpdMovetimeMilliseconds;
    int Optimization_EpdNodes;
    int Optimization_EpdFailureNodes;
    int Optimization_EpdPositionLimit;

    // UCI options
    std::map<std::string, UciOptionConfig> UciOptions;
};

class Config
{
public:

    static NetworkConfig Network;
    static MiscConfig Misc;

public:

    static void Initialize();
    static void Update(const std::map<std::string, int>& intUpdates, const std::map<std::string, float>& floatUpdates,
        const std::map<std::string, std::string>& stringUpdates, const std::map<std::string, bool>& boolUpdates);
    static void LookUp(std::map<std::string, int>& intLookups, std::map<std::string, float>& floatLookups,
        std::map<std::string, std::string>& stringLookups, std::map<std::string, bool>& boolLookups);

};

#endif // _CONFIG_H_