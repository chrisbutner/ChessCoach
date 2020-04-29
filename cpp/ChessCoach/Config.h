#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <array>

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
};

namespace Ucb
{
    static constexpr float MateN(int mateN)
    {
        return (1.f / (1 << mateN));
    }

    static_assert(MateN(2) == 0.25f);
    static_assert(MateN(3) == 0.125f);
}

class Config
{
public:

    static const int InputPreviousMoveCount = 8;
    static const int MaxBranchMoves = 80;
    static constexpr const char StartingPosition[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    static constexpr const std::array<float, 25> UcbMateTerm = { 0.f, 1.f, Ucb::MateN(2), Ucb::MateN(3), Ucb::MateN(4),
                                      Ucb::MateN(5), Ucb::MateN(6), Ucb::MateN(7), Ucb::MateN(8), Ucb::MateN(9),
                                      Ucb::MateN(10), Ucb::MateN(11), Ucb::MateN(12), Ucb::MateN(13), Ucb::MateN(14),
                                      Ucb::MateN(15), Ucb::MateN(16), Ucb::MateN(17), Ucb::MateN(18), Ucb::MateN(19),
                                      Ucb::MateN(20), Ucb::MateN(21), Ucb::MateN(22), Ucb::MateN(23), Ucb::MateN(24), };

public:

    static NetworkConfig TrainingNetwork;
    static NetworkConfig UciNetwork;
    static MiscConfig Misc;

public:

    static void Initialize();
};

#endif // _CONFIG_H_