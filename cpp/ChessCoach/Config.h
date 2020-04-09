#ifndef _CONFIG_H_
#define _CONFIG_H_

#define DEBUG_MCTS 0

struct Config
{
    static const int InputPreviousMoveCount = 8;
    static const int MaxBranchMoves = 80;

    // 8 GiB cache with 512-byte entries gives 16777216 entries.
    // perft(5) is 4,865,609. perft(6) is 119,060,324.
    static const int PredictionCacheSizeGb = 8;
    static const int MaxPredictionCachePly = 6;

    static const int BatchSize = 2048; // OOM on GTX 1080 @ 4096;
    static const float TrainingFactor;
    static const int TrainingSteps;
    static const int CheckpointInterval;
    static const int WindowSize;
    static const int SelfPlayGames;

    static const int NumSampingMoves;
    static const int MaxMoves = 512;
    static const int NumSimulations;

    static const float RootDirichletAlpha;
    static const float RootExplorationFraction;

    static const float PbCBase;
    static const float PbCInit;

    static const char* StartingPosition;

    static const int SelfPlayWorkerCount =
#ifdef _DEBUG
        1;
#else
        4;
#endif

    static const int PredictionBatchSize =
#ifdef _DEBUG
        1;
#else
        128;
#endif

    static const int MaxNodesPerThread = (2 * MaxMoves * MaxBranchMoves * PredictionBatchSize);
};

#endif // _CONFIG_H_