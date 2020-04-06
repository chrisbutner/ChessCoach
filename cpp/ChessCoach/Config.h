#ifndef _CONFIG_H_
#define _CONFIG_H_

#define DEBUG_MCTS 0

struct Config
{
    static const int BatchSize = 2048; // OOM on GTX 1080 @ 4096;
    static const float TrainingFactor;
    static const int TrainingSteps;
    static const int CheckpointInterval;
    static const int WindowSize;
    static const int SelfPlayGames;

    static const int NumSampingMoves;
    static const int MaxMoves;
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
        256;
#endif

};

#endif // _CONFIG_H_