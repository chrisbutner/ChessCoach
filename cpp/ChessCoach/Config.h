#ifndef _CONFIG_H_
#define _CONFIG_H_

#define DEBUG_MCTS 0

struct Config
{
    static const int BatchSize;
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
};

#endif // _CONFIG_H_