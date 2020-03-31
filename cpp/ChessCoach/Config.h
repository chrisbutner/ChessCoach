#ifndef _CONFIG_H_
#define _CONFIG_H_

#define DEBUG_MCTS 0

struct Config
{
    static int BatchSize;
    static float TrainingFactor;
    static int TrainingSteps;
    static int CheckpointInterval;
    static int WindowSize;
    static int SelfPlayGames;

    static int NumSampingMoves;
    static int MaxMoves;
    static int NumSimulations;

    static float RootDirichletAlpha;
    static float RootExplorationFraction;

    static float PbCBase;
    static float PbCInit;
};

#endif // _CONFIG_H_