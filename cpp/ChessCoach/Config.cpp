#include "Config.h"

int Config::BatchSize = 2048; // OOM on GTX 1080 @ 4096
float Config::TrainingFactor = 4096.f / Config::BatchSize; // Increase training to compensate for lower batch size.
int Config::TrainingSteps = static_cast<int>(700000 * Config::TrainingFactor);
int Config::CheckpointInterval = 10; // 1000; // Currently training about 10x as slowly as AlphaZero, but self-play 100x+, so reduce accordingly.
int Config::WindowSize = 1000000;
int Config::SelfPlayGames = 44000000;

int Config::NumSampingMoves = 30;
int Config::MaxMoves = 512;
int Config::NumSimulations = 800;

float Config::RootDirichletAlpha = 0.3f;
float Config::RootExplorationFraction = 0.25f;

float Config::PbCBase = 19652.f;
float Config::PbCInit = 1.25f;