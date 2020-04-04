#include "Config.h"

const int Config::BatchSize = 2048; // OOM on GTX 1080 @ 4096
const float Config::TrainingFactor = 4096.f / Config::BatchSize; // Increase training to compensate for lower batch size.
const int Config::TrainingSteps = static_cast<int>(700000 * Config::TrainingFactor);
const int Config::CheckpointInterval = 10; // 1000; // Currently training about 10x as slowly as AlphaZero, but self-play 100x+, so reduce accordingly.
const int Config::WindowSize = 1000000;
const int Config::SelfPlayGames = 44000000;

const int Config::NumSampingMoves = 30;
const int Config::MaxMoves = 512;
const int Config::NumSimulations = 800;

const float Config::RootDirichletAlpha = 0.3f;
const float Config::RootExplorationFraction = 0.25f;

const float Config::PbCBase = 19652.f;
const float Config::PbCInit = 1.25f;

const char* Config::StartingPosition = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";