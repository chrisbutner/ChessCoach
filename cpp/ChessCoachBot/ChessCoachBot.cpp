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

#include <ChessCoach/ChessCoach.h>
#include <ChessCoach/WorkerGroup.h>
#include <ChessCoach/Syzygy.h>

class ChessCoachBot : public ChessCoach
{
public:

    void Work();
};

int main()
{
    ChessCoachBot chessCoachBot;

    chessCoachBot.PrintExceptions();
    chessCoachBot.Initialize();

    chessCoachBot.Work();

    chessCoachBot.Finalize();

    return 0;
}

void ChessCoachBot::Work()
{
    // Add additional safety buffer to compensate for network calls, commentary,
    // amortizing ponder pruning, and lack of progress in no-increment games.
    Config::Misc.TimeControl_SafetyBufferMilliseconds += 1500;

    // Use the search loop and print principle variations periodically.
    std::unique_ptr<INetwork> network(CreateNetwork());
    WorkerGroup workerGroup;
    workerGroup.Initialize(network.get(), nullptr /* storage */, Config::Network.SelfPlay.PredictionNetworkType,
        Config::Misc.Search_SearchThreads, Config::Misc.Search_SearchParallelism, &SelfPlayWorker::LoopSearch);

    // Let the bot call back into Python to play a move after searching.
    InitializePythonModule(nullptr /* storage */, network.get(), &workerGroup);

    // Initialize tablebases.
    Syzygy::Reload();

    // Warm up commentary.
    workerGroup.controllerWorker->CommentOnPosition(network.get());

    // Call in to Python.
    network->RunBot();
}
