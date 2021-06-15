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
    // Add additional safety buffer to compensate for network calls and commentary.
    Config::Misc.TimeControl_SafetyBufferMilliseconds += 1000;

    // Turn on move diversity at low temperature even if it's weakening.
    Config::Network.SelfPlay.MoveDiversityTemperature = std::max(Config::Network.SelfPlay.MoveDiversityTemperature, 0.5f);

    // Use the lighter-weight strength test loop, rather than chattier search loop. 
    std::unique_ptr<INetwork> network(CreateNetwork());
    WorkerGroup workerGroup;
    workerGroup.Initialize(network.get(), nullptr /* storage */, Config::Network.SelfPlay.PredictionNetworkType,
        Config::Misc.Search_SearchThreads, Config::Misc.Search_SearchParallelism, &SelfPlayWorker::LoopStrengthTest);

    // Let the bot call back into Python to play a move after searching.
    InitializePythonModule(nullptr /* storage */, network.get(), &workerGroup);

    // Initialize tablebases.
    Syzygy::Reload();

    // Warm up commentary.
    workerGroup.controllerWorker->CommentOnPosition(network.get());

    // Call in to Python.
    network->RunBot();
}
