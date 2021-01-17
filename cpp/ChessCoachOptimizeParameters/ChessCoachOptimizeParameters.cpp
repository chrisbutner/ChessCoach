#include <thread>

#include <ChessCoach/ChessCoach.h>
#include <ChessCoach/SelfPlay.h>

class ChessCoachOptimizeParameters : public ChessCoach
{
public:

    void InitializeLight();
    void FinalizeLight();
    void Run();
};

int main()
{
    ChessCoachOptimizeParameters optimize;

    optimize.PrintExceptions();
    optimize.InitializeLight();

    optimize.Run();

    optimize.FinalizeLight();

    return 0;
}

void ChessCoachOptimizeParameters::InitializeLight()
{
    InitializePython();
    InitializeStockfish();
    InitializeChessCoach();
}

void ChessCoachOptimizeParameters::FinalizeLight()
{
    FinalizePython();
    FinalizeStockfish();
}

void ChessCoachOptimizeParameters::Run()
{
    // Prepare to run strength tests to evaluate parameters. Use the UCI network.
    std::unique_ptr<INetwork> network(CreateNetwork(Config::UciNetwork));
    SelfPlayWorker worker(Config::UciNetwork, nullptr /* storage */);
    InitializePythonModule(nullptr /* storage */, &worker, network.get());

    // Call in to Python.
    network->OptimizeParameters();
}