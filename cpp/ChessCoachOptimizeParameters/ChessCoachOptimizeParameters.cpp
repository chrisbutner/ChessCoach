#include <thread>

#include <ChessCoach/ChessCoach.h>
#include <ChessCoach/WorkerGroup.h>

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
    if (Config::Misc.Optimization_Mode == "epd")
    {
        // Prepare to run strength tests to evaluate parameters.
        std::unique_ptr<INetwork> network(CreateNetwork());
        WorkerGroup workerGroup;
        workerGroup.Initialize(network.get(), nullptr /* storage */, Config::Network.SelfPlay.PredictionNetworkType,
            Config::Misc.Search_SearchThreads, Config::Misc.Search_SearchParallelism, &SelfPlayWorker::LoopStrengthTest);
        InitializePythonModule(nullptr /* storage */, &workerGroup);

        // Call in to Python.
        network->OptimizeParameters();
    }
    else if (Config::Misc.Optimization_Mode == "tournament")
    {
        // This process needs to avoid accessing alpha TPUs in tournament mode so that ChessCoachUci is able to do so.
        // So, manually call in to optimization.py rather than loading network.py and initializing TensorFlow.
        //
        // If this happened in more situations, a complicated redesign/breakdown of the API could be worth it, but currently not.
        OptimizeParameters();
    }
    else
    {
        throw std::runtime_error("Unexpected parameter optimization mode (expected 'epd' or 'tournament'): " + Config::Misc.Optimization_Mode);
    }
}