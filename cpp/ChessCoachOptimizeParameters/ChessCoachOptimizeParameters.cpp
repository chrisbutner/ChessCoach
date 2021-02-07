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
    // Prepare to run strength tests to evaluate parameters.
    // Use the faster student network for strength testing.
    std::unique_ptr<INetwork> network(CreateNetwork());
    WorkerGroup workerGroup;
    workerGroup.Initialize(network.get(), NetworkType_Student, Config::Misc.Search_SearchThreads, Config::Misc.Search_SearchParallelism, &SelfPlayWorker::LoopStrengthTest);
    InitializePythonModule(nullptr /* storage */, &workerGroup);

    // Call in to Python.
    network->OptimizeParameters();
}