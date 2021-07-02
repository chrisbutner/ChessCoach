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
        InitializePythonModule(nullptr /* storage */, network.get(), nullptr /* workerGroup */);

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
        throw ChessCoachException("Unexpected parameter optimization mode (expected 'epd' or 'tournament'): " + Config::Misc.Optimization_Mode);
    }
}