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

#include <filesystem>

#include <tclap/CmdLine.h>

#include <ChessCoach/ChessCoach.h>
#include <ChessCoach/WorkerGroup.h>

class ChessCoachStrengthTest : public ChessCoach
{
public:

    ChessCoachStrengthTest(const std::filesystem::path& epdPath,
        int moveTimeMs, int nodes, int failureNodes, int positionLimit, float slopeArg, float interceptArg);

    void Initialize();

    void StrengthTest();

private:

    std::filesystem::path _epdPath;
    int _moveTimeMs;
    int _nodes;
    int _failureNodes;
    int _positionLimit;
    float _slope;
    float _intercept;
};

int main(int argc, char* argv[])
{
    std::string epdPath;
    int moveTimeMs;
    int nodes;
    int failureNodes;
    int positionLimit;
    float slope;
    float intercept;

    try
    {
        TCLAP::CmdLine cmd("ChessCoachStrengthTest: Tests ChessCoach using a provided .epd file to generate a score and optionally a rating", ' ', "0.9");

        TCLAP::ValueArg<std::string> epdArg("e", "epd", "Path to the .epd file to test", true /* req */, "", "string");
        TCLAP::ValueArg<int> moveTimeArg("t", "movetime", "Move time per position (ms)", false /* req */, 0, "whole number");
        TCLAP::ValueArg<int> nodesArg("o", "nodes", "Nodes per position", false /* req */, 0, "whole number");
        TCLAP::ValueArg<int> failureNodesArg("u", "failure", "Failure nodes per position", false /* req */, 0, "whole number");
        TCLAP::ValueArg<int> positionLimitArg("l", "limit", "Number of positions in the EPD to run", false /* req */, 0, "whole number");
        TCLAP::ValueArg<float> slopeArg("s", "slope", "Slope for linear rating calculation based on score", false /* req */, 0.f, "decimal");
        TCLAP::ValueArg<float> interceptArg("i", "intercept", "Intercept for linear rating calculation based on score", false /* req */, 0.f, "decimal");

        // Usage/help seems to reverse this order.
        cmd.add(interceptArg);
        cmd.add(slopeArg);
        cmd.add(positionLimitArg);
        cmd.add(failureNodesArg);
        cmd.add(nodesArg);
        cmd.add(moveTimeArg);
        cmd.add(epdArg);

        cmd.parse(argc, argv);

        epdPath = epdArg.getValue();
        moveTimeMs = moveTimeArg.getValue();
        nodes = nodesArg.getValue();
        failureNodes = failureNodesArg.getValue();
        positionLimit = positionLimitArg.getValue();
        slope = slopeArg.getValue();
        intercept = interceptArg.getValue();
    }
    catch (TCLAP::ArgException& e)
    {
        std::cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
        return 1;
    }

    ChessCoachStrengthTest strengthTest(epdPath, moveTimeMs, nodes, failureNodes, positionLimit, slope, intercept);

    strengthTest.PrintExceptions();
    strengthTest.Initialize();

    strengthTest.StrengthTest();

    strengthTest.Finalize();

    return 0;
}

ChessCoachStrengthTest::ChessCoachStrengthTest(const std::filesystem::path& epdPath,
    int moveTimeMs, int nodes, int failureNodes, int positionLimit, float slope, float intercept)
    : _epdPath(epdPath)
    , _moveTimeMs(moveTimeMs)
    , _nodes(nodes)
    , _failureNodes(failureNodes)
    , _positionLimit(positionLimit)
    , _slope(slope)
    , _intercept(intercept)
{
}

void ChessCoachStrengthTest::Initialize()
{
    // Suppress all Python/TensorFlow output so that output is readable, especially when running
    // multiple strength tests back-to-back.
    Platform::SetEnvironmentVariable("CHESSCOACH_SILENT", "1");

    InitializePython();
    InitializeStockfish();
    InitializeChessCoach();
    InitializePredictionCache();
}

static void PrintProgress(const std::string& fen, const std::string& target, const std::string& chosen, int score, int total, int nodeScore)
{
    std::cout << fen << ", " << target << ", " << chosen << ", " << score << ", " << total << ", " << nodeScore << std::endl;
}

void ChessCoachStrengthTest::StrengthTest()
{
    std::cout << "Preparing network..." << std::endl;

    std::unique_ptr<INetwork> network(CreateNetwork());
    WorkerGroup workerGroup;
    workerGroup.Initialize(network.get(), nullptr /* storage */, Config::Network.SelfPlay.PredictionNetworkType,
        Config::Misc.Search_SearchThreads, Config::Misc.Search_SearchParallelism, &SelfPlayWorker::LoopStrengthTest);

    std::cout << "Testing " << _epdPath.stem() << "...\n\nPosition, Target, Chosen, Score, Total, Nodes" << std::endl;

    const auto start = std::chrono::high_resolution_clock::now();

    const auto [score, total, positions, totalNodesRequired] = workerGroup.controllerWorker->StrengthTestEpd(workerGroup.workCoordinator.get(), _epdPath,
        _moveTimeMs, _nodes, _failureNodes, _positionLimit, PrintProgress);

    const float secondsTaken = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count();

    std::cout << "\nTested " << positions << " positions in " << secondsTaken << " seconds." << std::endl;
    std::cout << "Nodes required: " << totalNodesRequired << std::endl;
    std::cout << "Score: " << score << " out of " << total << std::endl;

    // Use score/positions (not score/total) with slope and intercept to match STS.
    if ((_slope != 0.f) || (_intercept != 0.f))
    {
        const int rating = static_cast<int>((_slope * score / positions) + _intercept);
        std::cout << "Rating: " << rating << std::endl;
    }

    workerGroup.ShutDown();
}